"""
On-Policy GRPO Training with Tinker

Implements Group Relative Policy Optimization (GRPO) for math reasoning on GSM8K,
running on Tinker's cloud infrastructure while reusing assignment functions.

Based on:
- DeepSeekMath: https://arxiv.org/abs/2402.03300
- DeepSeek-R1: https://arxiv.org/abs/2501.12948
"""

import logging
import random
import re
import sys
import time

import chz
import datasets
import tinker
import torch
from tinker.types.tensor_data import TensorData
from tinker_cookbook import checkpoint_utils
from tinker_cookbook.utils import ml_log

# Import from assignment code
sys.path.append("/Users/chenmoney/Documents/genai/cs336-assignment5")
from tests.adapters import run_compute_group_normalized_rewards

logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class GRPOConfig:
    """Configuration for GRPO training."""

    # Model & Service
    model_name: str = "Qwen/Qwen3-8B"
    base_url: str | None = None
    lora_rank: int = 32

    # Training hyperparameters (from assignment)
    n_grpo_steps: int = 64
    learning_rate: float = 1e-5
    rollout_batch_size: int = (
        256  # Total rollouts per batch (questions = rollout_batch_size // group_size)
    )
    group_size: int = 8  # samples per question
    gradient_accumulation_steps: int = 128  # Not directly used, Tinker handles batching

    # Off-policy training (for efficiency)
    epochs_per_rollout_batch: int = 1  # On-policy=1, off-policy>1 (e.g., 2-4)
    train_batch_size: int = 32  # Batch size for each gradient step (rollouts per batch)

    # GRPO-specific
    advantage_eps: float = 1e-6
    use_std_normalization: bool = True
    cliprange: float = 0.2  # For PPO-style clipping

    # Sampling
    sampling_temperature: float = 1.0
    sampling_max_tokens: int = 2048
    sampling_min_tokens: int = 4
    use_r1_zero_format: bool = True  # Use <think></think><answer></answer> format

    # Logging & Checkpointing
    log_path: str = "tmp_logging/grpo-tinker"
    save_every: int = 12
    eval_every: int = 12
    wandb_project: str | None = "grpo-tinker"
    wandb_name: str | None = "grpo-qwen3-8b-run-off-policy"
    force_fresh_start: bool = True  # Set to True to ignore existing checkpoints


def load_gsm8k_dataset():
    """Load GSM8K from HuggingFace."""
    logger.info("Loading GSM8K dataset...")
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    assert isinstance(dataset, datasets.DatasetDict)
    train_data = dataset["train"]
    test_data = dataset["test"]
    logger.info(
        f"Loaded {len(train_data)} training examples, {len(test_data)} test examples"
    )
    return train_data, test_data


def gsm8k_reward_fn(response: str, ground_truth: str) -> dict[str, float]:
    """
    Compute reward for GSM8K responses.

    GSM8K ground truth format: "...#### 72"
    Response should contain final answer (ideally in \\boxed{} or <answer>)

    Args:
        response: Model's generated response
        ground_truth: GSM8K ground truth with "#### answer" format

    Returns:
        dict with keys: "reward", "format_reward", "answer_reward"
    """
    # Extract final answer from ground truth (after ####)
    gt_match = re.search(r"####\s*(.+)$", ground_truth.strip())
    if not gt_match:
        return {"reward": 0.0, "format_reward": 0.0, "answer_reward": 0.0}

    gt_answer = gt_match.group(1).strip()

    # Try multiple extraction methods for model response
    # 1. Try <answer>...</answer> tags (if using r1_zero format)
    answer_match = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    if answer_match:
        model_answer = answer_match.group(1).strip()
        format_reward = 1.0
    # 2. Try \\boxed{}
    elif "\\boxed{" in response:
        boxed_match = re.search(r"\\boxed\{([^}]+)\}", response)
        model_answer = boxed_match.group(1).strip() if boxed_match else ""
        format_reward = 1.0 if boxed_match else 0.0
    else:
        # Fallback: take last number in response
        numbers = re.findall(r"-?\d+\.?\d*", response)
        model_answer = numbers[-1] if numbers else ""
        format_reward = 0.5 if numbers else 0.0

    # Grade numerical answer
    try:
        model_num = float(model_answer.replace(",", ""))
        gt_num = float(gt_answer.replace(",", ""))
        answer_reward = 1.0 if abs(model_num - gt_num) < 1e-4 else 0.0
    except ValueError:
        # Fallback to string comparison
        answer_reward = 1.0 if model_answer == gt_answer else 0.0

    # Combined reward (format 20%, answer 80%)
    reward = format_reward * 0.2 + answer_reward * 0.8

    return {
        "reward": reward,
        "format_reward": format_reward,
        "answer_reward": answer_reward,
    }


def build_gsm8k_prompt(question: str, use_r1_zero_format: bool = True) -> str:
    """Build prompt for GSM8K question."""
    if use_r1_zero_format:
        # Use <think></think><answer></answer> format
        return f"""A conversation between User and Assistant. The User asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the User with the answer. The reasoning process is enclosed within <think> </think> and answer is enclosed within <answer> </answer> tags, respectively.

User: {question}
Assistant: <think>"""
    else:
        # Simple format with \\boxed{}
        return f"""Solve the following math problem. Provide your final numerical answer inside \\boxed{{}}.

Question: {question}

Solution:"""


def validate_on_test_set(
    sampling_client,
    test_dataset,
    tokenizer,
    use_r1_zero_format: bool,
    n_samples: int = 100,
) -> dict[str, float]:
    """
    Evaluate model on GSM8K test set.

    Args:
        sampling_client: Tinker SamplingClient
        test_dataset: GSM8K test dataset
        tokenizer: Tokenizer for encoding/decoding
        use_r1_zero_format: Whether to use R1-Zero format
        n_samples: Number of test examples to evaluate

    Returns:
        dict with accuracy metrics
    """
    logger.info(f"Running validation on {n_samples} test examples...")
    test_sample = test_dataset.shuffle().select(
        range(min(n_samples, len(test_dataset)))
    )

    correct = 0
    format_correct = 0

    for question, answer in zip(test_sample["question"], test_sample["answer"]):
        prompt = build_gsm8k_prompt(question, use_r1_zero_format)

        try:
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            response = sampling_client.sample(
                prompt=tinker.types.ModelInput.from_ints(prompt_tokens),
                num_samples=1,
                sampling_params=tinker.types.SamplingParams(
                    max_tokens=2048,
                    temperature=0.0,  # Greedy for evaluation
                    stop=["</answer>", "\n\n\n"] if use_r1_zero_format else ["\n\n\n"],
                ),
            ).result()

            # Decode response tokens to text
            response_tokens = response.sequences[0].tokens
            response_text = tokenizer.decode(response_tokens, skip_special_tokens=False)
            reward_dict = gsm8k_reward_fn(response_text, answer)

            if reward_dict["answer_reward"] == 1.0:
                correct += 1
            if reward_dict["format_reward"] == 1.0:
                format_correct += 1

        except Exception as e:
            logger.warning(f"Validation error: {e}")
            continue

    accuracy = correct / n_samples
    format_accuracy = format_correct / n_samples

    logger.info(f"Validation - Accuracy: {accuracy:.1%}, Format: {format_accuracy:.1%}")

    return {
        "accuracy": accuracy,
        "format_accuracy": format_accuracy,
    }


def main(config: GRPOConfig):
    """Main GRPO training loop."""

    # 1. Setup logging
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        wandb_name=config.wandb_name,
        config=config,
        do_configure_logging_module=True,
    )

    # 1.5. Validate hyperparameters
    assert (
        config.rollout_batch_size % config.group_size == 0
    ), f"rollout_batch_size ({config.rollout_batch_size}) must be divisible by group_size ({config.group_size})"

    n_questions = config.rollout_batch_size // config.group_size
    assert (
        config.train_batch_size <= config.rollout_batch_size
    ), f"train_batch_size ({config.train_batch_size}) must be <= rollout_batch_size ({config.rollout_batch_size})"
    assert (
        config.epochs_per_rollout_batch >= 1
    ), f"epochs_per_rollout_batch must be >= 1, got {config.epochs_per_rollout_batch}"
    assert (
        config.rollout_batch_size % config.train_batch_size == 0
    ), f"rollout_batch_size ({config.rollout_batch_size}) must be divisible by train_batch_size ({config.train_batch_size})"

    batches_per_epoch = config.rollout_batch_size // config.train_batch_size
    total_gradient_steps = config.epochs_per_rollout_batch * batches_per_epoch

    logger.info("Hyperparameter validation passed:")
    logger.info(f"  Questions per step: {n_questions}")
    logger.info(f"  Rollouts per question (group_size): {config.group_size}")
    logger.info(f"  Total rollouts per step: {config.rollout_batch_size}")
    logger.info(f"  Train batch size: {config.train_batch_size}")
    logger.info(f"  Batches per epoch: {batches_per_epoch}")
    logger.info(f"  Epochs per rollout batch: {config.epochs_per_rollout_batch}")
    logger.info(f"  Total gradient steps per GRPO step: {total_gradient_steps}")

    # 2. Load dataset
    train_dataset, test_dataset = load_gsm8k_dataset()

    # 3. Setup Tinker clients
    logger.info("Setting up Tinker clients...")
    service_client = tinker.ServiceClient(base_url=config.base_url)

    # Check for resume (unless force_fresh_start is set)
    resume_info = checkpoint_utils.get_last_checkpoint(config.log_path)
    if resume_info and not config.force_fresh_start:
        training_client = service_client.create_training_client_from_state(
            resume_info["state_path"]
        )
        start_step = resume_info.get("loop_state", {}).get("grpo_step", 0)
        logger.info(f"Resuming from GRPO step {start_step}")
    else:
        if resume_info and config.force_fresh_start:
            logger.info(
                f"Found checkpoint at step {resume_info.get('loop_state', {}).get('grpo_step', 0)}, "
                f"but force_fresh_start=True, starting fresh instead"
            )
        training_client = service_client.create_lora_training_client(
            base_model=config.model_name, rank=config.lora_rank
        )
        start_step = 0
        logger.info(f"Starting fresh training with model: {config.model_name}")

    # 4. Setup optimizer and sampling params
    adam_params = tinker.types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    stop_sequences = (
        ["</answer>", "\n\n\n"] if config.use_r1_zero_format else ["\n\n\n"]
    )
    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.sampling_max_tokens,
        temperature=config.sampling_temperature,
        stop=stop_sequences,
    )

    # Load tokenizer for prompt length calculation
    logger.info(f"Loading tokenizer: {config.model_name}")
    tokenizer = training_client.get_tokenizer()

    # 5. Main GRPO loop
    n_questions_per_step = config.rollout_batch_size // config.group_size

    logger.info(f"Starting GRPO training for {config.n_grpo_steps} steps...")
    logger.info(
        f"Rollout batch size: {config.rollout_batch_size} "
        f"({n_questions_per_step} questions × {config.group_size} samples)"
    )
    if config.epochs_per_rollout_batch == 1:
        logger.info("On-policy: Training once per rollout batch (no replay)")
    else:
        logger.info(
            f"Off-policy: {config.epochs_per_rollout_batch} epochs per rollout batch"
        )

    for grpo_step in range(start_step, config.n_grpo_steps):
        t_start = time.time()

        metrics = {
            "progress/grpo_step": grpo_step,
            "progress/done_frac": (grpo_step + 1) / config.n_grpo_steps,
            "optim/lr": config.learning_rate,
        }

        # 5.1. Sample batch of questions
        # Simple approach: cycle through dataset, shuffle when we wrap around
        batch_start = (grpo_step * n_questions_per_step) % len(train_dataset)

        # If we're starting a new epoch, shuffle the dataset
        if batch_start == 0 and grpo_step > 0:
            train_dataset = train_dataset.shuffle()
            logger.info("  Shuffled dataset for new epoch")

        batch_end = min(batch_start + n_questions_per_step, len(train_dataset))
        batch_rows = train_dataset.select(range(batch_start, batch_end))
        actual_n_questions = len(batch_rows)

        logger.info(
            f"Step {grpo_step}/{config.n_grpo_steps}: Processing {actual_n_questions} questions "
            f"({actual_n_questions * config.group_size} rollouts)..."
        )

        # 5.2. Save current policy for sampling
        t_save_start = time.time()
        sampling_path = (
            training_client.save_weights_for_sampler(name=f"step_{grpo_step:06d}")
            .result()
            .path
        )
        sampling_client = service_client.create_sampling_client(
            model_path=sampling_path
        )
        metrics["time/save_weights"] = time.time() - t_save_start

        # 5.3. Generate rollouts (group_size per question)
        t_sample_start = time.time()

        # Prepare all prompts and launch sampling
        batch_futures = []
        batch_prompts = []
        batch_answers = []

        for question, answer in zip(batch_rows["question"], batch_rows["answer"]):
            prompt = build_gsm8k_prompt(question, config.use_r1_zero_format)
            prompt_tokens = tokenizer.encode(prompt, add_special_tokens=True)
            batch_prompts.append(prompt_tokens)
            batch_answers.append(answer)

            # Sample group_size responses per question in a single call (more efficient)
            future = sampling_client.sample(
                prompt=tinker.types.ModelInput.from_ints(prompt_tokens),
                num_samples=config.group_size,  # Sample group_size times per question
                sampling_params=sampling_params,
            )
            batch_futures.append(future)

        # Collect rollout results
        rollout_responses = []
        rollout_ground_truths = []
        rollout_prompt_tokens = []
        rollout_response_tokens = []
        rollout_logprobs = []

        for prompt_tokens, answer, future in zip(
            batch_prompts, batch_answers, batch_futures
        ):
            # prompt_tokens already tokenized earlier (line 325)
            # Reuse to avoid duplicate computation

            # Get result with group_size samples
            sample_result = future.result()

            # Iterate over all samples in the group
            for sequence in sample_result.sequences:
                sampled_tokens = sequence.tokens
                sampled_logprobs = sequence.logprobs

                if sampled_logprobs is None:
                    logger.warning("No logprobs returned! Skipping this sample.")
                    continue

                # Decode response text from tokens
                response_text = tokenizer.decode(
                    sampled_tokens, skip_special_tokens=False
                )

                rollout_responses.append(response_text)
                rollout_ground_truths.append(answer)
                # rollout_prompts not used, removed to avoid duplicate tokenization
                rollout_prompt_tokens.append(prompt_tokens)
                rollout_response_tokens.append(sampled_tokens)
                rollout_logprobs.append(sampled_logprobs)

        n_rollouts = len(rollout_responses)
        metrics["time/sampling"] = time.time() - t_sample_start
        metrics["rollout/n_samples"] = n_rollouts
        logger.info(
            f"  Generated {n_rollouts} rollouts ({config.group_size} per question)"
        )

        # 5.4. Compute rewards using assignment's reward function
        t_reward_start = time.time()

        advantages, raw_rewards, reward_metadata = run_compute_group_normalized_rewards(
            reward_fn=gsm8k_reward_fn,
            rollout_responses=rollout_responses,
            repeated_ground_truths=rollout_ground_truths,
            group_size=config.group_size,
            advantage_eps=config.advantage_eps,
            normalize_by_std=config.use_std_normalization,
        )

        metrics["time/compute_rewards"] = time.time() - t_reward_start
        metrics["reward/mean"] = reward_metadata["mean_reward"]
        metrics["reward/std"] = reward_metadata["std_reward"]
        metrics["reward/min"] = reward_metadata["min_reward"]
        metrics["reward/max"] = reward_metadata["max_reward"]

        logger.info(
            f"  Rewards - Mean: {reward_metadata['mean_reward']:.3f}, "
            f"Std: {reward_metadata['std_reward']:.3f}"
        )

        # 5.5. Build training data (Datum objects)
        t_build_data_start = time.time()
        training_datums = []
        skipped_samples = 0

        for idx, (prompt_tokens, response_tokens, old_logprobs, advantage) in enumerate(
            zip(
                rollout_prompt_tokens,
                rollout_response_tokens,
                rollout_logprobs,
                advantages,
            )
        ):
            # Skip if no response tokens
            if len(response_tokens) == 0:
                skipped_samples += 1
                continue

            # Construct full sequence: prompt + response
            full_tokens = prompt_tokens + response_tokens
            input_tokens = full_tokens[:-1]  # Shift for causal LM
            target_tokens = full_tokens[1:]

            prompt_len = len(prompt_tokens)

            # Construct advantages mask: 0 for prompt, advantage for response
            # Note: input_tokens = full_tokens[:-1], so we need to adjust indices
            all_advantages = [0.0] * (prompt_len - 1) + [advantage.item()] * len(
                response_tokens
            )

            # Old log probs: 0 for prompt, actual logprobs for response
            all_logprobs = [0.0] * (prompt_len - 1) + old_logprobs

            # Ensure lengths match
            if (
                len(input_tokens) != len(target_tokens)
                or len(input_tokens) != len(all_advantages)
                or len(input_tokens) != len(all_logprobs)
            ):
                logger.warning(
                    f"Length mismatch at sample {idx}: "
                    f"input={len(input_tokens)}, target={len(target_tokens)}, "
                    f"adv={len(all_advantages)}, logprobs={len(all_logprobs)}"
                )
                skipped_samples += 1
                continue

            # Create Datum
            datum = tinker.types.Datum(
                model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
                loss_fn_inputs={
                    "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                    "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                    "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                    # cliprange is passed via loss_fn_config, not loss_fn_inputs
                },
            )
            training_datums.append(datum)

        n_training_samples = len(training_datums)
        metrics["time/build_data"] = time.time() - t_build_data_start
        metrics["rollout/n_training_samples"] = n_training_samples
        metrics["rollout/skipped_samples"] = skipped_samples

        if n_training_samples == 0:
            logger.warning("No valid training samples! Skipping this step.")
            continue

        logger.info(
            f"  Built {n_training_samples} training samples ({skipped_samples} skipped)"
        )

        # 5.6. Training with multiple epochs (off-policy support)
        t_train_start = time.time()

        # Configure PPO clipping thresholds
        # cliprange=0.2 means clip ratio to [0.8, 1.2]
        clip_low = 1.0 - config.cliprange
        clip_high = 1.0 + config.cliprange

        # Calculate number of batches per epoch
        n_batches_per_epoch = len(training_datums) // config.train_batch_size
        n_total_updates = config.epochs_per_rollout_batch * n_batches_per_epoch

        logger.info(
            f"  Training: {config.epochs_per_rollout_batch} epochs, "
            f"{n_batches_per_epoch} batches per epoch, "
            f"{n_total_updates} total gradient steps"
        )

        # Outer loop: epochs
        for epoch in range(config.epochs_per_rollout_batch):
            # Shuffle training datums for this epoch
            shuffled_datums = training_datums.copy()
            random.shuffle(shuffled_datums)

            # Inner loop: batches
            for batch_idx in range(n_batches_per_epoch):
                # Extract batch of datums
                start_idx = batch_idx * config.train_batch_size
                end_idx = start_idx + config.train_batch_size
                batch_datums = shuffled_datums[start_idx:end_idx]

                # Tinker handles everything in forward_backward!
                # - Runs forward pass through CURRENT policy → NEW logprobs
                # - Compares against old_logprobs in datum.loss_fn_inputs["logprobs"]
                # - Computes GRPO-Clip loss with ratio and clipping
                fwd_bwd_future = training_client.forward_backward(
                    batch_datums,  # Contains old_logprobs from sampling
                    loss_fn="ppo",  # GRPO-Clip (PPO-style clipping)
                    loss_fn_config={
                        "clip_low_threshold": clip_low,
                        "clip_high_threshold": clip_high,
                    },
                )

                # Optimizer step
                optim_step_future = training_client.optim_step(adam_params)

                # Wait for completion
                fwd_bwd_result = fwd_bwd_future.result()
                optim_result = optim_step_future.result()

        metrics["time/training"] = time.time() - t_train_start
        metrics["training/n_gradient_steps"] = n_total_updates

        # 5.7. Log metrics
        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=grpo_step)

        logger.info(f"  Step {grpo_step} completed in {metrics['time/total']:.1f}s")

        # 5.8. Save checkpoint
        if (grpo_step + 1) % config.save_every == 0:
            logger.info(f"  Saving checkpoint at step {grpo_step}...")
            checkpoint_utils.save_checkpoint(
                training_client=training_client,
                name=f"grpo_step_{grpo_step:06d}",
                log_path=config.log_path,
                kind="both",
                loop_state={"grpo_step": grpo_step + 1},
            )

        # 5.9. Run validation
        if grpo_step % config.eval_every == 0:
            try:
                val_metrics = validate_on_test_set(
                    sampling_client,
                    test_dataset,
                    tokenizer,
                    config.use_r1_zero_format,
                    n_samples=60,
                )
                ml_logger.log_metrics(
                    {f"val/{k}": v for k, v in val_metrics.items()}, step=grpo_step
                )
            except Exception as e:
                logger.warning(f"Validation failed: {e}")

    # 6. Final checkpoint
    logger.info("Training complete! Saving final checkpoint...")
    checkpoint_utils.save_checkpoint(
        training_client=training_client,
        name="final",
        log_path=config.log_path,
        kind="both",
        loop_state={"grpo_step": config.n_grpo_steps},
    )

    ml_logger.close()
    logger.info("GRPO training completed successfully!")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
