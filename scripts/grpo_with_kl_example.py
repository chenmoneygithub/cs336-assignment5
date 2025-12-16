"""
Example: How to use GRPO with optional KL divergence penalty.

This shows two approaches:
1. Using built-in "ppo" loss (what you're doing now)
2. Using custom loss function with KL divergence
"""

import asyncio
import logging
import torch
import tinker
from tinker.types.tensor_data import TensorData

from grpo_kl_loss import compute_grpo_kl_loss

logger = logging.getLogger(__name__)


# =============================================================================
# Approach 1: Built-in PPO loss (your current approach)
# =============================================================================

def build_training_data_builtin_ppo(
    rollout_prompt_tokens,
    rollout_response_tokens,
    rollout_logprobs,
    advantages,
):
    """
    Build training data for built-in PPO loss.

    Note: cliprange is NOT included in loss_fn_inputs.
    It's passed via loss_fn_config in forward_backward().
    """
    training_datums = []

    for prompt_tokens, response_tokens, old_logprobs, advantage in zip(
        rollout_prompt_tokens,
        rollout_response_tokens,
        rollout_logprobs,
        advantages,
    ):
        if len(response_tokens) == 0:
            continue

        # Construct full sequence
        full_tokens = prompt_tokens + response_tokens
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]

        prompt_len = len(prompt_tokens)

        # Advantages: 0 for prompt, advantage for response
        all_advantages = [0.0] * (prompt_len - 1) + [advantage.item()] * len(
            response_tokens
        )

        # Old log probs: 0 for prompt, actual logprobs for response
        all_logprobs = [0.0] * (prompt_len - 1) + old_logprobs

        # Create Datum
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                # cliprange is passed via loss_fn_config, not here
            },
        )
        training_datums.append(datum)

    return training_datums


def train_with_builtin_ppo(training_client, training_datums, learning_rate, cliprange=0.2):
    """Train using built-in PPO loss (no KL penalty)."""
    # Configure PPO clipping thresholds
    clip_low = 1.0 - cliprange
    clip_high = 1.0 + cliprange

    # Forward-backward with built-in PPO
    fwd_bwd_future = training_client.forward_backward(
        training_datums,
        loss_fn="ppo",  # Built-in PPO loss
        loss_fn_config={
            "clip_low_threshold": clip_low,
            "clip_high_threshold": clip_high,
        },
    )

    # Optimizer step
    adam_params = tinker.types.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    optim_step_future = training_client.optim_step(adam_params)

    fwd_bwd_result = fwd_bwd_future.result()
    optim_result = optim_step_future.result()

    return fwd_bwd_result


# =============================================================================
# Approach 2: Custom loss with KL divergence
# =============================================================================


async def compute_reference_logprobs(
    ref_sampling_client: tinker.SamplingClient,
    prompt_tokens_list: list[list[int]],
    response_tokens_list: list[list[int]],
) -> list[list[float]]:
    """
    Compute reference policy log probabilities for KL penalty.

    Args:
        ref_sampling_client: Reference model sampling client (e.g., base model before training)
        prompt_tokens_list: List of prompt token sequences
        response_tokens_list: List of response token sequences

    Returns:
        List of reference log probabilities for each sequence

    Note:
        - sequence.tokens contains the generated token IDs
        - sequence.logprobs contains the log probabilities
        - There is NO sequence.text field - use tokenizer.decode(sequence.tokens) to get text
    """
    ref_logprobs_list = []

    # Compute reference logprobs for each sequence
    tasks = []
    for prompt_tokens, response_tokens in zip(prompt_tokens_list, response_tokens_list):
        full_tokens = prompt_tokens + response_tokens
        model_input = tinker.types.ModelInput.from_ints(tokens=full_tokens)
        tasks.append(ref_sampling_client.compute_logprobs_async(model_input))

    # Wait for all computations
    all_ref_logprobs = await asyncio.gather(*tasks)

    # Extract relevant logprobs (skip prompt tokens)
    for i, (prompt_tokens, response_tokens, ref_logprobs) in enumerate(
        zip(prompt_tokens_list, response_tokens_list, all_ref_logprobs)
    ):
        prompt_len = len(prompt_tokens)

        # Shift and extract response logprobs
        # ref_logprobs has length = len(full_tokens)
        # We want logprobs for response tokens only
        # After shifting for causal LM, we need [prompt_len-1 : prompt_len-1+len(response_tokens)]
        ref_logprobs_response = ref_logprobs[prompt_len : prompt_len + len(response_tokens)]

        ref_logprobs_list.append(ref_logprobs_response)

    return ref_logprobs_list


def build_training_data_with_kl(
    rollout_prompt_tokens,
    rollout_response_tokens,
    rollout_logprobs,
    advantages,
    ref_logprobs_list,  # NEW: Reference logprobs for KL
):
    """
    Build training data for custom loss with KL divergence.

    Similar to build_training_data_builtin_ppo, but includes reference logprobs.
    Note: For custom loss, cliprange is handled in the loss function itself.
    """
    training_datums = []

    for prompt_tokens, response_tokens, old_logprobs, advantage, ref_logprobs in zip(
        rollout_prompt_tokens,
        rollout_response_tokens,
        rollout_logprobs,
        advantages,
        ref_logprobs_list,
    ):
        if len(response_tokens) == 0:
            continue

        # Construct full sequence
        full_tokens = prompt_tokens + response_tokens
        input_tokens = full_tokens[:-1]
        target_tokens = full_tokens[1:]

        prompt_len = len(prompt_tokens)

        # Advantages: 0 for prompt, advantage for response
        all_advantages = [0.0] * (prompt_len - 1) + [advantage.item()] * len(
            response_tokens
        )

        # Old log probs: 0 for prompt, actual logprobs for response
        all_logprobs = [0.0] * (prompt_len - 1) + old_logprobs

        # Reference log probs: 0 for prompt, ref logprobs for response
        all_ref_logprobs = [0.0] * (prompt_len - 1) + ref_logprobs

        # Create Datum
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs={
                "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                "ref_logprobs": TensorData.from_torch(
                    torch.tensor(all_ref_logprobs)
                ),  # NEW for KL divergence
                # cliprange handled in custom loss function, not here
            },
        )
        training_datums.append(datum)

    return training_datums


def train_with_custom_loss(
    training_client, training_datums, learning_rate, kl_coef=0.01
):
    """Train using custom loss function with KL divergence."""

    # Define loss function closure
    def loss_fn(data, logprobs_list):
        return compute_grpo_kl_loss(data, logprobs_list, kl_coef=kl_coef)

    # Forward-backward with custom loss
    fwd_bwd_future = training_client.forward_backward_custom(training_datums, loss_fn)

    # Optimizer step
    adam_params = tinker.types.AdamParams(
        learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )
    optim_step_future = training_client.optim_step(adam_params)

    fwd_bwd_result = fwd_bwd_future.result()
    optim_result = optim_step_future.result()

    return fwd_bwd_result


# =============================================================================
# Example usage in your training loop
# =============================================================================


async def example_training_step(
    training_client,
    service_client,
    model_name,
    rollout_prompt_tokens,
    rollout_response_tokens,
    rollout_logprobs,
    advantages,
    use_kl_penalty=False,
    kl_coef=0.01,
):
    """
    Example of a single training step with optional KL divergence.

    Args:
        use_kl_penalty: If True, use custom loss with KL divergence.
                       If False, use built-in PPO loss.
        kl_coef: Coefficient for KL divergence penalty (only used if use_kl_penalty=True)
    """

    if use_kl_penalty:
        logger.info("Using custom loss with KL divergence")

        # Create reference sampling client (base model)
        ref_sampling_client = service_client.create_sampling_client(base_model=model_name)

        # Compute reference logprobs
        ref_logprobs_list = await compute_reference_logprobs(
            ref_sampling_client,
            rollout_prompt_tokens,
            rollout_response_tokens,
        )

        # Build training data with reference logprobs
        training_datums = build_training_data_with_kl(
            rollout_prompt_tokens,
            rollout_response_tokens,
            rollout_logprobs,
            advantages,
            ref_logprobs_list,
        )

        # Train with custom loss
        result = train_with_custom_loss(
            training_client,
            training_datums,
            learning_rate=1e-5,
            kl_coef=kl_coef,
        )
    else:
        logger.info("Using built-in PPO loss (no KL)")

        # Build training data (current approach)
        training_datums = build_training_data_builtin_ppo(
            rollout_prompt_tokens,
            rollout_response_tokens,
            rollout_logprobs,
            advantages,
        )

        # Train with built-in PPO (cliprange passed here)
        result = train_with_builtin_ppo(
            training_client,
            training_datums,
            learning_rate=1e-5,
            cliprange=0.2,  # PPO clipping parameter
        )

    return result
