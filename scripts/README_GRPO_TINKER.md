# GRPO Training with Tinker

On-policy Group Relative Policy Optimization (GRPO) training for math reasoning on GSM8K, running on Tinker's cloud infrastructure.

## Overview

This implementation:
- **Trains on GSM8K** using HuggingFace datasets
- **Uses Qwen/Qwen3-8B** as the base model (configurable)
- **Implements GRPO-Clip** (PPO-style clipping for stability)
- **Reuses assignment code** from `tests/adapters.py` (especially `run_compute_group_normalized_rewards`)
- **Runs on Tinker cloud** for distributed training and sampling

## Quick Start

### Prerequisites

1. **Tinker API access**: Set your API key
   ```bash
   export TINKER_API_KEY="your-api-key-here"
   ```

2. **Python dependencies**: Ensure tinker-cookbook is available
   ```bash
   # Should already be set up from your environment
   pip install tinker datasets transformers torch
   ```

### Running the Script

#### Smoke Test (5-10 minutes)
Test with a small configuration to verify everything works:

```bash
python scripts/grpo_train_tinker.py \
    --n_grpo_steps=5 \
    --rollout_batch_size=8 \
    --group_size=2 \
    --save_every=2 \
    --eval_every=2
```

**Expected output**:
- Script runs without errors
- Generates rollouts from current policy
- Computes rewards and advantages
- Trains with GRPO-Clip loss
- Saves checkpoints every 2 steps
- Logs metrics to `/tmp/grpo-tinker/`

#### Small-Scale Training (1-2 hours)
Run a smaller version for faster iteration:

```bash
python scripts/grpo_train_tinker.py \
    --n_grpo_steps=50 \
    --rollout_batch_size=64 \
    --group_size=8 \
    --learning_rate=1e-5 \
    --log_path=/tmp/grpo-tinker-small \
    --wandb_project=grpo-tinker-test
```

#### Full Training (4-8 hours)
Full assignment configuration:

```bash
python scripts/grpo_train_tinker.py \
    --n_grpo_steps=200 \
    --rollout_batch_size=256 \
    --group_size=8 \
    --learning_rate=1e-5 \
    --log_path=/tmp/grpo-tinker-full \
    --wandb_project=grpo-gsm8k-full \
    --wandb_name=grpo-qwen3-8b-run1
```

## Configuration Options

### Model & Service
- `--model_name`: Base model (default: `"Qwen/Qwen3-8B"`)
- `--base_url`: Tinker API base URL (default: uses `TINKER_API_KEY` env)
- `--lora_rank`: LoRA rank (default: `32`)

### Training Hyperparameters
- `--n_grpo_steps`: Number of GRPO steps (default: `200`)
- `--learning_rate`: Learning rate (default: `1e-5`)
- `--rollout_batch_size`: Questions per batch (default: `256`)
- `--group_size`: Samples per question (default: `8`)

### GRPO-Specific
- `--advantage_eps`: Epsilon for group normalization (default: `1e-6`)
- `--use_std_normalization`: Normalize by std (default: `True`)
- `--cliprange`: PPO clip range (default: `0.2`)

### Sampling
- `--sampling_temperature`: Sampling temperature (default: `1.0`)
- `--sampling_max_tokens`: Max tokens per sample (default: `1024`)
- `--use_r1_zero_format`: Use `<think></think><answer></answer>` format (default: `True`)

### Logging & Checkpointing
- `--log_path`: Directory for logs and checkpoints (default: `"/tmp/grpo-tinker"`)
- `--save_every`: Save checkpoint every N steps (default: `20`)
- `--eval_every`: Evaluate on test set every N steps (default: `20`)
- `--wandb_project`: Weights & Biases project name (default: `None`)
- `--wandb_name`: W&B run name (default: `None`)

## Architecture

The training loop follows this structure:

```
FOR each GRPO step (200 iterations):
├── Save current policy weights
├── Create SamplingClient from current weights
├── Sample rollouts (group_size per question) [CLOUD]
├── Compute rewards (local, using gsm8k_reward_fn)
├── Compute group-normalized advantages (local, using assignment's function)
├── Build training data (Datum objects)
├── Train with GRPO-Clip loss [CLOUD]
│   ├── forward_backward() with loss_fn="ppo"
│   └── optim_step()
├── Log metrics
├── Save checkpoint (every save_every steps)
└── Validate on test set (every eval_every steps)
```

## Key Implementation Details

### 1. Reward Function
The `gsm8k_reward_fn` extracts answers from:
- `<answer>...</answer>` tags (R1-Zero format)
- `\\boxed{...}` (LaTeX format)
- Last number in response (fallback)

Reward breakdown:
- **Format reward** (20%): Whether answer is properly formatted
- **Answer reward** (80%): Numerical correctness

### 2. Group Normalization
Uses `run_compute_group_normalized_rewards()` from `tests/adapters.py`:
- Groups rollouts by question (group_size samples per question)
- Computes advantages: `A = (r - mean(group_rewards)) / (std(group_rewards) + eps)`
- Returns advantages, raw rewards, and metadata

### 3. On-Policy Training
Enforces on-policy by:
1. Saving weights before each step
2. Creating fresh sampling client each step
3. Training immediately after collecting rollouts
4. Never reusing rollouts

### 4. Loss Function (GRPO-Clip)
Maps to Tinker's `"ppo"` loss:
- Computes ratio: `exp(new_logprob - old_logprob)`
- Clips to `[1-cliprange, 1+cliprange]`
- Loss: `-min(advantages * ratio, advantages * clipped_ratio)`

### 5. Advantage Masking
Advantages are masked to only compute loss on response tokens:
```python
all_advantages = [0.0] * (prompt_len - 1) + [advantage] * len(response_tokens)
```

## Monitoring

### Logged Metrics

**Progress**:
- `progress/grpo_step`: Current step
- `progress/done_frac`: Fraction complete

**Rewards**:
- `reward/mean`: Mean reward across batch
- `reward/std`: Std of rewards
- `reward/min`, `reward/max`: Min/max rewards

**Validation** (every `eval_every` steps):
- `val/accuracy`: Answer correctness
- `val/format_accuracy`: Format compliance

**Timing**:
- `time/save_weights`: Time to save weights
- `time/sampling`: Time to generate rollouts
- `time/compute_rewards`: Time to compute rewards
- `time/build_data`: Time to build training data
- `time/training`: Time for forward-backward and optimizer step
- `time/total`: Total time per step

### Weights & Biases

To use W&B:
```bash
python scripts/grpo_train_tinker.py \
    --wandb_project=my-project \
    --wandb_name=my-run \
    ...other args...
```

## Checkpointing

Checkpoints are saved to `{log_path}/checkpoints/`:
- Every `save_every` steps (default: 20)
- Final checkpoint at end of training
- Both training state and sampler weights

### Resuming Training

If a checkpoint exists at `log_path`, training will automatically resume:
```bash
# First run
python scripts/grpo_train_tinker.py --log_path=/tmp/my-run

# Interrupted at step 50

# Resume from step 50
python scripts/grpo_train_tinker.py --log_path=/tmp/my-run
```

## Troubleshooting

### ImportError: cannot import 'run_compute_group_normalized_rewards'

Make sure you're running from the assignment root:
```bash
cd /Users/chenmoney/Documents/genai/cs336-assignment5
python scripts/grpo_train_tinker.py ...
```

### Tokenization Mismatch

If you see length mismatch warnings:
1. Verify tokenizer matches model: `AutoTokenizer.from_pretrained(config.model_name)`
2. Check that prompt tokenization is consistent
3. Enable debug logging to inspect samples

### Low Rewards

If rewards are consistently low:
1. Check format compliance: `val/format_accuracy`
2. Try simpler prompt format: `--use_r1_zero_format=False`
3. Reduce sampling temperature: `--sampling_temperature=0.7`
4. Add few-shot examples to prompt

### Memory Issues

If running out of memory:
1. Reduce batch size: `--rollout_batch_size=128`
2. Reduce group size: `--group_size=4`
3. Reduce max tokens: `--sampling_max_tokens=512`

### Slow Sampling

If sampling is too slow:
1. Reduce max tokens: `--sampling_max_tokens=512`
2. Reduce group size: `--group_size=4`
3. Use smaller model (if available in Tinker)

## Expected Results

With default hyperparameters on GSM8K:
- **Initial accuracy**: 5-10% (base model)
- **After 50 steps**: 15-20%
- **After 200 steps**: 25-30%+ (target: >25%)

Training time:
- **Smoke test** (5 steps, small batch): 5-10 minutes
- **Small-scale** (50 steps): 1-2 hours
- **Full training** (200 steps): 4-8 hours

## Validation

Validation runs automatically every `eval_every` steps on 100 random test examples:
- Uses greedy sampling (temperature=0.0)
- Reports answer accuracy and format compliance
- Logs to W&B if configured

## Files

- **`grpo_train_tinker.py`**: Main training script (~500 lines)
- **`README_GRPO_TINKER.md`**: This documentation
- **Uses from assignment**:
  - `tests/adapters.py::run_compute_group_normalized_rewards`
  - Other helper functions as needed

## References

- **DeepSeekMath**: https://arxiv.org/abs/2402.03300
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **Tinker Docs**: https://tinker-docs.thinkingmachines.ai/
- **GSM8K**: https://huggingface.co/datasets/openai/gsm8k
