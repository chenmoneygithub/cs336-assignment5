# GRPO with KL Divergence - Implementation Guide

## Understanding GRPO with PPO

### What is GRPO?

**Group Relative Policy Optimization (GRPO)** is simply **PPO with group-normalized advantages**. That's it!

From the Tinker cookbook (`tinker_cookbook/rl/data_processing.py`):

```python
def compute_advantages(trajectory_groups):
    for traj_group in trajectory_groups:
        rewards = get_rewards(traj_group)
        # This is GRPO: subtract the mean within each group
        advantages = rewards - rewards.mean()
```

### How it works:

1. **Group Sampling**: For each question, sample `group_size` responses (e.g., 8 responses per question)
2. **Compute Rewards**: Score each response
3. **Normalize Within Group**: `advantage = reward - mean(group_rewards)`
4. **Train with PPO**: Use PPO loss with these group-normalized advantages

**That's all GRPO is!** The PPO loss handles the clipping automatically.

Your `run_compute_group_normalized_rewards` does exactly this (and optionally normalizes by std too):
```python
advantages = (rewards - mean) / (std + eps)  # Optional std normalization
```

---

## Adding KL Divergence

There are **two approaches** to add KL divergence penalty:

### Approach 1: Built-in PPO Loss (Current)

**What you're doing now:**
```python
training_client.forward_backward(
    training_datums,
    loss_fn="ppo",  # Built-in PPO
)
```

**Pros:**
- Simple, fast, well-tested
- No need for custom loss function

**Cons:**
- No KL divergence penalty
- Less flexible

---

### Approach 2: Custom Loss with KL Divergence (New)

**Custom loss function with KL penalty:**
```python
def compute_grpo_kl_loss(data, logprobs_list, kl_coef=0.01):
    total_loss = 0

    for datum, policy_logprobs in zip(data, logprobs_list):
        advantages = datum.loss_fn_inputs["advantages"]
        old_logprobs = datum.loss_fn_inputs["logprobs"]
        ref_logprobs = datum.loss_fn_inputs["ref_logprobs"]  # NEW

        # PPO loss
        ratio = torch.exp(policy_logprobs - old_logprobs)
        clipped_ratio = torch.clamp(ratio, 1-cliprange, 1+cliprange)
        ppo_loss = -torch.min(ratio * advantages, clipped_ratio * advantages)

        # KL divergence: KL(π_θ || π_ref) ≈ log π_ref - log π_θ
        kl_loss = kl_coef * (ref_logprobs - policy_logprobs)

        total_loss += ppo_loss.sum() + kl_loss.sum()

    return total_loss, metrics
```

**Usage:**
```python
# Use custom loss
training_client.forward_backward_custom(
    training_datums,
    loss_fn=lambda data, logprobs: compute_grpo_kl_loss(data, logprobs, kl_coef=0.01)
)
```

**Pros:**
- Prevents policy from diverging too far from reference (base model)
- More stable training
- Better control over exploration

**Cons:**
- Requires computing reference logprobs (extra computation)
- Slower than built-in PPO (1.5x FLOPs, up to 3x wall-clock time)

---

## Implementation Steps

### Option 1: No KL Penalty (Current Approach)

Your current code is **already correct**! You're using:
- Group-normalized advantages ✅
- Built-in PPO loss ✅

No changes needed unless you want KL divergence.

---

### Option 2: Add KL Divergence

**Step 1: Create reference sampling client**

```python
# Reference model (base model before training)
ref_sampling_client = service_client.create_sampling_client(
    base_model=config.model_name
)
```

**Step 2: Compute reference logprobs**

```python
async def compute_reference_logprobs(ref_client, prompt_tokens, response_tokens):
    """Compute log probabilities from reference model."""
    tasks = []
    for prompt, response in zip(prompt_tokens, response_tokens):
        full_tokens = prompt + response
        model_input = tinker.types.ModelInput.from_ints(tokens=full_tokens)
        tasks.append(ref_client.compute_logprobs_async(model_input))

    all_ref_logprobs = await asyncio.gather(*tasks)

    # Extract response logprobs only
    ref_logprobs_list = []
    for prompt, response, ref_logprobs in zip(prompt_tokens, response_tokens, all_ref_logprobs):
        prompt_len = len(prompt)
        # Skip prompt, get response logprobs
        ref_logprobs_response = ref_logprobs[prompt_len : prompt_len + len(response)]
        ref_logprobs_list.append(ref_logprobs_response)

    return ref_logprobs_list
```

**Step 3: Include reference logprobs in Datum**

```python
# Add ref_logprobs to loss_fn_inputs
datum = tinker.types.Datum(
    model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
        "ref_logprobs": TensorData.from_torch(torch.tensor(all_ref_logprobs)),  # NEW
        # NOTE: cliprange is NOT in loss_fn_inputs for custom loss
        # You handle clipping in your custom loss function
    },
)
```

**Step 4: Use custom loss**

```python
from grpo_kl_loss import compute_grpo_kl_loss

# Train with KL penalty
fwd_bwd_result = training_client.forward_backward_custom(
    training_datums,
    loss_fn=lambda data, logprobs: compute_grpo_kl_loss(
        data, logprobs, kl_coef=0.01
    )
).result()
```

---

## Configuration

### Clipping Range (`cliprange`)

**IMPORTANT: How to pass `cliprange` depends on which loss you use!**

#### For Built-in PPO Loss:
Pass via `loss_fn_config` (NOT in `loss_fn_inputs`):

```python
# cliprange=0.2 means clip ratio to [0.8, 1.2]
clip_low = 1.0 - cliprange   # 0.8
clip_high = 1.0 + cliprange  # 1.2

training_client.forward_backward(
    training_datums,
    loss_fn="ppo",
    loss_fn_config={
        "clip_low_threshold": clip_low,
        "clip_high_threshold": clip_high,
    },
)
```

**❌ WRONG - Don't put cliprange in loss_fn_inputs:**
```python
# This will cause a Pydantic validation error!
datum = tinker.types.Datum(
    loss_fn_inputs={
        "cliprange": 0.2,  # ❌ ERROR!
    }
)
```

#### For Custom Loss:
Handle clipping in your custom loss function (see `grpo_kl_loss.py`).

### KL Coefficient (`kl_coef`)

- **`kl_coef = 0.0`**: No KL penalty (equivalent to standard GRPO)
- **`kl_coef = 0.01`**: Light penalty (recommended starting point)
- **`kl_coef = 0.1`**: Strong penalty (conservative updates)

**Rule of thumb:** Start with `kl_coef = 0.01` and increase if policy diverges too quickly.

---

## Files Created

1. **`grpo_kl_loss.py`**: Custom loss function implementation
2. **`grpo_with_kl_example.py`**: Example integration code
3. **`README_GRPO_KL.md`**: This guide

---

## Example Integration

```python
# In your training loop (grpo_train_tinker.py)

# Add to config
kl_coef: float = 0.0  # Set to 0.01 to enable KL penalty

# Create reference client (once at start)
if config.kl_coef > 0:
    ref_sampling_client = service_client.create_sampling_client(
        base_model=config.model_name
    )

# In training loop
if config.kl_coef > 0:
    # Compute reference logprobs
    ref_logprobs_list = await compute_reference_logprobs(
        ref_sampling_client,
        rollout_prompt_tokens,
        rollout_response_tokens,
    )

    # Add ref_logprobs to datums
    all_ref_logprobs = [0.0] * (prompt_len - 1) + ref_logprobs

    # Use custom loss
    from grpo_kl_loss import compute_grpo_kl_loss
    fwd_bwd_future = training_client.forward_backward_custom(
        training_datums,
        loss_fn=lambda d, lp: compute_grpo_kl_loss(d, lp, kl_coef=config.kl_coef)
    )
else:
    # Use built-in PPO (current approach)
    fwd_bwd_future = training_client.forward_backward(
        training_datums,
        loss_fn="ppo",
    )
```

---

## References

- **DeepSeekMath GRPO**: https://arxiv.org/abs/2402.03300
- **DeepSeek-R1**: https://arxiv.org/abs/2501.12948
- **Tinker PPO Documentation**: https://tinker-docs.thinkingmachines.ai/losses#proximal-policy-optimization-ppo
- **Tinker Cookbook RL**: `tinker-cookbook/tinker_cookbook/rl/`

---

## Summary

✅ **GRPO = PPO + Group-Normalized Advantages**

✅ **Your current implementation is correct** (using built-in PPO)

✅ **To add KL divergence:**
1. Compute reference logprobs from base model
2. Include them in `loss_fn_inputs`
3. Use custom loss function with `forward_backward_custom`

✅ **KL penalty prevents policy from diverging too far from base model**
