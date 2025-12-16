# Tinker API Reference - Common Patterns

Quick reference for common Tinker API usage patterns.

## Sampling Response Structure

When you call `sampling_client.sample()`, you get a response object with:

```python
sample_result = sampling_client.sample(
    prompt=tinker.types.ModelInput.from_str(prompt),
    num_samples=8,
    sampling_params=sampling_params,
).result()
```

### Response Structure

```python
sample_result.sequences  # List of SampledSequence objects
```

### SampledSequence Fields

Each `sequence` in `sample_result.sequences` has:

| Field | Type | Description |
|-------|------|-------------|
| `sequence.tokens` | `list[int]` | Generated token IDs |
| `sequence.logprobs` | `list[float]` or `None` | Log probabilities for each token |
| `sequence.stop_reason` | `str` | Reason why sampling stopped |

**⚠️ IMPORTANT: There is NO `sequence.text` field!**

### Getting Text from Sequences

To convert tokens to text, use the tokenizer:

```python
for sequence in sample_result.sequences:
    # ❌ WRONG - sequence.text doesn't exist!
    # response_text = sequence.text

    # ✅ CORRECT - decode the tokens
    response_text = tokenizer.decode(
        sequence.tokens,
        skip_special_tokens=False
    )

    # Access log probabilities
    if sequence.logprobs is not None:
        log_probs = sequence.logprobs
```

## Tokenization

### Encoding (string → tokens)

```python
# Encode prompt to tokens
prompt_tokens = tokenizer.encode(
    prompt_string,
    add_special_tokens=False  # Usually False for prompts
)
```

### Decoding (tokens → string)

```python
# Decode tokens to string
response_text = tokenizer.decode(
    token_ids,
    skip_special_tokens=False  # Keep special tokens for debugging
)
```

## Computing Log Probabilities

To compute log probabilities from a reference model:

```python
# Create model input from tokens
full_tokens = prompt_tokens + response_tokens
model_input = tinker.types.ModelInput.from_ints(tokens=full_tokens)

# Compute log probabilities
ref_logprobs = await ref_sampling_client.compute_logprobs_async(model_input)

# ref_logprobs is a list of floats with length = len(full_tokens)
```

## Common Patterns

### Pattern 1: Sample and Extract Text

```python
# Sample
result = sampling_client.sample(
    prompt=tinker.types.ModelInput.from_str(prompt),
    num_samples=8,
    sampling_params=sampling_params,
).result()

# Extract text and logprobs
for sequence in result.sequences:
    tokens = sequence.tokens
    logprobs = sequence.logprobs

    # Decode to text
    text = tokenizer.decode(tokens, skip_special_tokens=False)

    # Process text
    reward = reward_fn(text, ground_truth)
```

### Pattern 2: Tokenize Prompt for Training

```python
# Tokenize prompt
prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)

# Sample response (gets tokens automatically)
result = sampling_client.sample(
    prompt=tinker.types.ModelInput.from_str(prompt),
    num_samples=1,
    sampling_params=sampling_params,
).result()

response_tokens = result.sequences[0].tokens
response_logprobs = result.sequences[0].logprobs

# Build training data
full_tokens = prompt_tokens + response_tokens
input_tokens = full_tokens[:-1]  # Shift for causal LM
target_tokens = full_tokens[1:]
```

### Pattern 3: Compute Reference Log Probabilities

```python
async def get_ref_logprobs(ref_client, prompt_tokens, response_tokens):
    # Combine prompt and response
    full_tokens = prompt_tokens + response_tokens

    # Compute logprobs
    model_input = tinker.types.ModelInput.from_ints(tokens=full_tokens)
    all_logprobs = await ref_client.compute_logprobs_async(model_input)

    # Extract only response logprobs
    prompt_len = len(prompt_tokens)
    response_logprobs = all_logprobs[prompt_len : prompt_len + len(response_tokens)]

    return response_logprobs
```

## Loss Functions

### Built-in PPO Loss

**Required `loss_fn_inputs` fields:**
- `target_tokens`: `array[(N,), int]` - Target token IDs
- `logprobs`: `array[(N,), float]` - Sampling log probabilities (old logprobs from sampling)
- `advantages`: `array[(N,), float]` - Advantage values for RL (positive=reinforce, negative=discourage)

**Optional `loss_fn_config` parameters:**
- `clip_low_threshold`: Lower bound for clipping (default: 0.8)
- `clip_high_threshold`: Upper bound for clipping (default: 1.2)

**Example:**
```python
# Build datum with PPO loss inputs
datum = tinker.types.Datum(
    model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
    loss_fn_inputs={
        "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
        "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
        "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
    },
)

# Use PPO loss with custom clipping
fwd_bwd_result = training_client.forward_backward(
    training_datums,
    loss_fn="ppo",
    loss_fn_config={
        "clip_low_threshold": 0.8,
        "clip_high_threshold": 1.2,
    }
).result()
```

**⚠️ Important:** All arrays (target_tokens, advantages, logprobs) must have the same length N.

### Custom Loss with KL Divergence

```python
# Define custom loss function
def my_loss_fn(data: list[tinker.Datum], logprobs: list[torch.Tensor]):
    # Your custom loss logic here
    loss = compute_loss(data, logprobs)
    metrics = {"loss": loss.item()}
    return loss, metrics

# Use custom loss
fwd_bwd_result = training_client.forward_backward_custom(
    training_datums,
    loss_fn=my_loss_fn,
).result()
```

## Common Gotchas

1. **No `sequence.text` field** - Always use `tokenizer.decode(sequence.tokens)`

2. **Prompt tokenization** - Need to tokenize prompts manually for training:
   ```python
   prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
   ```

3. **Log probs can be None** - Always check:
   ```python
   if sequence.logprobs is None:
       logger.warning("No logprobs returned!")
       continue
   ```

4. **Shifting for causal LM** - Remember to shift tokens:
   ```python
   input_tokens = full_tokens[:-1]   # Remove last token
   target_tokens = full_tokens[1:]    # Remove first token
   ```

5. **Advantages padding** - Pad advantages with zeros for prompt tokens:
   ```python
   all_advantages = [0.0] * (prompt_len - 1) + [advantage] * len(response_tokens)
   ```

## References

- Tinker API Docs: https://tinker-docs.thinkingmachines.ai/
- Types Reference: https://tinker-docs.thinkingmachines.ai/api-reference/types
- Loss Functions: https://tinker-docs.thinkingmachines.ai/losses
