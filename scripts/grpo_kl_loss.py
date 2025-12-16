"""
Custom GRPO loss function with optional KL divergence penalty.

This implements GRPO-Clip (PPO-style) loss with an optional KL divergence
term to prevent the policy from deviating too far from a reference policy.
"""

import torch
import tinker
from typing import List, Dict


def compute_grpo_kl_loss(
    data: List[tinker.Datum],
    logprobs_list: List[torch.Tensor],
    kl_coef: float = 0.0,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute GRPO-Clip loss with optional KL divergence penalty.

    Args:
        data: List of Datum objects with loss_fn_inputs containing:
            - target_tokens: Target token IDs
            - advantages: Advantage values (already group-normalized)
            - logprobs: Old policy log probabilities
            - cliprange: Clipping range for PPO (e.g., 0.2)
            - ref_logprobs (optional): Reference policy log probabilities for KL penalty
        logprobs_list: Log probabilities from current policy (from forward pass)
        kl_coef: Coefficient for KL divergence penalty (0.0 = no KL penalty)

    Returns:
        Tuple of (total_loss, metrics_dict)
    """
    total_loss = torch.tensor(0.0)
    total_tokens = 0

    # Metrics
    total_ppo_loss = 0.0
    total_kl_loss = 0.0
    total_clip_fraction = 0.0
    total_advantage_mean = 0.0
    total_policy_kl = 0.0

    for datum, policy_logprobs in zip(data, logprobs_list):
        # Extract data from datum
        advantages = torch.tensor(datum.loss_fn_inputs["advantages"].data)
        old_logprobs = torch.tensor(datum.loss_fn_inputs["logprobs"].data)
        cliprange = datum.loss_fn_inputs["cliprange"]

        # Number of tokens in this sequence
        n_tokens = len(advantages)
        total_tokens += n_tokens

        # === PPO / GRPO-Clip Loss ===
        # Compute probability ratio: π_θ / π_old = exp(log π_θ - log π_old)
        log_ratio = policy_logprobs - old_logprobs
        ratio = torch.exp(log_ratio)

        # Clipped objective
        clipped_ratio = torch.clamp(ratio, 1 - cliprange, 1 + cliprange)

        # PPO loss: -min(ratio * A, clipped_ratio * A)
        unclipped_objective = ratio * advantages
        clipped_objective = clipped_ratio * advantages
        ppo_loss = -torch.min(unclipped_objective, clipped_objective)

        # Sum over sequence
        ppo_loss_sum = ppo_loss.sum()
        total_loss = total_loss + ppo_loss_sum
        total_ppo_loss += ppo_loss_sum.item()

        # Clip fraction (for monitoring)
        is_clipped = (ratio < 1 - cliprange) | (ratio > 1 + cliprange)
        clip_fraction = is_clipped.float().mean().item()
        total_clip_fraction += clip_fraction * n_tokens

        # Advantage mean (for monitoring)
        total_advantage_mean += advantages.sum().item()

        # === KL Divergence Penalty (Optional) ===
        if kl_coef > 0.0 and "ref_logprobs" in datum.loss_fn_inputs:
            ref_logprobs = torch.tensor(datum.loss_fn_inputs["ref_logprobs"].data)

            # KL divergence: KL(π_θ || π_ref) ≈ log π_ref - log π_θ
            # We want to minimize this, so we add kl_coef * KL to the loss
            kl_divergence = ref_logprobs - policy_logprobs
            kl_loss_sum = kl_coef * kl_divergence.sum()

            total_loss = total_loss + kl_loss_sum
            total_kl_loss += kl_loss_sum.item()
            total_policy_kl += kl_divergence.sum().item()

    # Compute average metrics
    metrics = {
        "loss/ppo": total_ppo_loss / total_tokens,
        "loss/total": total_loss.item() / total_tokens,
        "ppo/clip_fraction": total_clip_fraction / total_tokens,
        "ppo/advantage_mean": total_advantage_mean / total_tokens,
    }

    if kl_coef > 0.0:
        metrics["loss/kl"] = total_kl_loss / total_tokens
        metrics["kl/policy_ref"] = total_policy_kl / total_tokens

    return total_loss, metrics
