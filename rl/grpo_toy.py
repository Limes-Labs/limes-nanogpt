#!/usr/bin/env python3
"""Tiny GRPO-style RL post-training toy for a character policy.

This is the critic-free sibling of ``ppo_toy.py``. It samples grouped,
variable-length completions, scores them with the same verifiable reward, and
uses within-group normalized rewards as the advantage estimate. The point is to
make the PPO-vs-GRPO tradeoff inspectable on CPU, not to claim model-quality
gains.
"""

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone

try:
    from rl.ppo_toy import EOS_TOKEN, VOCAB, TinyCharPolicy, score_completion
except ModuleNotFoundError:  # Allows ``python rl/grpo_toy.py`` from repo root.
    from ppo_toy import EOS_TOKEN, VOCAB, TinyCharPolicy, score_completion


@dataclass
class GroupedRollout:
    group_id: int
    actions: list
    old_logprob: float
    ref_logprob: float
    text: str
    ended: bool
    reward: float
    group_advantage: float


def collect_grouped_rollouts(policy, reference, rng, groups, group_size, target):
    rollouts = []
    reward_stds = []
    for group_id in range(groups):
        group = []
        for _ in range(group_size):
            actions, text, ended, old_logprob, _value = policy.sample(rng)
            group.append(
                {
                    "actions": actions,
                    "old_logprob": old_logprob,
                    "ref_logprob": reference.logprob(actions),
                    "text": text,
                    "ended": ended,
                    "reward": score_completion(text, target=target, ended=ended),
                }
            )

        rewards = [item["reward"] for item in group]
        mean_reward = sum(rewards) / len(rewards)
        variance = sum((reward - mean_reward) ** 2 for reward in rewards) / len(rewards)
        reward_std = math.sqrt(variance)
        reward_stds.append(reward_std)

        for item in group:
            advantage = 0.0 if reward_std == 0.0 else (item["reward"] - mean_reward) / (reward_std + 1e-8)
            rollouts.append(
                GroupedRollout(
                    group_id=group_id,
                    actions=item["actions"],
                    old_logprob=item["old_logprob"],
                    ref_logprob=item["ref_logprob"],
                    text=item["text"],
                    ended=item["ended"],
                    reward=item["reward"],
                    group_advantage=advantage,
                )
            )

    return rollouts, reward_stds


def update_policy(policy, reference, rollouts, learning_rate, clip_range, kl_beta, target):
    logit_grads = [[0.0 for _ in VOCAB] for _ in range(policy.max_new_tokens)]
    policy_losses = []

    for rollout in rollouts:
        new_logprob = policy.logprob(rollout.actions)
        ratio = math.exp(new_logprob - rollout.old_logprob)
        clipped = max(1.0 - clip_range, min(1.0 + clip_range, ratio))
        unclipped_obj = ratio * rollout.group_advantage
        clipped_obj = clipped * rollout.group_advantage
        active_policy_grad = unclipped_obj <= clipped_obj
        policy_losses.append(-min(unclipped_obj, clipped_obj))

        if active_policy_grad:
            logprob_coeff = -rollout.group_advantage * ratio
        else:
            logprob_coeff = 0.0

        for step, action in enumerate(rollout.actions):
            probs = policy.probs(step)
            for token_id, prob in enumerate(probs):
                indicator = 1.0 if token_id == action else 0.0
                logit_grads[step][token_id] += logprob_coeff * (indicator - prob) / len(rollouts)

    for step in range(policy.max_new_tokens):
        p = policy.probs(step)
        q = reference.probs(step)
        kl = sum(pi * (math.log(pi + 1e-12) - math.log(qi + 1e-12)) for pi, qi in zip(p, q))
        for token_id, prob in enumerate(p):
            logit_grads[step][token_id] += kl_beta * prob * (
                math.log(prob + 1e-12) - math.log(q[token_id] + 1e-12) - kl
            )

    for step in range(policy.max_new_tokens):
        for token_id in range(len(VOCAB)):
            policy.logits[step][token_id] -= learning_rate * logit_grads[step][token_id]

    rewards = [r.reward for r in rollouts]
    return {
        "avg_reward": sum(rewards) / len(rewards),
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "avg_group_advantage": sum(r.group_advantage for r in rollouts) / len(rollouts),
        "approx_kl_to_reference": policy.mean_kl_to(reference),
        "exact_match_rate": sum(1 for r in rollouts if r.text == target and r.ended) / len(rollouts),
        "avg_completion_len": sum(len(r.text) for r in rollouts) / len(rollouts),
        "ended_rate": sum(1 for r in rollouts if r.ended) / len(rollouts),
    }


def run_training(args):
    rng = random.Random(args.seed)
    policy = TinyCharPolicy(max_new_tokens=args.max_new_tokens)
    reference = copy.deepcopy(policy)
    history = []

    for step in range(args.steps):
        rollouts, reward_stds = collect_grouped_rollouts(
            policy,
            reference,
            rng,
            groups=args.groups,
            group_size=args.group_size,
            target=args.target,
        )
        metrics = update_policy(
            policy,
            reference,
            rollouts,
            learning_rate=args.learning_rate,
            clip_range=args.clip_range,
            kl_beta=args.kl_beta,
            target=args.target,
        )
        metrics["avg_group_reward_std"] = sum(reward_stds) / len(reward_stds)
        metrics["step"] = step + 1
        history.append(metrics)

    return {
        "schema_version": "limes-grpo-toy-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target": args.target,
            "seed": args.seed,
            "steps": args.steps,
            "groups": args.groups,
            "group_size": args.group_size,
            "max_new_tokens": args.max_new_tokens,
            "clip_range": args.clip_range,
            "kl_beta": args.kl_beta,
            "learning_rate": args.learning_rate,
            "uses_value_head": False,
            "vocab": list(VOCAB),
            "eos_token": EOS_TOKEN,
        },
        "history": history,
        "final": history[-1] if history else {},
        "limitations": [
            "Toy character policy, not a nanoGPT checkpoint.",
            "Group-relative rewards are useful for comparing grouped samples, not a replacement for all credit assignment.",
            "No learned value model or reward model is trained.",
            "KL is measured against the random initialization reference policy.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target", default="AB")
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.03)
    parser.add_argument("--learning_rate", type=float, default=0.15)
    parser.add_argument(
        "--output_json",
        default="out-rl/grpo_toy.json",
        help="Path for the reproducible toy GRPO run artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.groups <= 0:
        raise ValueError("--groups must be positive")
    if args.group_size <= 1:
        raise ValueError("--group_size must be greater than 1 for group-relative advantages")

    artifact = run_training(args)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    final = artifact["final"]
    print(
        "grpo_toy "
        f"avg_reward={final['avg_reward']:.4f} "
        f"kl={final['approx_kl_to_reference']:.6f} "
        f"group_reward_std={final['avg_group_reward_std']:.4f} "
        f"exact_match_rate={final['exact_match_rate']:.3f}"
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
