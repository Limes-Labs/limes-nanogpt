#!/usr/bin/env python3
"""Tiny PPO-style RL post-training toy for a character policy.

This is intentionally small and educational. It does not train nanoGPT or claim
model-quality gains; it shows the RL loop shape Limes can later replace with a
real checkpoint-backed policy: sample, score with a verifiable reward, update
with a clipped policy-gradient objective, and keep a KL tether to a frozen
reference policy.
"""

import argparse
import copy
import json
import math
import os
import random
from dataclasses import dataclass
from datetime import datetime, timezone


VOCAB = ("A", "B", "C", " ", "<eos>")
EOS_TOKEN = "<eos>"


def softmax(logits):
    m = max(logits)
    exps = [math.exp(x - m) for x in logits]
    total = sum(exps)
    return [x / total for x in exps]


def score_completion(text, target="AB", ended=True):
    """Reward exact formatted answers and penalize wrong length/content."""
    exact = 1.0 if text == target else 0.0
    prefix_matches = sum(1 for a, b in zip(text, target) if a == b)
    prefix_reward = 0.4 * prefix_matches / max(len(target), 1)
    format_bonus = 0.25 if ended else -0.25
    length_penalty = 0.08 * abs(len(text) - len(target))
    return exact + prefix_reward + format_bonus - length_penalty


@dataclass
class Rollout:
    actions: list
    old_logprob: float
    ref_logprob: float
    text: str
    ended: bool
    reward: float
    value: float


class TinyCharPolicy:
    """Position-wise categorical character policy with a scalar value head."""

    def __init__(self, max_new_tokens=4):
        self.max_new_tokens = max_new_tokens
        self.logits = [[0.0 for _ in VOCAB] for _ in range(max_new_tokens)]
        self.values = [0.0 for _ in range(max_new_tokens)]

    def probs(self, step):
        return softmax(self.logits[step])

    def sample(self, rng):
        actions = []
        text_parts = []
        ended = False
        logprob = 0.0
        value_terms = []
        for step in range(self.max_new_tokens):
            probs = self.probs(step)
            action = rng.choices(range(len(VOCAB)), weights=probs, k=1)[0]
            token = VOCAB[action]
            actions.append(action)
            logprob += math.log(probs[action] + 1e-12)
            value_terms.append(self.values[step])
            if token == EOS_TOKEN:
                ended = True
                break
            text_parts.append(token)
        return actions, "".join(text_parts), ended, logprob, sum(value_terms) / len(value_terms)

    def logprob(self, actions):
        total = 0.0
        for step, action in enumerate(actions):
            probs = self.probs(step)
            total += math.log(probs[action] + 1e-12)
        return total

    def value(self, actions):
        return sum(self.values[: len(actions)]) / len(actions)

    def mean_kl_to(self, reference):
        kls = []
        for step in range(self.max_new_tokens):
            p = self.probs(step)
            q = reference.probs(step)
            kls.append(sum(pi * (math.log(pi + 1e-12) - math.log(qi + 1e-12)) for pi, qi in zip(p, q)))
        return sum(kls) / len(kls)


def collect_rollouts(policy, reference, rng, batch_size, target):
    rollouts = []
    for _ in range(batch_size):
        actions, text, ended, old_logprob, value = policy.sample(rng)
        rollouts.append(
            Rollout(
                actions=actions,
                old_logprob=old_logprob,
                ref_logprob=reference.logprob(actions),
                text=text,
                ended=ended,
                reward=score_completion(text, target=target, ended=ended),
                value=value,
            )
        )
    return rollouts


def update_policy(
    policy,
    reference,
    rollouts,
    learning_rate,
    value_lr,
    clip_range,
    kl_beta,
    target,
):
    rewards = [r.reward for r in rollouts]
    mean_reward = sum(rewards) / len(rewards)
    centered = [r.reward - r.value for r in rollouts]
    mean_adv = sum(centered) / len(centered)
    variance = sum((a - mean_adv) ** 2 for a in centered) / len(centered)
    scale = math.sqrt(variance) + 1e-8

    logit_grads = [[0.0 for _ in VOCAB] for _ in range(policy.max_new_tokens)]
    value_grads = [0.0 for _ in range(policy.max_new_tokens)]
    policy_losses = []
    value_losses = []

    for rollout, advantage in zip(rollouts, centered):
        normalized_advantage = (advantage - mean_adv) / scale
        new_logprob = policy.logprob(rollout.actions)
        ratio = math.exp(new_logprob - rollout.old_logprob)
        clipped = max(1.0 - clip_range, min(1.0 + clip_range, ratio))
        unclipped_obj = ratio * normalized_advantage
        clipped_obj = clipped * normalized_advantage
        active_policy_grad = unclipped_obj <= clipped_obj
        policy_losses.append(-min(unclipped_obj, clipped_obj))
        value_error = policy.value(rollout.actions) - rollout.reward
        value_losses.append(value_error * value_error)

        if active_policy_grad:
            logprob_coeff = -normalized_advantage * ratio
        else:
            logprob_coeff = 0.0

        for step, action in enumerate(rollout.actions):
            probs = policy.probs(step)
            for token_id, prob in enumerate(probs):
                indicator = 1.0 if token_id == action else 0.0
                logit_grads[step][token_id] += logprob_coeff * (indicator - prob) / len(rollouts)
            value_grads[step] += (2.0 * value_error / len(rollout.actions)) / len(rollouts)

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
        policy.values[step] -= value_lr * value_grads[step]

    return {
        "avg_reward": mean_reward,
        "policy_loss": sum(policy_losses) / len(policy_losses),
        "value_loss": sum(value_losses) / len(value_losses),
        "approx_kl_to_reference": policy.mean_kl_to(reference),
        "exact_match_rate": sum(1 for r in rollouts if r.text == target and r.ended) / len(rollouts),
    }


def run_training(args):
    rng = random.Random(args.seed)
    policy = TinyCharPolicy(max_new_tokens=args.max_new_tokens)
    reference = copy.deepcopy(policy)
    history = []

    for step in range(args.steps):
        rollouts = collect_rollouts(policy, reference, rng, args.batch_size, args.target)
        metrics = update_policy(
            policy,
            reference,
            rollouts,
            learning_rate=args.learning_rate,
            value_lr=args.value_learning_rate,
            clip_range=args.clip_range,
            kl_beta=args.kl_beta,
            target=args.target,
        )
        metrics["step"] = step + 1
        history.append(metrics)

    return {
        "schema_version": "limes-rl-toy-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "target": args.target,
            "seed": args.seed,
            "steps": args.steps,
            "batch_size": args.batch_size,
            "max_new_tokens": args.max_new_tokens,
            "clip_range": args.clip_range,
            "kl_beta": args.kl_beta,
            "learning_rate": args.learning_rate,
            "value_learning_rate": args.value_learning_rate,
            "vocab": list(VOCAB),
        },
        "history": history,
        "final": history[-1] if history else {},
        "limitations": [
            "Toy character policy, not a nanoGPT checkpoint.",
            "Verifiable format/exact-match reward only; no helpfulness or capability claim.",
            "KL is measured against the random initialization reference policy.",
        ],
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--target", default="AB")
    parser.add_argument("--max_new_tokens", type=int, default=4)
    parser.add_argument("--clip_range", type=float, default=0.2)
    parser.add_argument("--kl_beta", type=float, default=0.03)
    parser.add_argument("--learning_rate", type=float, default=0.15)
    parser.add_argument("--value_learning_rate", type=float, default=0.05)
    parser.add_argument(
        "--output_json",
        default="out-rl/ppo_toy.json",
        help="Path for the reproducible toy RL run artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.batch_size <= 0:
        raise ValueError("--batch_size must be positive")

    artifact = run_training(args)
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(artifact, f, indent=2, sort_keys=True)

    final = artifact["final"]
    print(
        "ppo_toy "
        f"avg_reward={final['avg_reward']:.4f} "
        f"kl={final['approx_kl_to_reference']:.6f} "
        f"exact_match_rate={final['exact_match_rate']:.3f}"
    )
    print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
