#!/usr/bin/env python3
"""Compute small efficiency metrics for nanoGPT-scale experiments."""

import argparse
import json
import math
import os
from datetime import datetime, timezone


def bits_per_byte_from_nats(loss_nats_per_token, token_count, raw_byte_count):
    """Convert average token NLL in nats into tokenizer-agnostic BPB."""
    if token_count <= 0:
        raise ValueError("token_count must be positive")
    if raw_byte_count <= 0:
        raise ValueError("raw_byte_count must be positive")
    total_bits = loss_nats_per_token * token_count / math.log(2)
    return total_bits / raw_byte_count


def score_artifact(path, max_artifact_mib):
    artifact_bytes = os.path.getsize(path)
    max_bytes = int(max_artifact_mib * 1024 * 1024)
    return {
        "path": path,
        "bytes": artifact_bytes,
        "mib": artifact_bytes / (1024 * 1024),
        "max_mib": max_artifact_mib,
        "within_limit": artifact_bytes <= max_bytes,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", required=True, help="Model/checkpoint artifact to size.")
    parser.add_argument("--loss_nats", type=float, required=True)
    parser.add_argument("--tokens", type=int, required=True)
    parser.add_argument(
        "--raw_bytes",
        type=int,
        required=True,
        help="Raw evaluation corpus size in bytes, independent of tokenizer.",
    )
    parser.add_argument("--max_artifact_mib", type=float, default=16.0)
    parser.add_argument(
        "--output_json",
        default=None,
        help="Optional path for a JSON efficiency artifact.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    artifact = score_artifact(args.artifact, args.max_artifact_mib)
    bpb = bits_per_byte_from_nats(args.loss_nats, args.tokens, args.raw_bytes)
    result = {
        "schema_version": "limes-efficiency-v0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact": artifact,
        "metrics": {
            "loss_nats_per_token": args.loss_nats,
            "token_count": args.tokens,
            "raw_byte_count": args.raw_bytes,
            "bits_per_byte": bpb,
        },
        "limitations": [
            "BPB is comparable only when the same raw byte corpus and split are used.",
            "Artifact-size checks do not measure latency, memory bandwidth, or quality.",
        ],
    }

    print(
        f"artifact_mib={artifact['mib']:.4f} "
        f"within_limit={artifact['within_limit']} "
        f"bpb={bpb:.6f}"
    )
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, sort_keys=True)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
