#!/usr/bin/env python3
"""Generate random Reacher start-goal pairs for planning."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


DEFAULT_OUT_PATH = Path("reacher/plan/start_goal.pt")
DEFAULT_PAIR_COUNT = 100
DEFAULT_START_CENTER = np.array([3.1, -2.25], dtype=np.float32)
DEFAULT_START_HALF_WIDTH = np.array([0.05, 0.1], dtype=np.float32)
DEFAULT_GOAL_QPOS = np.array([0.37, -2.09], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--pair-count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.pair_count <= 0:
        raise ValueError(f"--pair-count must be positive, got {args.pair_count}.")

    rng = np.random.default_rng(args.seed)
    lows = DEFAULT_START_CENTER - DEFAULT_START_HALF_WIDTH
    highs = DEFAULT_START_CENTER + DEFAULT_START_HALF_WIDTH
    starts = rng.uniform(lows, highs, size=(args.pair_count, 2)).astype(np.float32)
    goal = DEFAULT_GOAL_QPOS.astype(np.float32)
    zero_qvel = np.zeros((2,), dtype=np.float32)

    pairs = []
    for idx in range(args.pair_count):
        pairs.append(
            {
                "start": {
                    "qpos": starts[idx],
                    "qvel": zero_qvel.copy(),
                },
                "goal": {
                    "qpos": goal.copy(),
                    "qvel": zero_qvel.copy(),
                },
            }
        )

    payload = {
        "pairs": pairs,
        "metadata": {
            "pair_count": int(args.pair_count),
            "seed": int(args.seed),
            "start_center": DEFAULT_START_CENTER.tolist(),
            "start_half_width": DEFAULT_START_HALF_WIDTH.tolist(),
            "goal_qpos": DEFAULT_GOAL_QPOS.tolist(),
        },
    }
    out_path = args.out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)

    print(f"Saved {args.pair_count} start-goal pairs to {out_path}")
    print(f"start range: low={lows.tolist()}, high={highs.tolist()}")
    print(f"goal qpos: {goal.tolist()}")


if __name__ == "__main__":
    main()
