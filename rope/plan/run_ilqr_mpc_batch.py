#!/usr/bin/env python3
"""Batch runner for rope/plan/plan_ilqr_mpc.py."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER = REPO_ROOT / "rope" / "plan" / "plan_ilqr_mpc.py"
DEFAULT_OUT_DIR = REPO_ROOT / "rope" / "plan" / "ilqr_mpc_batch"
DEFAULT_THRESHOLD = 0.075


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runs-per-seed", type=int, default=40)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=1)
    parser.add_argument("--success-threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device forwarded to plan_ilqr_mpc.py. Defaults to the planner's native auto selection.",
    )
    parser.add_argument(
        "--planner-arg",
        action="append",
        default=[],
        help="Extra argument to forward to plan_ilqr_mpc.py. Repeat this flag for multiple args.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip runs whose metrics already exist.")
    return parser.parse_args()


def find_metrics(run_dir: Path) -> Path:
    metrics_files = sorted(run_dir.glob("*/metrics.json"))
    if len(metrics_files) != 1:
        raise FileNotFoundError(f"Expected exactly one metrics.json under {run_dir}, found {len(metrics_files)}.")
    return metrics_files[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tail_text(path: Path, max_lines: int = 40) -> str:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return "".join(lines[-max_lines:]).strip()


def compute_success(metrics: dict[str, Any], threshold: float) -> tuple[bool, float]:
    distances = [float(value) for value in metrics.get("task_target_distances", [])]
    if not distances:
        raise ValueError("metrics.json does not contain task_target_distances.")
    min_distance = min(distances)
    return min_distance < threshold, min_distance


def planner_seed_for_run(base_seed: int, run_idx: int) -> int:
    return int(np.random.SeedSequence([base_seed, run_idx]).generate_state(1, dtype=np.uint32)[0])


def build_command(args: argparse.Namespace, planner_seed: int, run_dir: Path) -> list[str]:
    command = [
        str(args.python_bin),
        str(args.planner),
        "--seed",
        str(planner_seed),
        "--device",
        str(args.device),
        "--out-dir",
        str(run_dir),
    ]
    command.extend(args.planner_arg)
    return command


def run_once(args: argparse.Namespace, seed: int, run_idx: int) -> dict[str, Any]:
    run_dir = args.out_dir / f"seed_{seed:02d}" / f"run_{run_idx:03d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = None
    planner_seed = planner_seed_for_run(seed, run_idx)

    if args.resume:
        try:
            metrics_path = find_metrics(run_dir)
        except FileNotFoundError:
            metrics_path = None

    if metrics_path is None:
        command = build_command(args, planner_seed, run_dir)
        log_path = run_dir / "planner.log"
        started_at = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"Command: {shlex.join(command)}\n")
            log_handle.write(f"CWD: {REPO_ROOT}\n")
            log_handle.write("\n")
            log_handle.flush()
            process = subprocess.run(
                command,
                cwd=REPO_ROOT,
                env={**os.environ, "PYTHONPATH": str(REPO_ROOT)},
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
        elapsed_s = time.time() - started_at
        if process.returncode != 0:
            log_tail = tail_text(log_path)
            raise RuntimeError(
                f"Planner failed for seed={seed} run={run_idx} with exit code {process.returncode}. "
                f"See {log_path}.\n\nLast log lines:\n{log_tail}"
            )
        metrics_path = find_metrics(run_dir)
    else:
        elapsed_s = None
        log_path = run_dir / "planner.log"

    metrics = load_json(metrics_path)
    success, min_distance = compute_success(metrics, args.success_threshold)
    return {
        "seed": seed,
        "planner_seed": planner_seed,
        "run_idx": run_idx,
        "success": success,
        "min_task_target_distance": min_distance,
        "metrics_path": str(metrics_path),
        "planner_log_path": str(log_path),
        "episode_idx": int(metrics["episode_idx"]),
        "episode_seed": int(metrics["episode_seed"]),
        "num_executed_steps": int(metrics["num_executed_steps"]),
        "elapsed_s": elapsed_s,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = len(results)
    total_successes = sum(1 for result in results if result["success"])
    per_seed: dict[str, dict[str, Any]] = {}
    for result in results:
        seed_key = str(result["seed"])
        bucket = per_seed.setdefault(seed_key, {"runs": 0, "successes": 0})
        bucket["runs"] += 1
        bucket["successes"] += int(result["success"])
    for bucket in per_seed.values():
        bucket["success_percent"] = 100.0 * bucket["successes"] / bucket["runs"]
    return {
        "total_runs": total_runs,
        "total_successes": total_successes,
        "success_percent": 100.0 * total_successes / total_runs if total_runs else 0.0,
        "success_threshold": None,
        "per_seed": per_seed,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    args.planner = args.planner.resolve()
    args.python_bin = args.python_bin.expanduser()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_end + 1):
        for run_idx in range(args.runs_per_seed):
            planner_seed = planner_seed_for_run(seed, run_idx)
            print(f"[run] seed={seed} run={run_idx} planner_seed={planner_seed}")
            result = run_once(args, seed, run_idx)
            results.append(result)
            tag = "success" if result["success"] else "failure"
            print(
                f"[done] seed={seed} run={run_idx} planner_seed={result['planner_seed']} {tag} "
                f"min_task_target_distance={result['min_task_target_distance']:.6f}"
            )

    summary = summarize(results)
    summary["success_threshold"] = float(args.success_threshold)

    summary_path = args.out_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print()
    print(f"Summary saved to: {summary_path}")
    print(f"Overall success: {summary['total_successes']}/{summary['total_runs']} ({summary['success_percent']:.2f}%)")
    for seed_key, bucket in sorted(summary["per_seed"].items(), key=lambda item: int(item[0])):
        print(
            f"Seed {seed_key}: {bucket['successes']}/{bucket['runs']} "
            f"({bucket['success_percent']:.2f}%)"
        )


if __name__ == "__main__":
    main()
