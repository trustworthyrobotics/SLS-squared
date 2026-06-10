#!/usr/bin/env python3
"""Batch runner for pusht/plan/plan_mppi_ilqr_track.py."""

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
from tqdm.auto import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLANNER = REPO_ROOT / "pusht" / "plan" / "plan_mppi_ilqr_track.py"
DEFAULT_OUT_DIR = REPO_ROOT / "pusht" / "plan" / "mppi_ilqr_track_batch"
DEFAULT_SEED_START = 0
DEFAULT_SEED_END = 4
DEFAULT_RUNS_PER_SEED = 40


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--planner", type=Path, default=DEFAULT_PLANNER)
    parser.add_argument("--python-bin", type=Path, default=Path(sys.executable))
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--runs-per-seed", type=int, default=DEFAULT_RUNS_PER_SEED)
    parser.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    parser.add_argument("--seed-end", type=int, default=DEFAULT_SEED_END)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--planner-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to plan_mppi_ilqr_track.py. Repeat for multiple tokens.",
    )
    parser.add_argument("--resume", action="store_true", help="Skip runs whose metrics already exist.")
    return parser.parse_args()


def planner_seed_for_run(base_seed: int, run_idx: int) -> int:
    return int(np.random.SeedSequence([base_seed, run_idx]).generate_state(1, dtype=np.uint32)[0])


def find_metrics(run_dir: Path) -> Path:
    metrics_files = sorted(run_dir.glob("*/metrics.json"))
    if len(metrics_files) != 1:
        raise FileNotFoundError(f"Expected exactly one metrics.json under {run_dir}, found {len(metrics_files)}.")
    return metrics_files[0]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tail_text(path: Path, max_lines: int = 60) -> str:
    with path.open("r", encoding="utf-8") as handle:
        lines = handle.readlines()
    return "".join(lines[-max_lines:]).strip()


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
    planner_seed = planner_seed_for_run(seed, run_idx)
    log_path = run_dir / "planner.log"
    metrics_path: Path | None = None

    if args.resume:
        try:
            metrics_path = find_metrics(run_dir)
        except FileNotFoundError:
            metrics_path = None

    elapsed_s: float | None
    if metrics_path is None:
        command = build_command(args, planner_seed, run_dir)
        started_at = time.time()
        with log_path.open("w", encoding="utf-8") as log_handle:
            log_handle.write(f"Command: {shlex.join(command)}\n")
            log_handle.write(f"CWD: {REPO_ROOT}\n\n")
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
            raise RuntimeError(
                f"Planner failed for seed={seed} run={run_idx} with exit code {process.returncode}. "
                f"See {log_path}.\n\nLast log lines:\n{tail_text(log_path)}"
            )
        metrics_path = find_metrics(run_dir)
    else:
        elapsed_s = None

    metrics = load_json(metrics_path)
    min_position_distance = float(metrics.get("position_goal_distance_min", min(metrics["position_goal_distances"])))
    success = bool(metrics.get("success", metrics.get("stop_reason") == "goal_reached"))
    return {
        "seed": int(seed),
        "run_idx": int(run_idx),
        "planner_seed": int(planner_seed),
        "success": success,
        "min_goal_distance": min_position_distance,
        "position_goal_distance_final": float(metrics.get("position_goal_distance_final", float("nan"))),
        "yaw_goal_distance_min": float(metrics.get("yaw_goal_distance_min", float("nan"))),
        "yaw_goal_distance_final": float(metrics.get("yaw_goal_distance_final", float("nan"))),
        "block_goal_distance_min": float(metrics.get("block_goal_distance_min", float("nan"))),
        "block_goal_distance_final": float(metrics.get("block_goal_distance_final", float("nan"))),
        "stop_reason": str(metrics.get("stop_reason", "")),
        "episode_idx": int(metrics["episode_idx"]),
        "metrics_path": str(metrics_path),
        "planner_log_path": str(log_path),
        "elapsed_s": elapsed_s,
    }


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    total_runs = len(results)
    successes = sum(int(result["success"]) for result in results)
    min_distances = [float(result["min_goal_distance"]) for result in results]
    per_seed: dict[str, dict[str, Any]] = {}
    for result in results:
        bucket = per_seed.setdefault(str(result["seed"]), {"runs": 0, "successes": 0, "min_goal_distances": []})
        bucket["runs"] += 1
        bucket["successes"] += int(result["success"])
        bucket["min_goal_distances"].append(float(result["min_goal_distance"]))
    for bucket in per_seed.values():
        bucket["success_rate"] = bucket["successes"] / bucket["runs"] if bucket["runs"] else 0.0
        bucket["success_percent"] = 100.0 * bucket["success_rate"]
        bucket["avg_min_goal_distance"] = float(np.mean(bucket["min_goal_distances"])) if bucket["min_goal_distances"] else float("nan")
    return {
        "total_runs": total_runs,
        "total_successes": successes,
        "success_rate": successes / total_runs if total_runs else 0.0,
        "success_percent": 100.0 * successes / total_runs if total_runs else 0.0,
        "avg_min_goal_distance": float(np.mean(min_distances)) if min_distances else float("nan"),
        "per_seed": per_seed,
        "results": results,
    }


def main() -> None:
    args = parse_args()
    if args.runs_per_seed < 1:
        raise ValueError("--runs-per-seed must be positive.")
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start.")

    args.planner = args.planner.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser()
    args.out_dir = args.out_dir.expanduser().resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    for seed in range(args.seed_start, args.seed_end + 1):
        pbar = tqdm(range(args.runs_per_seed), desc=f"seed {seed}", unit="run")
        for run_idx in pbar:
            result = run_once(args, seed, run_idx)
            results.append(result)
            summary = summarize(results)
            pbar.set_postfix(
                avg_min=f"{summary['avg_min_goal_distance']:.3f}",
                success_rate=f"{summary['success_rate']:.3f}",
            )
            tqdm.write(
                f"[done] seed={seed} run={run_idx} success={int(result['success'])} "
                f"min_goal_dist={result['min_goal_distance']:.3f} "
                f"min_yaw_dist={result['yaw_goal_distance_min']:.3f} "
                f"avg_min_goal_dist={summary['avg_min_goal_distance']:.3f} "
                f"success_rate={summary['success_rate']:.3f}"
            )

            summary_path = args.out_dir / "batch_summary.json"
            with summary_path.open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2)

    summary = summarize(results)
    summary_path = args.out_dir / "batch_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Summary saved to: {summary_path}")
    print(
        f"Overall: success_rate={summary['success_rate']:.3f} "
        f"avg_min_goal_dist={summary['avg_min_goal_distance']:.3f} "
        f"successes={summary['total_successes']}/{summary['total_runs']}"
    )


if __name__ == "__main__":
    main()
