#!/usr/bin/env python3
"""Compute a calibrated constant Markov-state error covariance for Reacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_DATA_PATH = Path("reacher/eval/reacher_one_step_error_data_9.pt")
DEFAULT_OUT_FILE = Path("reacher/eval/fixed_error_covariance_9.pt")


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    n = int(scores.numel())
    if n == 0:
        raise ValueError("Cannot calibrate quantile from zero scores.")
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(torch.sort(scores.detach().cpu()).values[rank - 1].item())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument("--center-errors", action="store_true")
    args = parser.parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1).")

    data = torch.load(args.data_path.expanduser(), map_location="cpu", weights_only=False)
    errors = data["error"].float()
    if errors.ndim != 2:
        raise ValueError(f"Expected error tensor [N, state_dim], got {tuple(errors.shape)}.")
    n_total, state_dim = errors.shape
    n_fit = n_total // 2
    if n_fit < 2:
        raise ValueError(f"Need at least two fit errors, got {n_fit}.")

    fit_errors = errors[:n_fit]
    error_mean = fit_errors.mean(dim=0) if args.center_errors else torch.zeros(state_dim)
    centered = fit_errors - error_mean.unsqueeze(0)
    covariance = centered.T @ centered / float(n_fit - 1)
    covariance = covariance + torch.eye(state_dim, dtype=covariance.dtype) * float(args.jitter)
    cholesky = torch.linalg.cholesky(covariance)
    whitened = torch.linalg.solve_triangular(cholesky, centered.T, upper=False).T
    scores = torch.linalg.vector_norm(whitened, ord=2, dim=1)
    q_fixed = conformal_quantile(scores, args.alpha)
    calibrated_cholesky = cholesky * q_fixed
    calibrated_covariance = calibrated_cholesky @ calibrated_cholesky.T
    calibrated_eigvals = torch.linalg.eigvalsh(calibrated_covariance)

    payload = {
        "covariance": covariance,
        "cholesky": cholesky,
        "q_fixed": torch.tensor(q_fixed, dtype=torch.float32),
        "calibrated_cholesky": calibrated_cholesky,
        "calibrated_covariance": calibrated_covariance,
        "calibrated_covariance_eigvals": calibrated_eigvals,
        "error_mean": error_mean,
        "alpha": torch.tensor(float(args.alpha), dtype=torch.float32),
        "jitter": torch.tensor(float(args.jitter), dtype=torch.float32),
        "center_errors": torch.tensor(bool(args.center_errors)),
        "num_total": torch.tensor(n_total),
        "num_fit": torch.tensor(n_fit),
        "state_dim": torch.tensor(state_dim),
    }
    out_file = args.out_file.expanduser()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_file)
    summary = {
        "data_path": str(args.data_path),
        "out_file": str(out_file),
        "alpha": float(args.alpha),
        "jitter": float(args.jitter),
        "center_errors": bool(args.center_errors),
        "num_total": int(n_total),
        "num_fit": int(n_fit),
        "state_dim": int(state_dim),
        "q_fixed": float(q_fixed),
        "covariance_trace": float(torch.trace(covariance).item()),
        "score_mean": float(scores.mean().item()),
        "score_max": float(scores.max().item()),
        "calibrated_covariance_eigvals": calibrated_eigvals.detach().cpu().numpy().astype(float).tolist(),
    }
    with out_file.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"Saved fixed covariance calibration to {out_file}")
    print(f"q_fixed={q_fixed:.8g}")
    print("calibrated_cholesky:")
    print(calibrated_cholesky.detach().cpu().numpy())
    print("calibrated_covariance_eigvals:")
    print(calibrated_eigvals.detach().cpu().numpy())


if __name__ == "__main__":
    main()
