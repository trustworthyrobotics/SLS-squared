#!/usr/bin/env python3
"""Compute a calibrated constant error covariance from OGBench one-step error data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch


DEFAULT_DATA_PATH = "ogbench_cube/eval/ogbench_one_step_error_data_embed_12_strtn.pt"
DEFAULT_OUT_FILE = "ogbench_cube/eval/fixed_error_covariance_12_strtn.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--out-file", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--jitter", type=float, default=1e-6)
    parser.add_argument(
        "--fit-fraction",
        type=float,
        default=0.5,
        help="Fraction of errors used to fit and calibrate the fixed covariance.",
    )
    parser.add_argument(
        "--center-errors",
        action="store_true",
        help="Subtract the empirical mean before computing covariance and scores.",
    )
    return parser.parse_args()


def conformal_quantile(scores: torch.Tensor, alpha: float) -> float:
    n = int(scores.numel())
    if n == 0:
        raise ValueError("Cannot calibrate quantile from zero scores.")
    k = int(np.ceil((n + 1) * (1.0 - alpha)))
    sorted_scores = torch.sort(scores.detach().cpu()).values
    return float(sorted_scores[min(k - 1, n - 1)].item())


def main() -> None:
    args = parse_args()
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1).")
    if not 0.0 < args.fit_fraction <= 1.0:
        raise ValueError("--fit-fraction must be in (0, 1].")

    data = torch.load(args.data_path.expanduser(), map_location="cpu")
    if "error" not in data:
        raise KeyError(f"{args.data_path} does not contain an 'error' tensor.")
    errors = data["error"].float()
    if errors.ndim != 2:
        raise ValueError(f"Expected error tensor with shape [N, state_dim], got {tuple(errors.shape)}.")

    finite_mask = torch.isfinite(errors).all(dim=1)
    errors = errors[finite_mask]
    n_total, state_dim = errors.shape
    n_fit = int(np.floor(n_total * float(args.fit_fraction)))
    if n_fit < 2:
        raise ValueError(f"Need at least two finite errors for fitting; got {n_fit}.")

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

    payload = {
        "covariance": covariance,
        "cholesky": cholesky,
        "q_fixed": torch.tensor(q_fixed, dtype=torch.float32),
        "calibrated_cholesky": calibrated_cholesky,
        "error_mean": error_mean,
        "alpha": torch.tensor(float(args.alpha), dtype=torch.float32),
        "jitter": torch.tensor(float(args.jitter), dtype=torch.float32),
        "fit_fraction": torch.tensor(float(args.fit_fraction), dtype=torch.float32),
        "center_errors": torch.tensor(bool(args.center_errors)),
        "num_total": torch.tensor(n_total),
        "num_fit": torch.tensor(n_fit),
        "num_dropped_nonfinite": torch.tensor(int(finite_mask.numel() - finite_mask.sum().item())),
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
        "fit_fraction": float(args.fit_fraction),
        "center_errors": bool(args.center_errors),
        "num_total": int(n_total),
        "num_fit": int(n_fit),
        "num_dropped_nonfinite": int(finite_mask.numel() - finite_mask.sum().item()),
        "state_dim": int(state_dim),
        "q_fixed": q_fixed,
        "covariance_trace": float(torch.trace(covariance).item()),
        "covariance_diag_min": float(torch.diag(covariance).min().item()),
        "covariance_diag_max": float(torch.diag(covariance).max().item()),
        "score_mean": float(scores.mean().item()),
        "score_max": float(scores.max().item()),
    }
    with out_file.with_suffix(".json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"Saved fixed covariance calibration to {out_file}")
    print(f"q_fixed={q_fixed:.6f}, state_dim={state_dim}, fit_samples={n_fit}")


if __name__ == "__main__":
    main()
