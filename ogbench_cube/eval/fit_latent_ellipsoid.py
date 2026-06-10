#!/usr/bin/env python3
"""Fit conformal in-domain ellipsoids for OGBench cube latents and Markov states."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import numpy as np
import torch
from tqdm import tqdm

from ogbench_cube.train.mlpdyn_train import build_markov_state, preprocess_pixels, required_markov_history


DEFAULT_DATASET_PATH = Path("ogbench_cube/data/test_data/ogbench_cube_test.h5")
DEFAULT_MODEL_DIR = Path("ogbench_cube/models/mlpdyn_embd_12_strtn")
DEFAULT_OUT_DIR = Path("ogbench_cube/eval/latent_ellipsoid")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--markov-deriv", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.01, help="Target miscoverage. alpha=0.01 gives 99% coverage.")
    parser.add_argument("--ridge", type=float, default=1e-5, help="Diagonal regularizer added to zero-centered covariance.")
    parser.add_argument("--save-latents", action="store_true", help="Also save all encoded latents in the NPZ artifact.")
    return parser.parse_args()


def latest_object_checkpoint(model_dir: Path) -> Path:
    patterns = (
        re.compile(r".*_epoch_(\d+)_object\.ckpt$"),
        re.compile(r".*_epoch[=_](\d+).*\.ckpt$"),
    )
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*.ckpt"):
        for pattern in patterns:
            match = pattern.match(path.name)
            if match is not None:
                candidates.append((int(match.group(1)), path))
                break
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    pixels_np: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    frame_batch_size: int,
) -> np.ndarray:
    pixels = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).contiguous()
    pixels = preprocess_pixels(pixels.unsqueeze(0), img_size)[0]

    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb.detach().cpu().numpy().astype(np.float64))
    return np.concatenate(latents, axis=0)


def build_all_markov_states(latents: np.ndarray, markov_deriv: int) -> np.ndarray:
    latents_t = torch.from_numpy(np.asarray(latents, dtype=np.float32))
    history_len = required_markov_history(markov_deriv)
    states = []
    for t in range(latents_t.shape[0]):
        start = max(0, t - history_len + 1)
        history = latents_t[start : t + 1]
        if history.shape[0] < history_len:
            history = torch.cat((history[:1].repeat(history_len - history.shape[0], 1), history), dim=0)
        states.append(build_markov_state(history.unsqueeze(0), markov_deriv)[0])
    return torch.stack(states, dim=0).numpy().astype(np.float64)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if scores.ndim != 1 or scores.shape[0] == 0:
        raise ValueError("scores must be a nonempty 1D array.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    n = scores.shape[0]
    rank = int(np.ceil((n + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), n)
    return float(scores[rank - 1])


def fit_zero_centered_ellipsoid(samples: np.ndarray, *, alpha: float, ridge: float) -> dict[str, np.ndarray | float | int]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"Expected samples with shape [N, D], got {samples.shape}.")
    if samples.shape[0] <= samples.shape[1]:
        raise ValueError(
            f"Need more samples than dimensions to fit a stable ellipsoid, got N={samples.shape[0]}, D={samples.shape[1]}."
        )

    dim = samples.shape[1]
    second_moment = (samples.T @ samples) / float(samples.shape[0])
    covariance = second_moment + float(ridge) * np.eye(dim, dtype=np.float64)
    precision = np.linalg.pinv(covariance, hermitian=True)

    scores = np.einsum("ni,ij,nj->n", samples, precision, samples)
    threshold = conformal_quantile(scores, alpha)
    unit_precision = precision / max(threshold, 1e-12)
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    aligned_samples = samples @ eigvecs
    aligned_axis_max_abs = np.max(np.abs(aligned_samples), axis=0)
    calibrated_axis_radius = np.sqrt(np.maximum(threshold * eigvals, 0.0))
    coverage = float(np.mean(scores <= threshold))

    return {
        "center": np.zeros(dim, dtype=np.float64),
        "covariance": covariance,
        "precision": precision,
        "threshold": float(threshold),
        "unit_precision": unit_precision,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "aligned_axis_max_abs": aligned_axis_max_abs,
        "calibrated_axis_radius": calibrated_axis_radius,
        "scores": scores,
        "coverage": coverage,
        "min_covariance_eig": float(eigvals.min()),
        "max_covariance_eig": float(eigvals.max()),
        "num_samples": int(samples.shape[0]),
        "dim": int(dim),
    }


def add_prefixed_arrays(payload: dict[str, np.ndarray], prefix: str, artifact: dict[str, np.ndarray | float | int]) -> None:
    for key in (
        "center",
        "covariance",
        "precision",
        "unit_precision",
        "eigvals",
        "eigvecs",
        "aligned_axis_max_abs",
        "calibrated_axis_radius",
        "scores",
    ):
        payload[f"{prefix}_{key}"] = np.asarray(artifact[key])
    payload[f"{prefix}_threshold"] = np.asarray(artifact["threshold"], dtype=np.float64)


def torch_artifact(prefix: str, artifact: dict[str, np.ndarray | float | int]) -> dict[str, object]:
    return {
        f"{prefix}_center": torch.from_numpy(np.asarray(artifact["center"], dtype=np.float32)),
        f"{prefix}_covariance": torch.from_numpy(np.asarray(artifact["covariance"], dtype=np.float32)),
        f"{prefix}_precision": torch.from_numpy(np.asarray(artifact["precision"], dtype=np.float32)),
        f"{prefix}_unit_precision": torch.from_numpy(np.asarray(artifact["unit_precision"], dtype=np.float32)),
        f"{prefix}_eigvals": torch.from_numpy(np.asarray(artifact["eigvals"], dtype=np.float32)),
        f"{prefix}_eigvecs": torch.from_numpy(np.asarray(artifact["eigvecs"], dtype=np.float32)),
        f"{prefix}_aligned_axis_max_abs": torch.from_numpy(np.asarray(artifact["aligned_axis_max_abs"], dtype=np.float32)),
        f"{prefix}_calibrated_axis_radius": torch.from_numpy(np.asarray(artifact["calibrated_axis_radius"], dtype=np.float32)),
        f"{prefix}_threshold": float(artifact["threshold"]),
    }


def json_artifact(artifact: dict[str, np.ndarray | float | int]) -> dict[str, object]:
    return {
        "threshold": float(artifact["threshold"]),
        "dim": int(artifact["dim"]),
        "num_samples": int(artifact["num_samples"]),
        "coverage": float(artifact["coverage"]),
        "aligned_axis_max_abs": np.asarray(artifact["aligned_axis_max_abs"], dtype=np.float64).tolist(),
        "calibrated_axis_radius": np.asarray(artifact["calibrated_axis_radius"], dtype=np.float64).tolist(),
        "covariance_eigvals_descending": np.asarray(artifact["eigvals"], dtype=np.float64).tolist(),
        "min_covariance_eig": float(artifact["min_covariance_eig"]),
        "max_covariance_eig": float(artifact["max_covariance_eig"]),
    }


def print_aligned_axis_summary(name: str, artifact: dict[str, np.ndarray | float | int]) -> None:
    print(f"Max absolute {name} value on each covariance-aligned axis:")
    for axis_idx, (max_abs, radius) in enumerate(
        zip(
            np.asarray(artifact["aligned_axis_max_abs"], dtype=np.float64),
            np.asarray(artifact["calibrated_axis_radius"], dtype=np.float64),
        )
    ):
        print(f"  axis {axis_idx:02d}: max_abs={max_abs:.6g}, calibrated_radius={radius:.6g}")


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "config.json").open() as f:
        config = json.load(f)
    img_size = int(args.img_size if args.img_size is not None else config.get("img_size", 224))
    markov_deriv = int(args.markov_deriv if args.markov_deriv is not None else config.get("markov_deriv", 1))

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    device = resolve_device(args.device)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device).eval()
    model.requires_grad_(False)

    all_latents: list[np.ndarray] = []
    all_markov_states: list[np.ndarray] = []
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        episode_indices = np.arange(ep_len.shape[0], dtype=np.int64)
        if args.max_episodes is not None:
            episode_indices = episode_indices[: int(args.max_episodes)]

        for ep_idx in tqdm(episode_indices, desc="Encoding OGBench cube latents"):
            offset = int(ep_offset[ep_idx])
            length = int(ep_len[ep_idx])
            if length <= 0:
                continue
            rows = np.arange(offset, offset + length, dtype=np.int64)
            pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
            episode_latents = encode_pixels(
                model,
                pixels_np,
                device=device,
                img_size=img_size,
                frame_batch_size=int(args.frame_batch_size),
            )
            all_latents.append(episode_latents)
            all_markov_states.append(build_all_markov_states(episode_latents, markov_deriv))

    latents = np.concatenate(all_latents, axis=0)
    markov_states = np.concatenate(all_markov_states, axis=0)
    latent_artifact = fit_zero_centered_ellipsoid(latents, alpha=float(args.alpha), ridge=float(args.ridge))
    markov_artifact = fit_zero_centered_ellipsoid(markov_states, alpha=float(args.alpha), ridge=float(args.ridge))

    metadata = {
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "img_size": img_size,
        "markov_deriv": markov_deriv,
        "alpha": float(args.alpha),
        "target_coverage": float(1.0 - args.alpha),
        "ridge": float(args.ridge),
        "num_samples": int(latent_artifact["num_samples"]),
        "latent_dim": int(latent_artifact["dim"]),
        "markov_state_dim": int(markov_artifact["dim"]),
        "latent_coverage": float(latent_artifact["coverage"]),
        "markov_coverage": float(markov_artifact["coverage"]),
        "latent_score_definition": "z.T @ latent_precision @ z",
        "markov_score_definition": "x.T @ markov_precision @ x, where x=[z, delta_z, ...] from build_markov_state",
        "latent_in_distribution_test": "z.T @ latent_unit_precision @ z <= 1",
        "markov_in_distribution_test": "x.T @ markov_unit_precision @ x <= 1",
        "center": "zero",
    }

    npz_payload: dict[str, np.ndarray] = {}
    add_prefixed_arrays(npz_payload, "latent", latent_artifact)
    add_prefixed_arrays(npz_payload, "markov", markov_artifact)
    if args.save_latents:
        npz_payload["latents"] = latents.astype(np.float32)
        npz_payload["markov_states"] = markov_states.astype(np.float32)
    np.savez(out_dir / "latent_ellipsoid.npz", **npz_payload)

    torch.save(
        {
            **torch_artifact("latent", latent_artifact),
            **torch_artifact("markov", markov_artifact),
            "alpha": float(args.alpha),
            "metadata": metadata,
        },
        out_dir / "latent_ellipsoid.pt",
    )
    with (out_dir / "latent_ellipsoid.json").open("w") as f:
        json.dump(
            {
                **metadata,
                "latent": json_artifact(latent_artifact),
                "markov": json_artifact(markov_artifact),
            },
            f,
            indent=2,
        )

    print(
        f"Saved zero-centered OGBench cube latent and Markov ellipsoids to {out_dir}. "
        f"Latent coverage={float(latent_artifact['coverage']):.6f}; "
        f"Markov coverage={float(markov_artifact['coverage']):.6f}; "
        f"samples={int(latent_artifact['num_samples'])}."
    )
    print_aligned_axis_summary("latent", latent_artifact)
    print_aligned_axis_summary("Markov-state", markov_artifact)


if __name__ == "__main__":
    main()
