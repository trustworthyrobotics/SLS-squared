#!/usr/bin/env python3
"""Fit zero-centered latent and Markov-state ellipsoids for Reacher."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm


DEFAULT_DATASET_PATH = Path("reacher/data/test_data_50hz/reacher_test.h5")
DEFAULT_MODEL_DIR = Path("reacher/models/mlpdyn_ft_6")
DEFAULT_OUT_DIR = Path("reacher/eval/latent_ellipsoid")


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


def preprocess_pixels(pixels_np: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    pixels = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(pixels.shape[-2:]) != (img_size, img_size):
        pixels = torch.nn.functional.interpolate(pixels, size=(img_size, img_size), mode="bilinear", align_corners=False)
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return (pixels.to(device) - mean) / std


@torch.no_grad()
def encode_pixels(model: torch.nn.Module, pixels_np: np.ndarray, *, device: torch.device, img_size: int, frame_batch_size: int) -> np.ndarray:
    pixels = preprocess_pixels(pixels_np, img_size, device)
    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        output = model.encoder(pixels[start : start + frame_batch_size], interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]).detach().cpu().numpy().astype(np.float64))
    return np.concatenate(latents, axis=0)


def build_markov_states(latents: np.ndarray) -> np.ndarray:
    deltas = np.zeros_like(latents)
    deltas[1:] = latents[1:] - latents[:-1]
    return np.concatenate((latents, deltas), axis=-1).astype(np.float64)


def conformal_quantile(scores: np.ndarray, alpha: float) -> float:
    scores = np.sort(np.asarray(scores, dtype=np.float64))
    if scores.ndim != 1 or scores.shape[0] == 0:
        raise ValueError("scores must be a nonempty 1D array.")
    rank = int(np.ceil((scores.shape[0] + 1) * (1.0 - alpha)))
    rank = min(max(rank, 1), scores.shape[0])
    return float(scores[rank - 1])


def fit_zero_centered_ellipsoid(samples: np.ndarray, alpha: float, ridge: float) -> dict[str, object]:
    samples = np.asarray(samples, dtype=np.float64)
    if samples.ndim != 2:
        raise ValueError(f"Expected [N, D] samples, got {samples.shape}.")
    if samples.shape[0] <= samples.shape[1]:
        raise ValueError(f"Need more samples than dimensions, got N={samples.shape[0]}, D={samples.shape[1]}.")
    dim = samples.shape[1]
    covariance = (samples.T @ samples) / float(samples.shape[0])
    covariance = covariance + float(ridge) * np.eye(dim, dtype=np.float64)
    precision = np.linalg.pinv(covariance, hermitian=True)
    scores = np.einsum("ni,ij,nj->n", samples, precision, samples)
    threshold = conformal_quantile(scores, alpha)
    unit_precision = precision / max(threshold, 1e-12)
    coverage = float(np.mean(scores <= threshold))
    eigvals, eigvecs = np.linalg.eigh(covariance)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    aligned = samples @ eigvecs
    return {
        "center": np.zeros(dim, dtype=np.float64),
        "covariance": covariance,
        "precision": precision,
        "threshold": float(threshold),
        "unit_precision": unit_precision,
        "eigvals": eigvals,
        "eigvecs": eigvecs,
        "aligned_axis_max_abs": np.max(np.abs(aligned), axis=0),
        "calibrated_axis_radius": np.sqrt(np.maximum(threshold * eigvals, 0.0)),
        "scores": scores,
        "coverage": coverage,
        "num_samples": int(samples.shape[0]),
        "dim": int(dim),
    }


def add_prefixed_arrays(payload: dict[str, np.ndarray], prefix: str, artifact: dict[str, object]) -> None:
    for key in ("center", "covariance", "precision", "unit_precision", "eigvals", "eigvecs", "aligned_axis_max_abs", "calibrated_axis_radius", "scores"):
        payload[f"{prefix}_{key}"] = np.asarray(artifact[key])
    payload[f"{prefix}_threshold"] = np.asarray(artifact["threshold"], dtype=np.float64)


def torch_artifact(prefix: str, artifact: dict[str, object]) -> dict[str, object]:
    keys = ("center", "covariance", "precision", "unit_precision", "eigvals", "eigvecs", "aligned_axis_max_abs", "calibrated_axis_radius")
    result = {f"{prefix}_{key}": torch.from_numpy(np.asarray(artifact[key], dtype=np.float32)) for key in keys}
    result[f"{prefix}_threshold"] = float(artifact["threshold"])
    return result


def json_artifact(artifact: dict[str, object]) -> dict[str, object]:
    return {
        "threshold": float(artifact["threshold"]),
        "dim": int(artifact["dim"]),
        "num_samples": int(artifact["num_samples"]),
        "coverage": float(artifact["coverage"]),
        "aligned_axis_max_abs": np.asarray(artifact["aligned_axis_max_abs"], dtype=np.float64).tolist(),
        "calibrated_axis_radius": np.asarray(artifact["calibrated_axis_radius"], dtype=np.float64).tolist(),
        "covariance_eigvals_descending": np.asarray(artifact["eigvals"], dtype=np.float64).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=128)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--ridge", type=float, default=1e-5)
    parser.add_argument("--save-latents", action="store_true")
    args = parser.parse_args()

    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    with (model_dir / "config.json").open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    img_size = int(args.img_size if args.img_size is not None else config.get("img_size", 224))
    checkpoint = args.checkpoint.expanduser().resolve() if args.checkpoint else latest_object_checkpoint(model_dir).resolve()
    device = resolve_device(args.device)
    model = torch.load(checkpoint, map_location=device, weights_only=False).to(device).eval()
    model.requires_grad_(False)

    all_latents = []
    all_markov = []
    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        episode_indices = np.arange(ep_len.shape[0], dtype=np.int64)
        if args.max_episodes is not None:
            episode_indices = episode_indices[: int(args.max_episodes)]
        for ep_idx in tqdm(episode_indices, desc="Encoding reacher latents"):
            rows = np.arange(int(ep_offset[ep_idx]), int(ep_offset[ep_idx]) + int(ep_len[ep_idx]), dtype=np.int64)
            if rows.size == 0:
                continue
            latents = encode_pixels(
                model,
                np.asarray(h5["pixels"][rows], dtype=np.uint8),
                device=device,
                img_size=img_size,
                frame_batch_size=args.frame_batch_size,
            )
            all_latents.append(latents)
            all_markov.append(build_markov_states(latents))
    latents_np = np.concatenate(all_latents, axis=0)
    markov_np = np.concatenate(all_markov, axis=0)
    latent_artifact = fit_zero_centered_ellipsoid(latents_np, args.alpha, args.ridge)
    markov_artifact = fit_zero_centered_ellipsoid(markov_np, args.alpha, args.ridge)

    np_payload: dict[str, np.ndarray] = {}
    add_prefixed_arrays(np_payload, "latent", latent_artifact)
    add_prefixed_arrays(np_payload, "markov", markov_artifact)
    if args.save_latents:
        np_payload["latents"] = latents_np
        np_payload["markov_states"] = markov_np
    np.savez(out_dir / "latent_ellipsoid.npz", **np_payload)
    torch.save(
        {
            **torch_artifact("latent", latent_artifact),
            **torch_artifact("markov", markov_artifact),
            "alpha": float(args.alpha),
            "ridge": float(args.ridge),
            "dataset_path": str(dataset_path),
            "model_dir": str(model_dir),
            "checkpoint": str(checkpoint),
        },
        out_dir / "latent_ellipsoid.pt",
    )
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "dataset_path": str(dataset_path),
                "model_dir": str(model_dir),
                "checkpoint": str(checkpoint),
                "alpha": float(args.alpha),
                "ridge": float(args.ridge),
                "latent": json_artifact(latent_artifact),
                "markov": json_artifact(markov_artifact),
            },
            handle,
            indent=2,
        )
    print("latent_covariance_eigvals_descending:")
    print(np.asarray(latent_artifact["eigvals"], dtype=np.float64))
    print("markov_covariance_eigvals_descending:")
    print(np.asarray(markov_artifact["eigvals"], dtype=np.float64))
    print(f"Saved latent ellipsoid artifacts to {out_dir}")


if __name__ == "__main__":
    main()
