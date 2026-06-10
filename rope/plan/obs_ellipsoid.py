#!/usr/bin/env python3
"""Build a dimension-wise conformal ellipsoid for collected rope obstacle latents."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

DEFAULT_DATA_PATH = "rope/plan/obstacle_data/obstacle_classifier_data.pt"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn_noshadow"
DEFAULT_OUT_DIR = "rope/plan/obs_ellipsoid"
DEFAULT_ALPHA = 0.15
DEFAULT_EPSILON = 1e-12
DEFAULT_LABEL = "obstacle"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--epsilon", type=float, default=DEFAULT_EPSILON)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--label",
        choices=("obstacle", "non_obstacle", "all"),
        default=DEFAULT_LABEL,
        help="Subset of collected rendered states used as the target distribution.",
    )
    parser.add_argument("--artifact-name", type=str, default="ellipsoid.pt")
    return parser.parse_args()


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoints matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_config(model_dir: Path) -> dict[str, object]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Model config not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def load_world_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    pixel_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return pixel_mean, pixel_std


def preprocess_pixels(
    pixels: np.ndarray | torch.Tensor,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    if isinstance(pixels, np.ndarray):
        tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    else:
        tensor = pixels
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(
            tensor,
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        )
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    pixels: np.ndarray,
    indices: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    embed_dim: int,
    frame_batch_size: int,
) -> np.ndarray:
    if frame_batch_size <= 0:
        raise ValueError(f"Expected positive frame_batch_size, got {frame_batch_size}.")
    latents: list[np.ndarray] = []
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    for start in range(0, indices.shape[0], frame_batch_size):
        batch_idx = indices[start : start + frame_batch_size]
        batch = preprocess_pixels(
            pixels[batch_idx],
            img_size=img_size,
            pixel_mean=pixel_mean,
            pixel_std=pixel_std,
        )
        output = model.encoder(batch, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb[:, :embed_dim].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def load_rendered_obstacle_dataset(data_path: Path, label_mode: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Obstacle dataset not found: {data_path}")
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError(f"Unexpected obstacle dataset format in {data_path}.")

    dataset = payload["dataset"]
    if "pixels" not in dataset:
        raise KeyError("Obstacle dataset is missing required key dataset['pixels']; rerun obs_data_collect.py.")
    if "task_target" not in dataset:
        raise KeyError("Obstacle dataset is missing required key dataset['task_target'].")
    if "label" not in dataset and label_mode != "all":
        raise KeyError("Obstacle dataset is missing required key dataset['label']; use --label all to skip filtering.")

    pixels = np.asarray(dataset["pixels"], dtype=np.uint8)
    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected pixels with shape (N, H, W, 3), got {pixels.shape}.")

    states = np.asarray(dataset["task_target"], dtype=np.float64)
    if states.ndim != 2:
        raise ValueError(f"Expected task_target with shape (N, d), got {states.shape}.")
    if states.shape[0] != pixels.shape[0]:
        raise ValueError("task_target count does not match pixel count.")
    if states.shape[0] < 4:
        raise ValueError("Need at least four target samples to make non-empty PCA/norm/cal splits.")

    labels = np.full((states.shape[0],), -1, dtype=np.int64)
    if "label" in dataset:
        labels = np.asarray(dataset["label"], dtype=np.int64)
        if labels.shape != (states.shape[0],):
            raise ValueError(f"Expected label shape {(states.shape[0],)}, got {labels.shape}.")

    if label_mode == "obstacle":
        keep = labels == 1
    elif label_mode == "non_obstacle":
        keep = labels == 0
    elif label_mode == "all":
        keep = np.ones((states.shape[0],), dtype=bool)
    else:
        raise ValueError(f"Unsupported label mode: {label_mode}")

    selected_idx = np.flatnonzero(keep).astype(np.int64)
    if selected_idx.shape[0] < 4:
        raise ValueError(f"Need at least four samples for label mode {label_mode!r}, got {selected_idx.shape[0]}.")
    return {
        "pixels": pixels,
        "task_target": states,
        "labels": labels,
        "selected_indices": selected_idx,
    }, payload.get("metadata", {})


def split_pca_norm_cal_val(count: int, rng: np.random.Generator) -> dict[str, np.ndarray]:
    if count < 4:
        raise ValueError(f"Need at least four samples, got {count}.")
    shuffled = np.arange(count, dtype=np.int64)
    rng.shuffle(shuffled)

    pca_count = int(np.floor(0.5 * count))
    norm_count = int(np.floor(0.2 * count))
    val_count = max(1, int(np.floor(0.05 * count)))
    cal_count = count - pca_count - norm_count - val_count
    if min(pca_count, norm_count, cal_count, val_count) <= 0:
        raise ValueError(
            f"Split 50/20/25/5 produced an empty subset for {count} samples: "
            f"pca={pca_count}, norm={norm_count}, cal={cal_count}, val={val_count}."
        )

    return {
        "pca": shuffled[:pca_count],
        "norm": shuffled[pca_count : pca_count + norm_count],
        "calibration": shuffled[pca_count + norm_count : pca_count + norm_count + cal_count],
        "validation": shuffled[pca_count + norm_count + cal_count :],
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, int]:
    if not (0.0 < float(alpha) < 1.0):
        raise ValueError(f"Expected alpha in (0, 1), got {alpha}.")
    score_arr = np.asarray(scores, dtype=np.float64).reshape(-1)
    if score_arr.size == 0:
        raise ValueError("Calibration scores cannot be empty.")
    if not np.all(np.isfinite(score_arr)):
        raise ValueError("Calibration scores must be finite.")

    augmented = np.concatenate((np.sort(score_arr), np.array([np.inf], dtype=np.float64)))
    rank = int(np.ceil((score_arr.size + 1) * (1.0 - float(alpha))))
    rank = min(max(rank, 1), augmented.size)
    return float(augmented[rank - 1]), rank


def build_dimension_wise_conformal_ellipsoid(
    states: np.ndarray,
    *,
    alpha: float,
    epsilon: float,
    rng: np.random.Generator,
) -> dict[str, Any]:
    if float(epsilon) <= 0.0:
        raise ValueError(f"Expected positive epsilon, got {epsilon}.")
    x = np.asarray(states, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected states with shape (N, d), got {x.shape}.")
    n, d = x.shape
    if d < 1:
        raise ValueError("State dimensionality must be positive.")

    splits = split_pca_norm_cal_val(n, rng)
    x_pca = x[splits["pca"]]
    x_norm = x[splits["norm"]]
    x_cal = x[splits["calibration"]]

    mean = np.mean(x_pca, axis=0)
    centered_pca = x_pca - mean
    covariance = np.cov(centered_pca, rowvar=False, bias=False)
    covariance = np.atleast_2d(covariance).astype(np.float64)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]

    norm_projection = (x_norm - mean) @ eigenvectors
    max_abs_projection = np.max(np.abs(norm_projection), axis=0)
    gamma = 1.0 / (max_abs_projection + float(epsilon))

    cal_projection = (x_cal - mean) @ eigenvectors
    cal_scores = np.sqrt(np.sum((cal_projection * gamma.reshape(1, -1)) ** 2, axis=1))
    threshold, quantile_rank = conformal_quantile(cal_scores, alpha)
    if not np.isfinite(threshold):
        raise ValueError(
            "Conformal threshold is infinite. Increase calibration sample count or choose a larger alpha."
        )

    axis_bounds = threshold / gamma
    conformal_eigenvalues = axis_bounds**2
    conformal_covariance = (eigenvectors * conformal_eigenvalues.reshape(1, -1)) @ eigenvectors.T
    conformal_precision = np.linalg.pinv(conformal_covariance)
    mahalanobis_squared = np.einsum("ni,ij,nj->n", x - mean, conformal_precision, x - mean)
    contained = mahalanobis_squared <= 1.0

    return {
        "mean": mean.astype(np.float64),
        "base_covariance": covariance.astype(np.float64),
        "eigenvectors": eigenvectors.astype(np.float64),
        "eigenvalues": eigenvalues.astype(np.float64),
        "max_abs_projection": max_abs_projection.astype(np.float64),
        "gamma": gamma.astype(np.float64),
        "alpha": float(alpha),
        "epsilon": float(epsilon),
        "conformal_threshold": float(threshold),
        "conformal_score_type": "normalized_l2_radius",
        "ellipsoid_limit": 1.0,
        "conformal_quantile_rank": int(quantile_rank),
        "calibration_scores": cal_scores.astype(np.float64),
        "axis_bounds": axis_bounds.astype(np.float64),
        "conformal_covariance": conformal_covariance.astype(np.float64),
        "conformal_precision": conformal_precision.astype(np.float64),
        "containment_mahalanobis_squared": mahalanobis_squared.astype(np.float64),
        "contained": contained.astype(bool),
        "splits": splits,
        "split_fractions": {"pca": 0.5, "norm": 0.2, "calibration": 0.25, "validation": 0.05},
    }


def save_latent_pc_diagnostic(path: Path, latents: np.ndarray, ellipsoid: dict[str, Any]) -> None:
    if latents.shape[1] < 2:
        return

    contained = np.asarray(ellipsoid["contained"], dtype=bool)
    mean = np.asarray(ellipsoid["mean"], dtype=np.float64)
    cov = np.asarray(ellipsoid["conformal_covariance"], dtype=np.float64)
    sub_cov = cov[:2, :2]
    sub_vals, sub_vecs = np.linalg.eigh(sub_cov)
    sub_vals = np.maximum(sub_vals, 0.0)
    order = np.argsort(sub_vals)[::-1]
    sub_vals = sub_vals[order]
    sub_vecs = sub_vecs[:, order]
    angle = np.linspace(0.0, 2.0 * np.pi, 361)
    circle = np.stack((np.cos(angle), np.sin(angle)), axis=0)
    boundary = mean[:2, None] + sub_vecs @ (np.sqrt(sub_vals).reshape(2, 1) * circle)

    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=180)
    if np.any(contained):
        ax.scatter(
            latents[contained, 0],
            latents[contained, 1],
            s=14.0,
            c="#0072b2",
            alpha=0.55,
            edgecolors="none",
            label="inside ellipsoid",
        )
    if np.any(~contained):
        ax.scatter(
            latents[~contained, 0],
            latents[~contained, 1],
            s=18.0,
            c="#d55e00",
            alpha=0.8,
            edgecolors="none",
            label="outside ellipsoid",
        )
    ax.plot(boundary[0], boundary[1], color="#4d4d4d", linewidth=1.5, label="latent dim 0-1 projection")
    ax.scatter(mean[0], mean[1], marker="x", s=70.0, c="#000000", linewidths=1.5, label="PCA mean")
    ax.set_xlabel("latent dim 0")
    ax.set_ylabel("latent dim 1")
    ax.set_title("Latent conformal ellipsoid diagnostic")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def save_latent_pair_grid_diagnostic(
    path: Path,
    latents: np.ndarray,
    ellipsoid: dict[str, Any],
    *,
    rng: np.random.Generator,
    rows: int = 4,
    cols: int = 5,
) -> list[tuple[int, int]]:
    dim = int(latents.shape[1])
    pair_count = int(rows * cols)
    if dim < 2:
        return []

    all_pairs = np.array([(i, j) for i in range(dim) for j in range(i + 1, dim)], dtype=np.int64)
    if all_pairs.shape[0] <= pair_count:
        chosen_pairs = all_pairs
    else:
        chosen_pairs = all_pairs[rng.choice(all_pairs.shape[0], size=pair_count, replace=False)]

    contained = np.asarray(ellipsoid["contained"], dtype=bool)
    mean = np.asarray(ellipsoid["mean"], dtype=np.float64)
    cov = np.asarray(ellipsoid["conformal_covariance"], dtype=np.float64)
    fig, axes = plt.subplots(rows, cols, figsize=(15.0, 12.0), dpi=180)
    flat_axes = np.asarray(axes).reshape(-1)
    for ax, pair in zip(flat_axes, chosen_pairs, strict=False):
        i, j = int(pair[0]), int(pair[1])
        if np.any(contained):
            ax.scatter(
                latents[contained, i],
                latents[contained, j],
                s=3.5,
                c="#0072b2",
                alpha=0.35,
                edgecolors="none",
            )
        if np.any(~contained):
            ax.scatter(
                latents[~contained, i],
                latents[~contained, j],
                s=5.0,
                c="#d55e00",
                alpha=0.65,
                edgecolors="none",
            )

        sub_cov = cov[np.ix_([i, j], [i, j])]
        sub_vals, sub_vecs = np.linalg.eigh(sub_cov)
        sub_vals = np.maximum(sub_vals, 0.0)
        order = np.argsort(sub_vals)[::-1]
        sub_vals = sub_vals[order]
        sub_vecs = sub_vecs[:, order]
        angle = np.linspace(0.0, 2.0 * np.pi, 241)
        circle = np.stack((np.cos(angle), np.sin(angle)), axis=0)
        boundary = mean[[i, j], None] + sub_vecs @ (np.sqrt(sub_vals).reshape(2, 1) * circle)
        ax.plot(boundary[0], boundary[1], color="#4d4d4d", linewidth=0.9)
        ax.scatter(mean[i], mean[j], marker="x", s=18.0, c="#000000", linewidths=0.8)
        ax.set_title(f"z{i} vs z{j}", fontsize=9)
        ax.tick_params(axis="both", labelsize=7, length=2)
        ax.grid(alpha=0.16, linewidth=0.5)

    for ax in flat_axes[len(chosen_pairs) :]:
        ax.axis("off")

    fig.suptitle("Random latent-dimension conformal ellipsoid projections", fontsize=13)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.975))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return [(int(pair[0]), int(pair[1])) for pair in chosen_pairs]


def save_task_projection_diagnostic(path: Path, task_states: np.ndarray, contained: np.ndarray) -> None:
    if task_states.shape[1] < 2:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=180)
    inside = np.asarray(contained, dtype=bool)
    if np.any(inside):
        ax.scatter(
            task_states[inside, 0],
            task_states[inside, 1],
            s=14.0,
            c="#0072b2",
            alpha=0.55,
            edgecolors="none",
            label="latent-inside",
        )
    if np.any(~inside):
        ax.scatter(
            task_states[~inside, 0],
            task_states[~inside, 1],
            s=18.0,
            c="#d55e00",
            alpha=0.8,
            edgecolors="none",
            label="latent-outside",
        )
    ax.set_xlabel("reach")
    ax.set_ylabel("height")
    ax.set_title("Task-space view of latent ellipsoid containment")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    data_path = args.data_path.expanduser().resolve()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = load_config(model_dir)
    embed_dim = int(config_dict.get("embed_dim", 12))
    img_size = int(config_dict.get("img_size", 224))
    device = require_device(str(args.device))

    rng = np.random.default_rng(args.seed)
    data, metadata = load_rendered_obstacle_dataset(data_path, str(args.label))
    source_indices = np.asarray(data["selected_indices"], dtype=np.int64)
    task_states = np.asarray(data["task_target"], dtype=np.float64)[source_indices]
    pixels = np.asarray(data["pixels"], dtype=np.uint8)

    print(f"Encoding {source_indices.shape[0]} rendered {args.label} samples on {device}.")
    world_model = load_world_model(checkpoint_path, device)
    latents = encode_pixels(
        world_model,
        pixels,
        source_indices,
        device=device,
        img_size=img_size,
        embed_dim=embed_dim,
        frame_batch_size=int(args.frame_batch_size),
    ).astype(np.float64)
    del world_model

    ellipsoid = build_dimension_wise_conformal_ellipsoid(
        latents,
        alpha=float(args.alpha),
        epsilon=float(args.epsilon),
        rng=rng,
    )

    local_splits = ellipsoid["splits"]
    source_splits = {name: source_indices[idx].astype(np.int64) for name, idx in local_splits.items()}
    artifact = {
        **ellipsoid,
        "source_indices": source_indices.astype(np.int64),
        "source_splits": source_splits,
        "latents": latents.astype(np.float32),
        "task_target": task_states.astype(np.float32),
        "label_mode": str(args.label),
        "feature_space": "world_model_latent",
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "embed_dim": int(embed_dim),
        "img_size": int(img_size),
        "data_path": str(data_path),
        "source_metadata": jsonable(metadata),
        "ellipsoid_definition": "(x - mean)^T conformal_precision (x - mean) <= 1",
    }

    artifact_path = out_dir / str(args.artifact_name)
    torch.save(artifact, artifact_path)

    diagnostic_path = out_dir / "ellipsoid_latent_dim_0_1.png"
    grid_diagnostic_path = out_dir / "ellipsoid_latent_pair_grid.png"
    task_diagnostic_path = out_dir / "ellipsoid_task_reach_height.png"
    save_latent_pc_diagnostic(diagnostic_path, latents, ellipsoid)
    latent_pair_grid = save_latent_pair_grid_diagnostic(grid_diagnostic_path, latents, ellipsoid, rng=rng)
    save_task_projection_diagnostic(task_diagnostic_path, task_states, np.asarray(ellipsoid["contained"], dtype=bool))

    contained = np.asarray(ellipsoid["contained"], dtype=bool)
    cal_idx = np.asarray(local_splits["calibration"], dtype=np.int64)
    val_idx = np.asarray(local_splits["validation"], dtype=np.int64)
    summary = {
        "data_path": str(data_path),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "artifact_path": str(artifact_path),
        "diagnostic_path": str(diagnostic_path),
        "grid_diagnostic_path": str(grid_diagnostic_path),
        "task_diagnostic_path": str(task_diagnostic_path),
        "label_mode": str(args.label),
        "feature_space": "world_model_latent",
        "total_selected": int(latents.shape[0]),
        "dimension": int(latents.shape[1]),
        "embed_dim": int(embed_dim),
        "img_size": int(img_size),
        "alpha": float(args.alpha),
        "target_coverage": float(1.0 - float(args.alpha)),
        "epsilon": float(args.epsilon),
        "seed": int(args.seed),
        "frame_batch_size": int(args.frame_batch_size),
        "split_counts": {name: int(idx.shape[0]) for name, idx in local_splits.items()},
        "conformal_threshold": float(ellipsoid["conformal_threshold"]),
        "conformal_score_type": str(ellipsoid["conformal_score_type"]),
        "ellipsoid_limit": float(ellipsoid["ellipsoid_limit"]),
        "conformal_quantile_rank": int(ellipsoid["conformal_quantile_rank"]),
        "axis_bounds": jsonable(ellipsoid["axis_bounds"]),
        "latent_pair_grid": jsonable(latent_pair_grid),
        "mean": jsonable(ellipsoid["mean"]),
        "empirical_containment_all_selected": float(np.mean(contained)),
        "empirical_containment_calibration": float(np.mean(contained[cal_idx])),
        "empirical_containment_validation": float(np.mean(contained[val_idx])),
        "empirical_miscoverage_validation": float(1.0 - np.mean(contained[val_idx])),
        "source_metadata": jsonable(metadata),
    }
    save_json(out_dir / "summary.json", summary)

    print(f"Saved ellipsoid artifact: {artifact_path}")
    print(f"Saved summary:            {out_dir / 'summary.json'}")
    print(f"Saved diagnostic:         {diagnostic_path}")
    print(f"Saved grid diagnostic:    {grid_diagnostic_path}")
    print(f"Saved task diagnostic:    {task_diagnostic_path}")
    print(f"Selected samples:         {latents.shape[0]}")
    print(f"Latent dimension:         {latents.shape[1]}")
    print(f"Split counts:             {summary['split_counts']}")
    print(f"Validation coverage:      {summary['empirical_containment_validation']:.6f}")
    print(f"Validation miscoverage:   {summary['empirical_miscoverage_validation']:.6f}")


if __name__ == "__main__":
    main()
