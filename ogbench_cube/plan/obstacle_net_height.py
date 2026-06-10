#!/usr/bin/env python3
"""Train and conformalize an OGBench cube latent height-constraint classifier from collected height data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

DEFAULT_MODEL_DIR = "ogbench_cube/models/mlpdyn_embd_12_strtn"
DEFAULT_DATA_PATH = "ogbench_cube/plan/height_data/height_classifier_data.pt"
DEFAULT_OUT_DIR = "ogbench_cube/plan/obs_net_height_embd_12_strtn"
DEFAULT_ACTIVATION = nn.Tanh
DEFAULT_SOURCE_TRAIN_FRAC = 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path(DEFAULT_MODEL_DIR))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--data-path", type=Path, default=Path(DEFAULT_DATA_PATH))
    parser.add_argument("--out-dir", type=Path, default=Path(DEFAULT_OUT_DIR))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=6)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--margin", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=0.01)
    parser.add_argument("--source-train-frac", type=float, default=DEFAULT_SOURCE_TRAIN_FRAC)
    parser.add_argument("--validation-frac", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[ogbench_cube_obstacle_net_height] {message}", flush=True)


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
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


def load_obstacle_dataset(data_path: Path) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Height dataset not found: {data_path}")
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError(f"Unexpected height dataset format in {data_path}.")
    dataset = payload["dataset"]
    required = ("pixels", "label")
    missing = [key for key in required if key not in dataset]
    if missing:
        raise KeyError(f"Height dataset is missing required keys: {missing}")
    return payload


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
    progress_desc: str,
) -> np.ndarray:
    latents: list[np.ndarray] = []
    iterator = tqdm(range(0, indices.shape[0], frame_batch_size), desc=progress_desc, unit="batch")
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    for start in iterator:
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


class ObstacleMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, depth: int, dropout: float) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"Expected depth >= 1, got {depth}.")
        layers: list[nn.Module] = []
        in_dim = input_dim
        for _ in range(depth - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(DEFAULT_ACTIVATION())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def split_indices(indices: np.ndarray, rng: np.random.Generator, train_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("--source-train-frac must be between 0 and 1.")
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    if shuffled.size < 2:
        raise ValueError("Need at least two samples per class for source train/calibration splitting.")
    train_count = int(np.floor(train_fraction * float(shuffled.size)))
    train_count = min(max(train_count, 1), shuffled.size - 1)
    return shuffled[:train_count], shuffled[train_count:]


def split_stratified_train_calibration(
    labels: np.ndarray,
    *,
    source_train_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    train_parts: list[np.ndarray] = []
    cal_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels).tolist()):
        class_idx = np.flatnonzero(labels == label)
        train_idx, cal_idx = split_indices(class_idx, rng, source_train_frac)
        train_parts.append(train_idx)
        cal_parts.append(cal_idx)
    source_train_idx = np.concatenate(train_parts, axis=0).astype(np.int64)
    calibration_idx = np.concatenate(cal_parts, axis=0).astype(np.int64)
    rng.shuffle(source_train_idx)
    rng.shuffle(calibration_idx)
    return source_train_idx, calibration_idx


def split_stratified_validation(
    train_idx: np.ndarray,
    labels: np.ndarray,
    *,
    validation_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    if not 0.0 < validation_frac < 1.0:
        raise ValueError("--validation-frac must be between 0 and 1.")
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels[train_idx]).tolist()):
        class_idx = train_idx[labels[train_idx] == label].copy()
        rng.shuffle(class_idx)
        if class_idx.shape[0] < 2:
            raise ValueError(f"Need at least two source-train samples for class {label}.")
        val_count = int(np.ceil(validation_frac * class_idx.shape[0]))
        val_count = min(max(val_count, 1), class_idx.shape[0] - 1)
        val_parts.append(class_idx[:val_count])
        train_parts.append(class_idx[val_count:])
    split_train = np.concatenate(train_parts, axis=0).astype(np.int64)
    split_val = np.concatenate(val_parts, axis=0).astype(np.int64)
    rng.shuffle(split_train)
    rng.shuffle(split_val)
    return split_train, split_val


def to_signed_labels(binary_labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(binary_labels, dtype=np.int64)
    return np.where(labels == 1, -1.0, 1.0).astype(np.float32)


def hinge_loss(scores: torch.Tensor, labels: torch.Tensor, *, margin: float) -> torch.Tensor:
    return torch.clamp(float(margin) - labels * scores, min=0.0).mean()


def compute_signed_metrics(scores: torch.Tensor, labels: torch.Tensor, *, threshold: float = 0.0) -> dict[str, float]:
    preds_obstacle = scores <= float(threshold)
    labels_obstacle = labels < 0.0
    accuracy = float((preds_obstacle == labels_obstacle).float().mean().item())
    pos_mask = labels_obstacle
    neg_mask = ~pos_mask
    recall = float(preds_obstacle[pos_mask].float().mean().item()) if torch.any(pos_mask) else 0.0
    specificity = float((~preds_obstacle[neg_mask]).float().mean().item()) if torch.any(neg_mask) else 0.0
    tp = float(torch.sum(preds_obstacle & pos_mask).item())
    fp = float(torch.sum(preds_obstacle & neg_mask).item())
    precision = tp / max(tp + fp, 1.0)
    return {
        "accuracy": accuracy,
        "recall": recall,
        "specificity": specificity,
        "precision": precision,
    }


def evaluate_signed_model(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    batch_size: int,
    device: torch.device,
    margin: float,
) -> dict[str, Any]:
    model.eval()
    score_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, x.shape[0], batch_size):
            xb = x[start : start + batch_size].to(device)
            score_chunks.append(model(xb).cpu())
    scores = torch.cat(score_chunks, dim=0)
    loss = float(hinge_loss(scores, y, margin=margin).item())
    metrics = compute_signed_metrics(scores, y, threshold=0.0)
    return {
        "loss": loss,
        "scores": scores.numpy().astype(np.float32),
        **metrics,
    }


def conformal_quantile(scores: np.ndarray, delta: float) -> float:
    if not 0.0 < delta < 1.0:
        raise ValueError("--delta must be between 0 and 1.")
    n = int(scores.shape[0])
    augmented = np.concatenate((np.sort(scores.astype(np.float64)), np.array([np.inf], dtype=np.float64)))
    rank = int(np.ceil((n + 1) * (1.0 - delta))) - 1
    rank = int(np.clip(rank, 0, augmented.shape[0] - 1))
    return float(augmented[rank])


def compute_conformal_score_threshold(obstacle_nonconformity_cal: np.ndarray, delta: float) -> dict[str, float]:
    if obstacle_nonconformity_cal.ndim != 1:
        raise ValueError(
            f"Expected 1D obstacle calibration nonconformity scores, got shape {obstacle_nonconformity_cal.shape}."
        )
    if obstacle_nonconformity_cal.size == 0:
        raise ValueError("Need at least one obstacle calibration sample for conformal calibration.")
    threshold = conformal_quantile(obstacle_nonconformity_cal.astype(np.float64), delta=delta)
    return {
        "threshold": float(threshold),
        "score_quantile": float(threshold),
        "num_obstacle_calibration": int(obstacle_nonconformity_cal.size),
    }


def score_threshold_metrics(eval_payload: dict[str, Any], labels: np.ndarray, threshold: float) -> dict[str, float]:
    scores = np.asarray(eval_payload["scores"], dtype=np.float64)
    labels = np.asarray(labels, dtype=np.float32)
    preds_obstacle = scores <= threshold
    pos_mask = labels < 0.0
    neg_mask = ~pos_mask
    above_threshold_coverage = float(np.mean(preds_obstacle[pos_mask])) if np.any(pos_mask) else 0.0
    below_threshold_activation_rate = float(np.mean(preds_obstacle[neg_mask])) if np.any(neg_mask) else 0.0
    return {
        "above_threshold_coverage": above_threshold_coverage,
        "below_threshold_activation_rate": below_threshold_activation_rate,
        "threshold": float(threshold),
    }


@dataclass(frozen=True)
class ObstacleNetConfig:
    model_dir: str
    checkpoint_path: str
    data_path: str
    seed: int
    frame_batch_size: int
    hidden_dim: int
    depth: int
    dropout: float
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    margin: float
    delta: float
    source_train_frac: float
    validation_frac: float
    embed_dim: int
    img_size: int


@dataclass(frozen=True)
class ArtifactPaths:
    cache_dir: Path
    summary: Path
    model: Path
    splits: Path

    @property
    def required_for_cache_hit(self) -> tuple[Path, ...]:
        return (self.summary, self.model, self.splits)


def build_run_config(
    args: argparse.Namespace,
    *,
    model_dir: Path,
    checkpoint_path: Path,
    data_path: Path,
    embed_dim: int,
    img_size: int,
) -> ObstacleNetConfig:
    return ObstacleNetConfig(
        model_dir=str(model_dir),
        checkpoint_path=str(checkpoint_path),
        data_path=str(data_path),
        seed=int(args.seed),
        frame_batch_size=int(args.frame_batch_size),
        hidden_dim=int(args.hidden_dim),
        depth=int(args.depth),
        dropout=float(args.dropout),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
        margin=float(args.margin),
        delta=float(args.delta),
        source_train_frac=float(args.source_train_frac),
        validation_frac=float(args.validation_frac),
        embed_dim=int(embed_dim),
        img_size=int(img_size),
    )


def cache_key_for_config(config: ObstacleNetConfig) -> str:
    payload = json.dumps(asdict(config), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def artifact_paths(out_root: Path, config: ObstacleNetConfig) -> ArtifactPaths:
    cache_dir = out_root / cache_key_for_config(config)
    return ArtifactPaths(
        cache_dir=cache_dir,
        summary=cache_dir / "summary.json",
        model=cache_dir / "model.pt",
        splits=cache_dir / "splits.pt",
    )


def make_feature_tensors(
    train_latents: np.ndarray,
    val_latents: np.ndarray,
    cal_latents: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    train_x = torch.from_numpy(train_latents.astype(np.float32))
    train_mean = train_x.mean(dim=0)
    train_std = train_x.std(dim=0).clamp_min(1e-6)

    def normalize(x_np: np.ndarray) -> torch.Tensor:
        return (torch.from_numpy(x_np.astype(np.float32)) - train_mean) / train_std

    return normalize(train_latents), normalize(val_latents), normalize(cal_latents), train_mean, train_std


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    if args.frame_batch_size <= 0:
        raise ValueError("--frame-batch-size must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    device = require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    data_path = args.data_path.expanduser().resolve()
    config_dict = load_config(model_dir)
    embed_dim = int(config_dict.get("embed_dim", 8))
    img_size = int(config_dict.get("img_size", 224))

    run_config = build_run_config(
        args,
        model_dir=model_dir,
        checkpoint_path=checkpoint_path,
        data_path=data_path,
        embed_dim=embed_dim,
        img_size=img_size,
    )
    out_root = args.out_dir.expanduser().resolve()
    paths = artifact_paths(out_root, run_config)
    out_root.mkdir(parents=True, exist_ok=True)

    if all(path.is_file() for path in paths.required_for_cache_hit) and not args.force_retrain:
        log_progress(f"Using cached artifact at {paths.cache_dir}.")
        print(f"Cache dir:  {paths.cache_dir}")
        print(f"Model path: {paths.model}")
        return

    log_progress("Loading collected height dataset.")
    data_payload = load_obstacle_dataset(data_path)
    metadata = data_payload.get("metadata", {})
    dataset = data_payload["dataset"]
    pixels = np.asarray(dataset["pixels"], dtype=np.uint8)
    labels_binary = np.asarray(dataset["label"], dtype=np.int64)

    if pixels.ndim != 4 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected pixels with shape (N, H, W, 3), got {pixels.shape}.")
    if labels_binary.shape[0] != pixels.shape[0]:
        raise ValueError("Label count does not match pixel count.")
    if not set(np.unique(labels_binary).tolist()).issubset({0, 1}):
        raise ValueError("Expected binary labels with 1=above-threshold and 0=below-threshold.")

    source_train_idx = (
        np.asarray(dataset["train_idx"], dtype=np.int64)
        if "train_idx" in dataset and "calibration_idx" in dataset
        else None
    )
    calibration_idx = (
        np.asarray(dataset["calibration_idx"], dtype=np.int64)
        if "train_idx" in dataset and "calibration_idx" in dataset
        else None
    )
    split_origin = "dataset"
    if source_train_idx is None or calibration_idx is None:
        source_train_idx, calibration_idx = split_stratified_train_calibration(
            labels_binary,
            source_train_frac=float(args.source_train_frac),
            rng=rng,
        )
        split_origin = "generated_in_height_net"

    train_idx, val_idx = split_stratified_validation(
        source_train_idx,
        labels_binary,
        validation_frac=float(args.validation_frac),
        rng=rng,
    )
    y_train = to_signed_labels(labels_binary[train_idx])
    y_val = to_signed_labels(labels_binary[val_idx])
    y_cal = to_signed_labels(labels_binary[calibration_idx])
    cal_obstacle_mask = labels_binary[calibration_idx] == 1

    log_progress("Loading OGBench world model and encoding train, validation, and calibration frames.")
    world_model = load_world_model(checkpoint_path, device)
    encode_start = time.perf_counter()
    train_latents = encode_pixels(
        world_model,
        pixels,
        train_idx,
        device=device,
        img_size=img_size,
        embed_dim=embed_dim,
        frame_batch_size=int(args.frame_batch_size),
        progress_desc="Encoding train",
    )
    val_latents = encode_pixels(
        world_model,
        pixels,
        val_idx,
        device=device,
        img_size=img_size,
        embed_dim=embed_dim,
        frame_batch_size=int(args.frame_batch_size),
        progress_desc="Encoding validation",
    )
    cal_latents = encode_pixels(
        world_model,
        pixels,
        calibration_idx,
        device=device,
        img_size=img_size,
        embed_dim=embed_dim,
        frame_batch_size=int(args.frame_batch_size),
        progress_desc="Encoding calibration",
    )
    encode_seconds = time.perf_counter() - encode_start
    del world_model

    normalized_train, normalized_val, normalized_cal, train_mean, train_std = make_feature_tensors(
        train_latents,
        val_latents,
        cal_latents,
    )
    y_train_tensor = torch.from_numpy(y_train)
    y_val_tensor = torch.from_numpy(y_val)
    y_cal_tensor = torch.from_numpy(y_cal)

    train_ds = TensorDataset(normalized_train, y_train_tensor)
    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(args.batch_size), max(1, len(train_ds))),
        shuffle=True,
        num_workers=int(args.num_workers),
    )

    model = ObstacleMLP(embed_dim, int(args.hidden_dim), int(args.depth), float(args.dropout)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))

    log_progress(
        f"Training classifier on {len(train_ds)} train / {len(val_idx)} validation / {len(calibration_idx)} calibration samples."
    )
    train_start = time.perf_counter()
    for _ in tqdm(range(int(args.epochs)), desc="Training epochs", unit="epoch"):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            scores = model(xb)
            loss = hinge_loss(scores, yb, margin=float(args.margin))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    train_seconds = time.perf_counter() - train_start

    train_eval = evaluate_signed_model(
        model,
        normalized_train,
        y_train_tensor,
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    val_eval = evaluate_signed_model(
        model,
        normalized_val,
        y_val_tensor,
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )
    cal_eval = evaluate_signed_model(
        model,
        normalized_cal,
        y_cal_tensor,
        batch_size=int(args.batch_size),
        device=device,
        margin=float(args.margin),
    )

    cal_scores = np.asarray(cal_eval["scores"], dtype=np.float64)
    cal_obstacle_scores = cal_scores[cal_obstacle_mask]
    cal_obstacle_nonconformity = np.maximum(cal_obstacle_scores, 0.0)
    conformal = compute_conformal_score_threshold(cal_obstacle_nonconformity, float(args.delta))
    safe_score_threshold = float(conformal["threshold"])

    train_cp = score_threshold_metrics(train_eval, y_train, safe_score_threshold)
    val_cp = score_threshold_metrics(val_eval, y_val, safe_score_threshold)
    cal_cp = score_threshold_metrics(cal_eval, y_cal, safe_score_threshold)

    paths.cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": int(embed_dim),
            "hidden_dim": int(args.hidden_dim),
            "depth": int(args.depth),
            "dropout": float(args.dropout),
            "feature_mean": train_mean.numpy().astype(np.float32),
            "feature_std": train_std.numpy().astype(np.float32),
            "score_sign_convention": {
                "above_threshold": "negative",
                "below_threshold": "positive",
                "binary_source_labels": {"above_threshold": 1, "below_threshold": 0},
            },
            "base_decision_threshold": 0.0,
            "conformal_safe_score_threshold": float(safe_score_threshold),
            "conformal_delta": float(args.delta),
            "conformal_nonconformity_definition": "max(0, NN(x)) on above-threshold calibration samples",
            "conformal_score_quantile": float(conformal["score_quantile"]),
            "cache_config": asdict(run_config),
            "source_metadata": jsonable(metadata),
        },
        paths.model,
    )
    torch.save(
        {
            "train_idx": train_idx.astype(np.int64),
            "val_idx": val_idx.astype(np.int64),
            "calibration_idx": calibration_idx.astype(np.int64),
            "source_train_idx": source_train_idx.astype(np.int64),
            "train_latents": train_latents.astype(np.float32),
            "val_latents": val_latents.astype(np.float32),
            "calibration_latents": cal_latents.astype(np.float32),
            "train_labels_signed": y_train.astype(np.float32),
            "val_labels_signed": y_val.astype(np.float32),
            "calibration_labels_signed": y_cal.astype(np.float32),
            "train_labels_binary": labels_binary[train_idx].astype(np.int64),
            "val_labels_binary": labels_binary[val_idx].astype(np.int64),
            "calibration_labels_binary": labels_binary[calibration_idx].astype(np.int64),
            "split_origin": split_origin,
            "eval": {
                "train": train_eval,
                "val": val_eval,
                "cal": cal_eval,
            },
            "conformal": {
                "base_decision_threshold": 0.0,
                "safe_score_threshold": float(safe_score_threshold),
                "delta": float(args.delta),
                "nonconformity_definition": "max(0, NN(x)) on above-threshold calibration samples",
                "score_quantile": float(conformal["score_quantile"]),
                "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
                "metrics": {
                    "train": train_cp,
                    "val": val_cp,
                    "cal": cal_cp,
                },
            },
            "feature_mean": train_mean.numpy().astype(np.float32),
            "feature_std": train_std.numpy().astype(np.float32),
        },
        paths.splits,
    )

    def compact_eval(eval_payload: dict[str, Any]) -> dict[str, float]:
        return {k: float(v) for k, v in eval_payload.items() if k != "scores"}

    summary = {
        "cache_key": cache_key_for_config(run_config),
        "cache_config": asdict(run_config),
        "cache_dir": str(paths.cache_dir),
        "model_path": str(paths.model),
        "splits_path": str(paths.splits),
        "source_metadata": jsonable(metadata),
        "split_origin": split_origin,
        "encode_seconds": float(encode_seconds),
        "train_seconds": float(train_seconds),
        "dataset_sizes": {
            "total": int(labels_binary.shape[0]),
            "source_train": int(source_train_idx.shape[0]),
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "cal": int(calibration_idx.shape[0]),
            "train_above_threshold": int(np.sum(labels_binary[train_idx] == 1)),
            "train_below_threshold": int(np.sum(labels_binary[train_idx] == 0)),
            "val_above_threshold": int(np.sum(labels_binary[val_idx] == 1)),
            "val_below_threshold": int(np.sum(labels_binary[val_idx] == 0)),
            "cal_above_threshold": int(np.sum(labels_binary[calibration_idx] == 1)),
            "cal_below_threshold": int(np.sum(labels_binary[calibration_idx] == 0)),
        },
        "metrics": {
            "train": compact_eval(train_eval),
            "val": compact_eval(val_eval),
            "cal": compact_eval(cal_eval),
        },
        "conformal": {
            "base_decision_threshold": 0.0,
            "safe_score_threshold": float(safe_score_threshold),
            "delta": float(args.delta),
            "nonconformity_definition": "max(0, NN(x)) on above-threshold calibration samples",
            "score_quantile": float(conformal["score_quantile"]),
            "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
            "metrics": {
                "train": train_cp,
                "val": val_cp,
                "cal": cal_cp,
            },
        },
        "score_sign_convention": {
            "above_threshold": "negative",
            "below_threshold": "positive",
            "binary_source_labels": {"above_threshold": 1, "below_threshold": 0},
        },
    }
    save_json(paths.summary, summary)

    log_progress("Height net complete.")
    print(f"Cache dir:  {paths.cache_dir}")
    print(f"Model path: {paths.model}")
    print(f"Train acc:  {train_eval['accuracy']:.4f}")
    print(f"Val acc:    {val_eval['accuracy']:.4f}")
    print(f"Cal acc:    {cal_eval['accuracy']:.4f}")
    print("Conformal nonconformity: max(0, NN(x)) on above-threshold calibration samples")
    print(f"Applied safe score threshold: {safe_score_threshold:.6f}")


if __name__ == "__main__":
    main()
