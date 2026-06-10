#!/usr/bin/env python3
"""Train and conformalize a Reacher latent obstacle classifier from collected obstacle data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
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
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm

from rope.plan.obstacle_net import (
    ObstacleMLP,
    compute_conformal_score_threshold,
    encode_pixels,
    evaluate_signed_model,
    hinge_loss,
    jsonable,
    latest_object_checkpoint,
    load_config,
    load_world_model,
    require_device,
    save_json,
    score_threshold_metrics,
    to_signed_labels,
)

DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_embd_5"
DEFAULT_DATA_PATH = "reacher/plan/obstacle_data_joint_box/obstacle_classifier_data.pt"
DEFAULT_OUT_DIR = "reacher/plan/obs_net"


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
    parser.add_argument("--train-frac", type=float, default=0.8, help="Fraction of each class used for classifier training.")
    parser.add_argument(
        "--val-frac",
        "--validation-frac",
        dest="val_frac",
        type=float,
        default=0.1,
        help="Fraction of each class used for validation.",
    )
    parser.add_argument("--cal-frac", type=float, default=0.1, help="Fraction of each class used for conformal calibration.")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--force-retrain", action="store_true")
    return parser.parse_args()


def log_progress(message: str) -> None:
    print(f"[reacher_obstacle_net] {message}", flush=True)


def install_numpy_pickle_compat() -> None:
    if "numpy._core" in sys.modules:
        return
    try:
        import numpy.core as numpy_core
        import numpy.core.multiarray as numpy_multiarray
        import numpy.core.numeric as numpy_numeric
    except Exception:
        return
    sys.modules.setdefault("numpy._core", numpy_core)
    sys.modules.setdefault("numpy._core.multiarray", numpy_multiarray)
    sys.modules.setdefault("numpy._core.numeric", numpy_numeric)


def load_obstacle_dataset(data_path: Path) -> dict[str, Any]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Obstacle dataset not found: {data_path}")
    install_numpy_pickle_compat()
    payload = torch.load(data_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError(f"Unexpected obstacle dataset format in {data_path}.")
    dataset = payload["dataset"]
    required = ("pixels", "label")
    missing = [key for key in required if key not in dataset]
    if missing:
        raise KeyError(f"Obstacle dataset is missing required keys: {missing}")
    return payload


def validate_split_fractions(train_frac: float, val_frac: float, cal_frac: float) -> None:
    fractions = {
        "train-frac": float(train_frac),
        "val-frac": float(val_frac),
        "cal-frac": float(cal_frac),
    }
    for name, value in fractions.items():
        if not 0.0 < value < 1.0:
            raise ValueError(f"--{name} must be strictly between 0 and 1.")
    total = sum(fractions.values())
    if not np.isclose(total, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError(f"--train-frac + --val-frac + --cal-frac must equal 1.0, got {total:.8f}.")


def split_indices(indices: np.ndarray, rng: np.random.Generator, train_fraction: float, val_fraction: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = np.asarray(indices, dtype=np.int64).copy()
    rng.shuffle(shuffled)
    if shuffled.size < 3:
        raise ValueError("Need at least three samples per class to create train/validation/calibration splits.")
    train_count = int(np.floor(float(train_fraction) * float(shuffled.size)))
    val_count = int(np.floor(float(val_fraction) * float(shuffled.size)))
    train_count = max(train_count, 1)
    val_count = max(val_count, 1)
    if train_count + val_count >= shuffled.size:
        overflow = train_count + val_count - shuffled.size + 1
        if train_count >= val_count:
            train_count -= overflow
        else:
            val_count -= overflow
    if train_count <= 0 or val_count <= 0 or shuffled.size - train_count - val_count <= 0:
        raise ValueError("Split fractions produced an empty train, validation, or calibration class split.")
    return (
        shuffled[:train_count],
        shuffled[train_count : train_count + val_count],
        shuffled[train_count + val_count :],
    )


def split_stratified_train_val_cal(
    labels_binary: np.ndarray,
    *,
    train_frac: float,
    val_frac: float,
    cal_frac: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    validate_split_fractions(train_frac, val_frac, cal_frac)
    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []
    calibration_parts: list[np.ndarray] = []
    for label in sorted(np.unique(labels_binary).tolist()):
        class_idx = np.flatnonzero(labels_binary == label).astype(np.int64)
        train_idx, val_idx, calibration_idx = split_indices(class_idx, rng, train_frac, val_frac)
        train_parts.append(train_idx)
        val_parts.append(val_idx)
        calibration_parts.append(calibration_idx)

    train_idx = np.concatenate(train_parts, axis=0).astype(np.int64)
    val_idx = np.concatenate(val_parts, axis=0).astype(np.int64)
    calibration_idx = np.concatenate(calibration_parts, axis=0).astype(np.int64)
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    rng.shuffle(calibration_idx)
    return train_idx, val_idx, calibration_idx


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
    train_frac: float
    val_frac: float
    cal_frac: float
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
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        cal_frac=float(args.cal_frac),
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
    validate_split_fractions(float(args.train_frac), float(args.val_frac), float(args.cal_frac))

    device = require_device(args.device)
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    data_path = args.data_path.expanduser().resolve()
    config_dict = load_config(model_dir)
    embed_dim = int(config_dict.get("embed_dim", 7))
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

    log_progress("Loading collected obstacle dataset.")
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
        raise ValueError("Expected binary labels with 1=obstacle and 0=non-obstacle.")

    train_idx, val_idx, calibration_idx = split_stratified_train_val_cal(
        labels_binary,
        train_frac=float(args.train_frac),
        val_frac=float(args.val_frac),
        cal_frac=float(args.cal_frac),
        rng=rng,
    )
    y_train = to_signed_labels(labels_binary[train_idx])
    y_val = to_signed_labels(labels_binary[val_idx])
    y_cal = to_signed_labels(labels_binary[calibration_idx])
    cal_obstacle_mask = labels_binary[calibration_idx] == 1

    log_progress("Loading Reacher world model and encoding train, validation, and calibration frames.")
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

    log_progress(f"Training classifier on {len(train_ds)} train / {len(val_idx)} validation / {len(calibration_idx)} calibration samples.")
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
    nominal_position = train_latents[labels_binary[train_idx] == 1].mean(axis=0).astype(np.float32)

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
                "obstacle": "negative",
                "non_obstacle": "positive",
                "binary_source_labels": {"obstacle": 1, "non_obstacle": 0},
            },
            "base_decision_threshold": 0.0,
            "conformal_safe_score_threshold": float(safe_score_threshold),
            "conformal_delta": float(args.delta),
            "conformal_nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
            "conformal_score_quantile": float(conformal["score_quantile"]),
            "nominal_position": nominal_position,
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
            "train_latents": train_latents.astype(np.float32),
            "val_latents": val_latents.astype(np.float32),
            "calibration_latents": cal_latents.astype(np.float32),
            "train_labels_signed": y_train.astype(np.float32),
            "val_labels_signed": y_val.astype(np.float32),
            "calibration_labels_signed": y_cal.astype(np.float32),
            "train_labels_binary": labels_binary[train_idx].astype(np.int64),
            "val_labels_binary": labels_binary[val_idx].astype(np.int64),
            "calibration_labels_binary": labels_binary[calibration_idx].astype(np.int64),
            "eval": {
                "train": train_eval,
                "val": val_eval,
                "cal": cal_eval,
            },
            "conformal": {
                "base_decision_threshold": 0.0,
                "safe_score_threshold": float(safe_score_threshold),
                "delta": float(args.delta),
                "nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
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
            "nominal_position": nominal_position,
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
        "split_fractions": {
            "train": float(args.train_frac),
            "validation": float(args.val_frac),
            "calibration": float(args.cal_frac),
        },
        "encode_seconds": float(encode_seconds),
        "train_seconds": float(train_seconds),
        "dataset_sizes": {
            "total": int(labels_binary.shape[0]),
            "train": int(train_idx.shape[0]),
            "val": int(val_idx.shape[0]),
            "cal": int(calibration_idx.shape[0]),
            "train_obstacle": int(np.sum(labels_binary[train_idx] == 1)),
            "train_non_obstacle": int(np.sum(labels_binary[train_idx] == 0)),
            "val_obstacle": int(np.sum(labels_binary[val_idx] == 1)),
            "val_non_obstacle": int(np.sum(labels_binary[val_idx] == 0)),
            "cal_obstacle": int(np.sum(labels_binary[calibration_idx] == 1)),
            "cal_non_obstacle": int(np.sum(labels_binary[calibration_idx] == 0)),
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
            "nonconformity_definition": "max(0, NN(x)) on obstacle calibration samples",
            "score_quantile": float(conformal["score_quantile"]),
            "num_obstacle_calibration": int(conformal["num_obstacle_calibration"]),
            "metrics": {
                "train": train_cp,
                "val": val_cp,
                "cal": cal_cp,
            },
        },
        "score_sign_convention": {
            "obstacle": "negative",
            "non_obstacle": "positive",
            "binary_source_labels": {"obstacle": 1, "non_obstacle": 0},
        },
    }
    save_json(paths.summary, summary)

    log_progress("Obstacle net complete.")
    print(f"Cache dir:  {paths.cache_dir}")
    print(f"Model path: {paths.model}")
    print(f"Train acc:  {train_eval['accuracy']:.4f}")
    print(f"Val acc:    {val_eval['accuracy']:.4f}")
    print(f"Cal acc:    {cal_eval['accuracy']:.4f}")
    print("Conformal nonconformity: max(0, NN(x)) on obstacle calibration samples")
    print(f"Applied safe score threshold: {safe_score_threshold:.6f}")


if __name__ == "__main__":
    main()
