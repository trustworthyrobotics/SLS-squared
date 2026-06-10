#!/usr/bin/env python3
"""Compute per-dimension latent, Markov-state, and action domain statistics for Rope planning."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from rope.train.mlpdyn_train import build_markov_state, preprocess_pixels, required_markov_history


DEFAULT_DATASET_PATH = "rope/data/test_data_noshadow/rope_random_cubic_spline.h5"
DEFAULT_MODEL_DIR = "rope/models/mlpdyn_noshadow"
DEFAULT_OUT_DIR = "rope/eval/latent_domain_stats_large"


@dataclass
class RunningStats:
    dim: int

    def __post_init__(self) -> None:
        self.count = 0
        self.min = np.full(self.dim, np.inf, dtype=np.float64)
        self.max = np.full(self.dim, -np.inf, dtype=np.float64)
        self.mean = np.zeros(self.dim, dtype=np.float64)
        self.m2 = np.zeros(self.dim, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.dim:
            raise ValueError(f"Expected values with shape [N, {self.dim}], got {values.shape}.")
        if values.shape[0] == 0:
            return

        self.min = np.minimum(self.min, values.min(axis=0))
        self.max = np.maximum(self.max, values.max(axis=0))

        batch_count = values.shape[0]
        batch_mean = values.mean(axis=0)
        centered = values - batch_mean[None, :]
        batch_m2 = np.sum(centered * centered, axis=0)

        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        total = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean = self.mean + delta * (batch_count / total)
        self.m2 = self.m2 + batch_m2 + delta * delta * (self.count * batch_count / total)
        self.count = total

    def as_arrays(self) -> dict[str, np.ndarray]:
        variance = self.m2 / max(self.count - 1, 1)
        std = np.sqrt(np.maximum(variance, 0.0))
        return {
            "count": np.full(self.dim, self.count, dtype=np.int64),
            "min": self.min,
            "max": self.max,
            "mean": self.mean,
            "variance": variance,
            "std": std,
            "range": self.max - self.min,
            "mean_minus_3std": self.mean - 3.0 * std,
            "mean_plus_3std": self.mean + 3.0 * std,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--markov-deriv", type=int, default=None)
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--max-episodes", type=int, default=None)
    return parser.parse_args()


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


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def apply_config_defaults(args: argparse.Namespace, config: dict[str, object]) -> None:
    if args.markov_deriv is None:
        args.markov_deriv = int(config.get("markov_deriv", 1))
    if args.img_size is None:
        args.img_size = int(config.get("img_size", 224))


@torch.no_grad()
def encode_pixels(
    model: torch.nn.Module,
    pixels_np: np.ndarray,
    *,
    device: torch.device,
    img_size: int,
    frame_batch_size: int,
) -> torch.Tensor:
    pixels = torch.from_numpy(pixels_np.copy()).permute(0, 3, 1, 2).contiguous()
    pixels = preprocess_pixels(pixels.unsqueeze(0), img_size)[0]

    latents = []
    for start in range(0, pixels.shape[0], frame_batch_size):
        chunk = pixels[start : start + frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        latents.append(model.projector(output.last_hidden_state[:, 0]).detach().cpu())
    return torch.cat(latents, dim=0)


def build_all_markov_states(latents: torch.Tensor, markov_deriv: int) -> torch.Tensor:
    history_len = required_markov_history(markov_deriv)
    states = []
    for t in range(latents.shape[0]):
        start = max(0, t - history_len + 1)
        history = latents[start : t + 1]
        if history.shape[0] < history_len:
            history = torch.cat((history[:1].repeat(history_len - history.shape[0], 1), history), dim=0)
        states.append(build_markov_state(history.unsqueeze(0), markov_deriv)[0])
    return torch.stack(states, dim=0)


def write_stats_csv(path: Path, arrays: dict[str, np.ndarray], *, prefix: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dimension",
        "name",
        "count",
        "min",
        "max",
        "mean",
        "variance",
        "std",
        "range",
        "mean_minus_3std",
        "mean_plus_3std",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dim in range(arrays["mean"].shape[0]):
            writer.writerow(
                {
                    "dimension": dim,
                    "name": f"{prefix}_{dim}",
                    **{key: arrays[key][dim] for key in fieldnames[2:]},
                }
            )


def write_stats_json(path: Path, arrays: dict[str, np.ndarray], *, metadata: dict[str, object]) -> None:
    payload = {
        "metadata": metadata,
        "stats": {key: value.tolist() for key, value in arrays.items()},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2)


def plot_ranges(path: Path, arrays: dict[str, np.ndarray], *, title: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = np.arange(arrays["mean"].shape[0])
    fig, ax = plt.subplots(figsize=(max(10.0, 0.35 * len(dims)), 5.5))
    ax.fill_between(dims, arrays["min"], arrays["max"], color="tab:blue", alpha=0.18, label="observed min/max")
    ax.plot(dims, arrays["mean"], color="tab:blue", linewidth=1.6, label="mean")
    ax.plot(dims, arrays["mean_minus_3std"], color="tab:orange", linewidth=1.0, linestyle="--", label="mean +/- 3 std")
    ax.plot(dims, arrays["mean_plus_3std"], color="tab:orange", linewidth=1.0, linestyle="--")
    ax.set_title(title)
    ax.set_xlabel("dimension")
    ax.set_ylabel("value")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def finite_action_stats(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_actions = actions[~np.isnan(actions).any(axis=1)]
    if finite_actions.shape[0] == 0:
        raise ValueError("No finite action rows found in dataset.")
    action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
    action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
    action_std = np.maximum(action_std, 1e-6)
    return action_mean, action_std


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    dataset_path = args.dataset_path.expanduser().resolve()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "config.json").open() as f:
        config = json.load(f)
    apply_config_defaults(args, config)

    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    device = require_device(args.device)
    model = torch.load(checkpoint_path, map_location=device, weights_only=False).to(device).eval()
    model.requires_grad_(False)

    embed_dim = int(config.get("embed_dim", 0))
    if embed_dim <= 0:
        embed_dim = int(config.get("markov_state_dim", 0)) // (int(args.markov_deriv) + 1)
    if embed_dim <= 0:
        raise ValueError("Could not infer embed_dim from config.json.")
    markov_dim = embed_dim * (int(args.markov_deriv) + 1)

    latent_stats = RunningStats(embed_dim)
    markov_stats = RunningStats(markov_dim)

    with h5py.File(dataset_path, "r") as h5:
        ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
        ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
        action_dim = int(h5["action"].shape[-1])
        action_mean, action_std = finite_action_stats(np.asarray(h5["action"][:], dtype=np.float32))
        raw_action_stats = RunningStats(action_dim)
        norm_action_stats = RunningStats(action_dim)
        episode_indices = np.arange(ep_len.shape[0], dtype=np.int64)
        if args.max_episodes is not None:
            episode_indices = episode_indices[: int(args.max_episodes)]

        for ep_idx in tqdm(episode_indices, desc="Encoding rope episodes"):
            offset = int(ep_offset[ep_idx])
            length = int(ep_len[ep_idx])
            if length <= 0:
                continue
            rows = np.arange(offset, offset + length, dtype=np.int64)
            pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
            actions_np = np.asarray(h5["action"][rows], dtype=np.float32)
            finite_action_rows = actions_np[~np.isnan(actions_np).any(axis=1)]
            if finite_action_rows.shape[0] > 0:
                raw_action_stats.update(finite_action_rows)
            norm_actions = (np.nan_to_num(actions_np, nan=0.0) - action_mean) / action_std
            norm_action_stats.update(norm_actions)
            latents = encode_pixels(
                model,
                pixels_np,
                device=device,
                img_size=int(args.img_size),
                frame_batch_size=int(args.frame_batch_size),
            )
            markov_states = build_all_markov_states(latents, int(args.markov_deriv))
            latent_stats.update(latents.numpy())
            markov_stats.update(markov_states.numpy())

    metadata = {
        "dataset_path": str(dataset_path),
        "model_dir": str(model_dir),
        "checkpoint": str(checkpoint_path),
        "markov_deriv": int(args.markov_deriv),
        "img_size": int(args.img_size),
        "embed_dim": embed_dim,
        "markov_state_dim": markov_dim,
        "action_dim": action_dim,
        "action_mean_for_normalization": action_mean.reshape(-1).tolist(),
        "action_std_for_normalization": action_std.reshape(-1).tolist(),
        "num_frames": int(latent_stats.count),
        "num_action_rows": int(norm_action_stats.count),
        "num_finite_raw_action_rows": int(raw_action_stats.count),
        "num_episodes": int(len(episode_indices)),
    }
    latent_arrays = latent_stats.as_arrays()
    markov_arrays = markov_stats.as_arrays()
    raw_action_arrays = raw_action_stats.as_arrays()
    norm_action_arrays = norm_action_stats.as_arrays()

    write_stats_csv(out_dir / "latent_embedding_stats.csv", latent_arrays, prefix="z")
    write_stats_csv(out_dir / "markov_state_stats.csv", markov_arrays, prefix="x")
    write_stats_csv(out_dir / "raw_action_stats.csv", raw_action_arrays, prefix="raw_u")
    write_stats_csv(out_dir / "normalized_action_stats.csv", norm_action_arrays, prefix="norm_u")
    write_stats_json(out_dir / "latent_embedding_stats.json", latent_arrays, metadata=metadata)
    write_stats_json(out_dir / "markov_state_stats.json", markov_arrays, metadata=metadata)
    write_stats_json(out_dir / "raw_action_stats.json", raw_action_arrays, metadata=metadata)
    write_stats_json(out_dir / "normalized_action_stats.json", norm_action_arrays, metadata=metadata)
    np.savez(
        out_dir / "latent_domain_stats.npz",
        **{f"latent_{key}": value for key, value in latent_arrays.items()},
        **{f"markov_{key}": value for key, value in markov_arrays.items()},
        **{f"raw_action_{key}": value for key, value in raw_action_arrays.items()},
        **{f"normalized_action_{key}": value for key, value in norm_action_arrays.items()},
    )
    plot_ranges(out_dir / "latent_embedding_ranges.png", latent_arrays, title="Rope embedding latent domain")
    plot_ranges(out_dir / "markov_state_ranges.png", markov_arrays, title="Rope Markov planning-state domain")
    plot_ranges(out_dir / "raw_action_ranges.png", raw_action_arrays, title="Rope raw action domain")
    plot_ranges(out_dir / "normalized_action_ranges.png", norm_action_arrays, title="Rope normalized action domain")

    print(f"Processed {latent_stats.count} frames from {len(episode_indices)} episodes.")
    print(f"Processed {norm_action_stats.count} normalized action rows.")
    print(f"Saved latent domain stats to {out_dir}")


if __name__ == "__main__":
    main()
