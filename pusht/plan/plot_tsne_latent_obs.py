#!/usr/bin/env python3
"""Plot PushT insertion obstacle/no-obstacle samples in the learned latent t-SNE space."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/codex_mplconfig")

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


DEFAULT_DATA_PATH = REPO_ROOT / "pusht" / "plan" / "obstacle_data_insert" / "obstacle_classifier_data.pt"
DEFAULT_MODEL_DIR = REPO_ROOT / "pusht" / "models" / "mlpdyn_embd_48"
DEFAULT_OUT_PATH = REPO_ROOT / "pusht" / "plan" / "tsne_latent_obs.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--frame-batch-size", type=int, default=64)
    parser.add_argument("--perplexity", type=float, default=80.0)
    parser.add_argument("--learning-rate", type=str, default="auto")
    parser.add_argument("--max-iter", type=int, default=1500)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--font-size", type=float, default=13.0)
    parser.add_argument("--legend-font-size", type=float, default=11.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"])
    return parser.parse_args()


def install_numpy_core_pickle_aliases() -> None:
    try:
        import numpy.core

        sys.modules.setdefault("numpy._core", numpy.core)
        sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    except Exception:
        return


def torch_load_local(path: Path) -> Any:
    install_numpy_core_pickle_aliases()
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No object checkpoint matching '*_epoch_N_object.ckpt' found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_model_config(model_dir: Path) -> dict[str, Any]:
    config_path = model_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing model config: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return torch.device(device_arg)


def load_world_model(checkpoint_path: Path, device: torch.device) -> torch.nn.Module:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")
    install_numpy_core_pickle_aliases()
    model = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model


def load_obstacle_pixels(data_path: Path, seed: int, max_samples: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not data_path.is_file():
        raise FileNotFoundError(f"Missing obstacle dataset: {data_path}")
    payload = torch_load_local(data_path)
    if not isinstance(payload, dict) or "dataset" not in payload:
        raise ValueError(f"Unexpected obstacle dataset format in {data_path}")
    dataset = payload["dataset"]
    for key in ("pixels", "label"):
        if key not in dataset:
            raise KeyError(f"Obstacle dataset is missing dataset[{key!r}]")

    pixels = np.asarray(dataset["pixels"], dtype=np.uint8)
    labels = np.asarray(dataset["label"], dtype=np.int64)
    sample_indices = np.arange(labels.shape[0], dtype=np.int64)
    if sample_indices.size == 0:
        raise ValueError(f"No samples found in {data_path}")

    if max_samples is not None and max_samples > 0 and sample_indices.size > max_samples:
        rng = np.random.default_rng(seed)
        selected: list[np.ndarray] = []
        per_class = max(1, max_samples // 2)
        for label in sorted(np.unique(labels).tolist()):
            label_indices = sample_indices[labels == label]
            count = min(per_class, label_indices.shape[0])
            selected.append(rng.choice(label_indices, size=count, replace=False))
        sample_indices = np.sort(np.concatenate(selected, axis=0))
    return pixels, sample_indices.astype(np.int64), labels[sample_indices].astype(np.int64)


def imagenet_pixel_stats(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device).view(1, 3, 1, 1)
    return mean, std


def preprocess_pixels(
    pixels: np.ndarray,
    *,
    img_size: int,
    pixel_mean: torch.Tensor,
    pixel_std: torch.Tensor,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(pixels))
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
    if tuple(tensor.shape[-2:]) != (img_size, img_size):
        tensor = torch.nn.functional.interpolate(tensor, size=(img_size, img_size), mode="bilinear", align_corners=False)
    tensor = tensor.to(device=pixel_mean.device)
    return (tensor - pixel_mean) / pixel_std


@torch.no_grad()
def encode_dataset_latents(
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
        raise ValueError("--frame-batch-size must be positive.")
    pixel_mean, pixel_std = imagenet_pixel_stats(device)
    latents: list[np.ndarray] = []
    for start in tqdm(range(0, indices.shape[0], frame_batch_size), desc="Encoding PushT images", unit="batch"):
        batch_idx = indices[start : start + frame_batch_size]
        batch = preprocess_pixels(pixels[batch_idx], img_size=img_size, pixel_mean=pixel_mean, pixel_std=pixel_std)
        output = model.encoder(batch, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb[:, :embed_dim].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(latents, axis=0)


def fit_tsne(latents: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    if latents.shape[0] < 3:
        raise ValueError("Need at least three dataset samples for t-SNE.")
    perplexity = min(float(args.perplexity), max(1.0, (latents.shape[0] - 1) / 3.0))
    learning_rate: str | float = "auto" if str(args.learning_rate) == "auto" else float(args.learning_rate)
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        learning_rate=learning_rate,
        max_iter=int(args.max_iter),
        init="pca",
        random_state=int(args.seed),
        metric="euclidean",
    )
    return tsne.fit_transform(latents).astype(np.float32)


def save_plot(out_path: Path, formats: list[str], fig: plt.Figure) -> list[Path]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stem_path = out_path.with_suffix("")
    written: list[Path] = []
    for fmt in formats:
        path = stem_path.with_suffix(f".{fmt.lstrip('.')}")
        fig.savefig(path)
        written.append(path)
    return written


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else latest_object_checkpoint(model_dir).resolve()
    )
    config = load_model_config(model_dir)
    img_size = int(config.get("img_size", 224))
    embed_dim = int(config.get("embed_dim", 8))
    device = resolve_device(str(args.device))

    pixels, dataset_indices, dataset_labels = load_obstacle_pixels(
        args.data_path.expanduser().resolve(),
        int(args.seed),
        args.max_samples,
    )
    world_model = load_world_model(checkpoint_path, device)
    dataset_latents = encode_dataset_latents(
        world_model,
        pixels,
        dataset_indices,
        device=device,
        img_size=img_size,
        embed_dim=embed_dim,
        frame_batch_size=int(args.frame_batch_size),
    )
    del world_model

    dataset_scaled = StandardScaler().fit_transform(dataset_latents)
    dataset_xy = fit_tsne(dataset_scaled, args)

    base_font_size = float(args.font_size)
    legend_font_size = float(args.legend_font_size)
    plt.rcParams.update(
        {
            "font.size": base_font_size,
            "axes.labelsize": base_font_size,
            "xtick.labelsize": base_font_size,
            "ytick.labelsize": base_font_size,
            "legend.fontsize": legend_font_size,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 5.5), dpi=int(args.dpi))
    no_obstacle_mask = dataset_labels == 0
    obstacle_mask = dataset_labels == 1
    ax.scatter(
        dataset_xy[no_obstacle_mask, 0],
        dataset_xy[no_obstacle_mask, 1],
        s=7.0,
        color="#4e79a7",
        alpha=0.28,
        edgecolors="none",
        label="no obstacle",
        zorder=1,
    )
    ax.scatter(
        dataset_xy[obstacle_mask, 0],
        dataset_xy[obstacle_mask, 1],
        s=7.0,
        color="#f28e2b",
        alpha=0.34,
        edgecolors="none",
        label="obstacle",
        zorder=1,
    )
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.grid(alpha=0.16, linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.92)
    ax.set_aspect("equal", adjustable="datalim")
    fig.tight_layout()

    written = save_plot(args.out_path.expanduser().resolve(), list(args.formats), fig)
    plt.close(fig)

    metadata = {
        "data_path": str(args.data_path.expanduser().resolve()),
        "model_dir": str(model_dir),
        "checkpoint_path": str(checkpoint_path),
        "dataset_sample_count": int(dataset_indices.shape[0]),
        "label_0_no_obstacle_count": int(np.sum(dataset_labels == 0)),
        "label_1_obstacle_count": int(np.sum(dataset_labels == 1)),
        "embed_dim": int(embed_dim),
        "img_size": int(img_size),
        "tsne_fit_samples": "all_selected_obstacle_dataset_samples_both_labels",
        "perplexity": float(min(float(args.perplexity), max(1.0, (dataset_latents.shape[0] - 1) / 3.0))),
        "max_iter": int(args.max_iter),
        "seed": int(args.seed),
        "written": [str(path) for path in written],
    }
    metadata_path = args.out_path.expanduser().resolve().with_suffix(".json")
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)

    print(
        f"Encoded dataset samples: {dataset_indices.shape[0]} "
        f"({np.sum(dataset_labels == 0)} no obstacle, {np.sum(dataset_labels == 1)} obstacle)"
    )
    print(f"Wrote plot(s): {', '.join(str(path) for path in written)}")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
