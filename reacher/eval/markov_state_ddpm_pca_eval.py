#!/usr/bin/env python3
"""Evaluate a Markov-state DDPM with PCA visualizations and noisy/OOD state diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch

from reacher.train.latent_state_ddpm import LatentStateDDPM, ResidualBlock, SinusoidalTimeEmbedding


globals()["LatentStateDDPM"] = LatentStateDDPM
globals()["ResidualBlock"] = ResidualBlock
globals()["SinusoidalTimeEmbedding"] = SinusoidalTimeEmbedding


DEFAULT_MODEL_DIR = Path("reacher/models/markov_state_ddpm")
DEFAULT_OUT_DIRNAME = "eval_pca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--model-object", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--max-train-points", type=int, default=5000)
    parser.add_argument("--max-val-points", type=int, default=2000)
    parser.add_argument("--max-plot-points", type=int, default=2000)
    parser.add_argument("--timestep", type=int, nargs="+", default=[50, 200, 500, 900])
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def require_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        device_arg = "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    return torch.device(device_arg)


def latest_model_object(model_dir: Path) -> Path:
    final_candidates = sorted(model_dir.glob("*_final_object.pt"))
    if final_candidates:
        return final_candidates[-1]
    pattern = re.compile(r".*_epoch_(\d+)_object\.pt$")
    candidates: list[tuple[int, Path]] = []
    for path in model_dir.glob("*_epoch_*_object.pt"):
        match = pattern.match(path.name)
        if match is not None:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No DDPM object checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]


def load_json(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_cached_states(cache_manifest_path: Path) -> torch.Tensor:
    manifest = load_json(cache_manifest_path)
    states = []
    for entry in manifest:
        cache_path = Path(entry["cache_path"]).expanduser().resolve()
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        states.append(payload["states"].float().contiguous())
    if not states:
        raise ValueError(f"No cached states found in manifest: {cache_manifest_path}")
    return torch.cat(states, dim=0)


def split_states(states: torch.Tensor, train_split: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    if states.shape[0] < 2:
        raise ValueError(f"Need at least 2 states for splitting, got {states.shape[0]}.")
    train_len = int(states.shape[0] * train_split)
    train_len = min(max(train_len, 1), states.shape[0] - 1)
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(states.shape[0], generator=generator)
    train_idx = permutation[:train_len]
    val_idx = permutation[train_len:]
    return states[train_idx], states[val_idx]


def maybe_subsample(x: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    if x.shape[0] <= max_points:
        return x
    generator = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=generator)[:max_points]
    return x[idx]


def fit_pca(x: torch.Tensor, n_components: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    x_mean = x.mean(dim=0, keepdim=True)
    x_centered = x - x_mean
    _, _, v = torch.linalg.svd(x_centered, full_matrices=False)
    components = v[:n_components].T.contiguous()
    return x_mean, components


def pca_project(x: torch.Tensor, mean: torch.Tensor, components: torch.Tensor) -> np.ndarray:
    projected = (x - mean) @ components
    return projected.detach().cpu().numpy()


def pairwise_min_distance(query: torch.Tensor, reference: torch.Tensor, chunk_size: int = 2048) -> torch.Tensor:
    mins = []
    for start in range(0, query.shape[0], chunk_size):
        chunk = query[start : start + chunk_size]
        dists = torch.cdist(chunk, reference)
        mins.append(dists.min(dim=1).values.cpu())
    return torch.cat(mins, dim=0)


def build_corrupted_states(states: torch.Tensor, markov_deriv: int, seed: int) -> torch.Tensor:
    if markov_deriv <= 0:
        generator = torch.Generator().manual_seed(seed)
        noise = 0.25 * torch.randn(states.shape, generator=generator, dtype=states.dtype)
        return states + noise
    component_count = markov_deriv + 1
    if states.shape[1] % component_count != 0:
        raise ValueError(
            f"State dim {states.shape[1]} is not divisible by component count {component_count}."
        )
    embed_dim = states.shape[1] // component_count
    z = states[:, :embed_dim]
    deriv = states[:, embed_dim:]
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(states.shape[0], generator=generator)
    deriv_bad = deriv[perm]
    return torch.cat((z, deriv_bad), dim=-1)


def scatter_plot(
    points: np.ndarray,
    values: np.ndarray | None,
    *,
    title: str,
    out_path: Path,
    cmap: str = "viridis",
    colorbar_label: str | None = None,
    alpha: float = 0.7,
    point_size: float = 8.0,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    if values is None:
        ax.scatter(points[:, 0], points[:, 1], s=point_size, alpha=alpha, linewidths=0.0)
    else:
        scatter = ax.scatter(points[:, 0], points[:, 1], c=values, cmap=cmap, s=point_size, alpha=alpha, linewidths=0.0)
        cbar = fig.colorbar(scatter, ax=ax)
        if colorbar_label is not None:
            cbar.set_label(colorbar_label)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def overlay_plot(
    base_points: np.ndarray,
    overlay_points: np.ndarray,
    *,
    title: str,
    out_path: Path,
    base_label: str,
    overlay_label: str,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 6), dpi=160)
    ax.scatter(base_points[:, 0], base_points[:, 1], s=7.0, alpha=0.18, linewidths=0.0, label=base_label, color="tab:blue")
    ax.scatter(overlay_points[:, 0], overlay_points[:, 1], s=8.0, alpha=0.65, linewidths=0.0, label=overlay_label, color="tab:orange")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(title)
    ax.grid(True, alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def summarize(values: torch.Tensor) -> dict[str, float]:
    return {
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def main() -> None:
    args = parse_args()
    model_dir = args.model_dir.expanduser().resolve()
    config = load_json(model_dir / "config.json")
    cache_manifest_path = model_dir / "cache_manifest.json"
    if not cache_manifest_path.is_file():
        raise FileNotFoundError(f"Cache manifest not found: {cache_manifest_path}")

    model_object = args.model_object.expanduser().resolve() if args.model_object is not None else latest_model_object(model_dir)
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir is not None else model_dir / DEFAULT_OUT_DIRNAME
    device = require_device(args.device)

    model = torch.load(model_object, map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)

    states = load_cached_states(cache_manifest_path)
    train_states, val_states = split_states(states, float(config["train_split"]), int(config["seed"]))

    train_states = maybe_subsample(train_states, args.max_train_points, args.seed)
    val_states = maybe_subsample(val_states, args.max_val_points, args.seed + 1)
    plot_train_states = maybe_subsample(train_states, args.max_plot_points, args.seed + 2)
    plot_val_states = maybe_subsample(val_states, args.max_plot_points, args.seed + 3)

    pca_mean, pca_components = fit_pca(plot_train_states)
    train_pca = pca_project(plot_train_states, pca_mean, pca_components)
    val_pca = pca_project(plot_val_states, pca_mean, pca_components)

    scatter_plot(
        train_pca,
        None,
        title="Train Markov States PCA",
        out_path=out_dir / "pca_train_states.png",
    )
    overlay_plot(
        train_pca,
        val_pca,
        title="Train vs Val Markov States PCA",
        out_path=out_dir / "pca_train_vs_val.png",
        base_label="train",
        overlay_label="val",
    )

    markov_deriv = int(config["markov_deriv"])
    corrupted_states = build_corrupted_states(plot_val_states, markov_deriv, args.seed + 4)
    corrupted_pca = pca_project(corrupted_states, pca_mean, pca_components)
    overlay_plot(
        train_pca,
        corrupted_pca,
        title="Train vs Corrupted Val Markov States PCA",
        out_path=out_dir / "pca_corrupted_overlay.png",
        base_label="train",
        overlay_label="corrupted val",
    )

    reference_train = train_states.to(device)
    plot_val_states_device = plot_val_states.to(device)
    clean_nn = pairwise_min_distance(plot_val_states_device, reference_train).numpy()
    scatter_plot(
        val_pca,
        clean_nn,
        title="Val Markov States PCA Colored by NN Distance",
        out_path=out_dir / "pca_val_nn_distance.png",
        cmap="magma",
        colorbar_label="min train distance",
    )

    summary: dict[str, object] = {
        "model_object": str(model_object),
        "num_train_states": int(train_states.shape[0]),
        "num_val_states": int(val_states.shape[0]),
        "plots": [],
        "timesteps": {},
    }

    for timestep in args.timestep:
        if timestep < 0 or timestep >= int(model.diffusion_steps):
            raise ValueError(f"timestep must be in [0, {int(model.diffusion_steps) - 1}], got {timestep}.")
        t_tensor = torch.full((plot_val_states.shape[0],), timestep, device=device, dtype=torch.long)
        x0_norm = model.normalize(plot_val_states_device)
        noise = torch.randn_like(x0_norm)
        x_t_norm = model.q_sample(x0_norm, t_tensor, noise)
        pred_noise = model(x_t_norm, t_tensor)
        denoise_mse = ((pred_noise - noise) ** 2).mean(dim=1)
        x_t_raw = model.denormalize(x_t_norm)
        score = model.score(x_t_raw, t_tensor)
        score_norm = torch.linalg.vector_norm(score, dim=1)
        noisy_nn = pairwise_min_distance(x_t_raw, reference_train)

        noisy_pca = pca_project(x_t_raw.cpu(), pca_mean, pca_components)
        denoise_np = denoise_mse.detach().cpu().numpy()
        score_norm_np = score_norm.detach().cpu().numpy()
        noisy_nn_np = noisy_nn.detach().cpu().numpy()

        overlay_plot(
            train_pca,
            noisy_pca,
            title=f"Train vs Noisy Val States PCA (t={timestep})",
            out_path=out_dir / f"pca_noisy_overlay_t{timestep:04d}.png",
            base_label="train",
            overlay_label=f"noisy val t={timestep}",
        )
        scatter_plot(
            noisy_pca,
            denoise_np,
            title=f"Noisy Val States PCA Colored by Denoising MSE (t={timestep})",
            out_path=out_dir / f"pca_noisy_denoise_mse_t{timestep:04d}.png",
            cmap="viridis",
            colorbar_label="denoising MSE",
        )
        scatter_plot(
            noisy_pca,
            score_norm_np,
            title=f"Noisy Val States PCA Colored by Score Norm (t={timestep})",
            out_path=out_dir / f"pca_noisy_score_norm_t{timestep:04d}.png",
            cmap="plasma",
            colorbar_label="score norm",
        )
        scatter_plot(
            noisy_pca,
            noisy_nn_np,
            title=f"Noisy Val States PCA Colored by NN Distance (t={timestep})",
            out_path=out_dir / f"pca_noisy_nn_distance_t{timestep:04d}.png",
            cmap="magma",
            colorbar_label="min train distance",
        )

        summary["timesteps"][str(timestep)] = {
            "denoising_mse": summarize(denoise_mse.detach().cpu()),
            "score_norm": summarize(score_norm.detach().cpu()),
            "min_train_distance": summarize(noisy_nn.detach().cpu()),
        }
        summary["plots"].extend(
            [
                str(out_dir / f"pca_noisy_overlay_t{timestep:04d}.png"),
                str(out_dir / f"pca_noisy_denoise_mse_t{timestep:04d}.png"),
                str(out_dir / f"pca_noisy_score_norm_t{timestep:04d}.png"),
                str(out_dir / f"pca_noisy_nn_distance_t{timestep:04d}.png"),
            ]
        )

    corrupted_states_device = corrupted_states.to(device)
    corrupted_pca_full = pca_project(corrupted_states_device.cpu(), pca_mean, pca_components)
    corrupted_nn = pairwise_min_distance(corrupted_states_device, reference_train)
    corrupted_score_t = torch.full((corrupted_states_device.shape[0],), min(args.timestep), device=device, dtype=torch.long)
    corrupted_score_norm = torch.linalg.vector_norm(model.score(corrupted_states_device, corrupted_score_t), dim=1)

    scatter_plot(
        corrupted_pca_full,
        corrupted_nn.cpu().numpy(),
        title="Corrupted Val States PCA Colored by NN Distance",
        out_path=out_dir / "pca_corrupted_nn_distance.png",
        cmap="magma",
        colorbar_label="min train distance",
    )
    scatter_plot(
        corrupted_pca_full,
        corrupted_score_norm.detach().cpu().numpy(),
        title=f"Corrupted Val States PCA Colored by Score Norm (t={min(args.timestep)})",
        out_path=out_dir / "pca_corrupted_score_norm.png",
        cmap="plasma",
        colorbar_label="score norm",
    )

    summary["plots"].extend(
        [
            str(out_dir / "pca_train_states.png"),
            str(out_dir / "pca_train_vs_val.png"),
            str(out_dir / "pca_val_nn_distance.png"),
            str(out_dir / "pca_corrupted_overlay.png"),
            str(out_dir / "pca_corrupted_nn_distance.png"),
            str(out_dir / "pca_corrupted_score_norm.png"),
        ]
    )
    summary["clean_val_nn_distance"] = summarize(torch.from_numpy(clean_nn))
    summary["corrupted_nn_distance"] = summarize(corrupted_nn.cpu())
    summary["corrupted_score_norm"] = summarize(corrupted_score_norm.detach().cpu())

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
