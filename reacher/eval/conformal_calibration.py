#!/usr/bin/env python3
"""Train and calibrate a Reacher conformal error predictor."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

if "MPLCONFIGDIR" not in os.environ or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

REPO_ROOT = Path(__file__).resolve().parents[2]
ERROR_MODEL_ROOT = REPO_ROOT / "error_calib" / "error_model"
for path in (REPO_ROOT, ERROR_MODEL_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import lightning as L
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import seaborn as sns
import torch
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

from conformal_prediction import conformal_calibration
from datamodule import ErrorDataModule, ErrorDataset
from error_model import MGNLLPredictor


DEFAULT_DATA_PATH = Path("reacher/eval/reacher_one_step_error_data.pt")
DEFAULT_LOG_DIR = Path("reacher/models/err_pred_logs")
DEFAULT_EXPERIMENT_NAME = "reacher_experiment_markov"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-epochs", type=int, default=3200)
    parser.add_argument("--devices", default=1)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout-prob", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--reg-scale", type=float, default=10.0)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--diagonal", action="store_true")
    parser.add_argument("--disable-spectral-norm", action="store_true")
    parser.add_argument("--cluster-count", type=int, default=30)
    parser.add_argument("--state-pdf-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


@torch.no_grad()
def plot_sigma_eigenvalues(model: MGNLLPredictor, dataloader, save_dir: Path) -> None:
    print("\nComputing eigenvalues of Sigma = L * L^T...")
    model.eval()
    all_eigvals = []
    for x, _ in dataloader:
        L_mat = model(x.to(model.device))
        sigma = torch.bmm(L_mat, L_mat.transpose(1, 2))
        all_eigvals.append(torch.linalg.eigvalsh(sigma).cpu())

    all_eigvals_np = torch.cat(all_eigvals, dim=0).numpy()
    state_dim = all_eigvals_np.shape[1]
    cols = 6
    rows = math.ceil(state_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()
    for idx in range(state_dim):
        ax = axes[idx]
        ax.hist(all_eigvals_np[:, idx], bins=50, color="royalblue", alpha=0.7)
        ax.set_title(f"Eigval {idx + 1} (Ascending)")
        ax.tick_params(axis="both", which="major", labelsize=8)
        ax.set_yscale("log")
    for idx in range(state_dim, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    save_path = save_dir / "sigma_eigenvalues.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved eigenvalue histograms to '{save_path}'")


@torch.no_grad()
def plot_state_pdfs(dataset: ErrorDataset, start_idx: int, end_idx: int, save_dir: Path, num_workers: int) -> None:
    print(f"\nExtracting data from index {start_idx} to {end_idx} for PDF plotting...")
    subset = Subset(dataset, list(range(start_idx, end_idx)))
    loader = DataLoader(
        subset,
        batch_size=4096,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    all_targets = []
    for _, y in loader:
        all_targets.append(y.cpu())
    all_targets_np = torch.cat(all_targets, dim=0).numpy()
    state_dim = all_targets_np.shape[1]
    cols = 6
    rows = math.ceil(state_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()
    for idx in range(state_dim):
        ax = axes[idx]
        data_dim = all_targets_np[:, idx]
        try:
            kde = stats.gaussian_kde(data_dim)
            x_min, x_max = data_dim.min(), data_dim.max()
            padding = 0.1 * (x_max - x_min) if (x_max - x_min) > 0 else 0.1
            x_grid = np.linspace(x_min - padding, x_max + padding, 200)
            ax.plot(x_grid, kde(x_grid), color="darkorange", lw=2)
            ax.hist(data_dim, bins=50, density=True, color="gray", alpha=0.3)
        except Exception as exc:
            print(f"  [Warning] KDE failed for dim {idx}: {exc}. Plotting histogram only.")
            ax.hist(data_dim, bins=50, density=True, color="gray", alpha=0.5)
        ax.set_title(f"State Dim {idx + 1}")
        ax.tick_params(axis="both", which="major", labelsize=8)
    for idx in range(state_dim, len(axes)):
        axes[idx].axis("off")
    plt.tight_layout()
    save_path = save_dir / "state_pdfs_kde.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved state PDF plots to '{save_path}'")


def plot_ratio_distribution(ratios: np.ndarray, alpha: float, save_dir: Path) -> None:
    plt.figure(figsize=(10, 6))
    log_ratios = np.log10(ratios)
    sns.histplot(log_ratios, kde=True, color="royalblue", bins=50)
    plt.axvline(np.mean(log_ratios), color="red", linestyle="--", label=f"Mean: $10^{{{np.mean(log_ratios):.2f}}}$")
    plt.title(f"Distribution of Volume Ratios (Learned/Fixed) | $\\alpha$={alpha}")
    plt.xlabel(r"$\log_{10}(\text{Volume Ratio})$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_path = save_dir / "volume_ratio_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Distribution plot saved as '{save_path}'")


def plot_clustered_coverage(metrics: list[dict[str, float]], target_coverage: float, save_dir: Path) -> None:
    clusters = [m["cluster"] for m in metrics]
    learned = [m["learned_cov"] for m in metrics]
    fixed = [m["fixed_cov"] for m in metrics]
    plt.figure(figsize=(12, 5))
    x = np.arange(len(clusters))
    width = 0.35
    plt.bar(x - width / 2, learned, width, label="Learned (Adaptive)", color="royalblue")
    plt.bar(x + width / 2, fixed, width, label="Fixed (Baseline)", color="lightgrey")
    plt.axhline(y=target_coverage, color="red", linestyle="--", label=f"Target {target_coverage:.2f}")
    plt.ylim(0.7, 1.0)
    plt.xlabel("State-Action Cluster ID")
    plt.ylabel("Empirical Coverage")
    plt.title("Conditional Coverage across State-Action Clusters")
    plt.legend()
    save_path = save_dir / "clustered_coverage.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Clustered coverage plot saved as '{save_path}'")


def calculate_fixed_baseline(dataloader, alpha: float) -> tuple[torch.Tensor, float]:
    all_errors = []
    for _, errors in dataloader:
        all_errors.append(errors)
    all_errors_t = torch.cat(all_errors, dim=0)
    n = all_errors_t.shape[0]
    sigma_emp = (all_errors_t.T @ all_errors_t) / (n - 1)
    L_fixed = torch.linalg.cholesky(sigma_emp + torch.eye(sigma_emp.shape[0]) * 1e-6)
    inv_L_errors = torch.linalg.solve_triangular(L_fixed, all_errors_t.T, upper=False).T
    scores = torch.norm(inv_L_errors, p=2, dim=1).numpy()
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q_fixed = np.sort(scores)[k - 1] if k <= n else np.max(scores)
    return L_fixed, float(q_fixed)


@torch.no_grad()
def evaluate_uncertainty_metrics(model: MGNLLPredictor, dataloader, q_learned: float, L_fixed: torch.Tensor, q_fixed: float):
    model.eval()
    learned_covered = 0
    fixed_covered = 0
    total_points = 0
    volume_ratios = []
    device = model.device
    L_fixed = L_fixed.to(device)
    state_dim = L_fixed.shape[-1]
    log_det_V_fixed = state_dim * torch.log(torch.tensor(q_fixed, device=device)) + torch.logdet(L_fixed)
    for inputs, errors in dataloader:
        inputs = inputs.to(device)
        errors = errors.to(device)
        batch_size = inputs.size(0)
        L_learned = model(inputs)
        alpha_learned = torch.linalg.solve_triangular(L_learned, errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_learned = torch.norm(alpha_learned, p=2, dim=1)
        learned_covered += (scores_learned <= q_learned).sum().item()
        alpha_fixed = torch.linalg.solve_triangular(L_fixed.unsqueeze(0), errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_fixed = torch.norm(alpha_fixed, p=2, dim=1)
        fixed_covered += (scores_fixed <= q_fixed).sum().item()
        log_det_L_learned = torch.logdet(L_learned)
        log_det_V_learned = state_dim * torch.log(torch.tensor(q_learned, device=device)) + log_det_L_learned
        volume_ratios.append(torch.exp(log_det_V_learned - log_det_V_fixed).cpu())
        total_points += batch_size
    all_ratios = torch.cat(volume_ratios).numpy()
    return learned_covered / total_points, fixed_covered / total_points, all_ratios


@torch.no_grad()
def evaluate_clustered_coverage(
    model: MGNLLPredictor,
    dataloader,
    q_learned: float,
    L_fixed: torch.Tensor,
    q_fixed: float,
    n_clusters: int,
) -> list[dict[str, float]]:
    model.eval()
    device = model.device
    L_fixed = L_fixed.to(device)
    all_inputs = []
    all_learned_scores = []
    all_fixed_scores = []
    for inputs, errors in dataloader:
        inputs = inputs.to(device)
        errors = errors.to(device)
        L_learned = model(inputs)
        alpha_learned = torch.linalg.solve_triangular(L_learned, errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_learned = torch.norm(alpha_learned, p=2, dim=1)
        alpha_fixed = torch.linalg.solve_triangular(L_fixed.unsqueeze(0), errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_fixed = torch.norm(alpha_fixed, p=2, dim=1)
        all_inputs.append(inputs.cpu())
        all_learned_scores.append(scores_learned.cpu())
        all_fixed_scores.append(scores_fixed.cpu())
    all_inputs_np = torch.cat(all_inputs).numpy()
    all_learned_scores_np = torch.cat(all_learned_scores).numpy()
    all_fixed_scores_np = torch.cat(all_fixed_scores).numpy()
    scaled_inputs = StandardScaler().fit_transform(all_inputs_np)
    cluster_labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(scaled_inputs)
    cluster_metrics = []
    for cluster_idx in range(n_clusters):
        mask = cluster_labels == cluster_idx
        if np.sum(mask) == 0:
            continue
        cluster_metrics.append(
            {
                "cluster": int(cluster_idx),
                "size": int(np.sum(mask)),
                "learned_cov": float(np.mean(all_learned_scores_np[mask] <= q_learned)),
                "fixed_cov": float(np.mean(all_fixed_scores_np[mask] <= q_fixed)),
            }
        )
    return cluster_metrics


def run_pipeline(args: argparse.Namespace) -> None:
    if not 0.0 < args.alpha < 1.0:
        raise ValueError("--alpha must be in (0, 1).")
    data_path = args.data_path.expanduser().resolve()
    log_dir = args.log_dir.expanduser().resolve()
    log_dir.mkdir(parents=True, exist_ok=True)
    L.seed_everything(args.seed, workers=True)

    dm = ErrorDataModule(data_path=str(data_path), batch_size=args.batch_size)
    dm.setup()

    model = MGNLLPredictor(
        input_dim=dm.input_dim,
        state_dim=dm.state_dim,
        num_layers=args.num_layers,
        hidden_dim=args.hidden_dim,
        diagonal=args.diagonal,
        lr=args.lr,
        reg_scale=args.reg_scale,
        dropout_prob=args.dropout_prob,
        use_spectral_norm= False,
    )

    logger = TensorBoardLogger(save_dir=str(log_dir), name=args.experiment_name)
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-error-model-{epoch:02d}",
    )
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices=args.devices,
        logger=logger,
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=args.gradient_clip_val,
    )
    trainer.fit(model, datamodule=dm)

    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        artifact_dir = Path(best_ckpt).parent
        print(f"\n[Artifacts] Detected checkpoint folder: {artifact_dir}")
    else:
        artifact_dir = Path(logger.log_dir) / "checkpoints"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Warning] Best checkpoint path unresolvable. Defaulting to logger dir: {artifact_dir}")

    print(f"\nLoading best model weights from: {best_ckpt}")
    model = MGNLLPredictor.load_from_checkpoint(best_ckpt)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    full_dataset = ErrorDataset(torch.load(data_path, map_location="cpu", weights_only=False))
    n_total = len(full_dataset)
    calib_start = int(0.5 * n_total)
    calib_end = n_total

    print("\n--- Performing Conformal Calibration ---")
    q_learned = conformal_calibration(model, dm.calib_dataloader(), alpha=args.alpha)
    L_fixed, q_fixed = calculate_fixed_baseline(dm.calib_dataloader(), alpha=args.alpha)
    print(f"Resulting Learned Quantile (q_learned): {q_learned:.6f}")
    print(f"Resulting Fixed Quantile (q_fixed):    {q_fixed:.6f}")

    print("\nEvaluating global metrics across holdout dataset...")
    learned_cov, fixed_cov, all_ratios = evaluate_uncertainty_metrics(model, dm.val_dataloader(), q_learned, L_fixed, q_fixed)
    cluster_metrics = evaluate_clustered_coverage(
        model,
        dm.val_dataloader(),
        q_learned,
        L_fixed,
        q_fixed,
        n_clusters=args.cluster_count,
    )

    print(f"\nWriting diagnostic visualizations to: {artifact_dir}")
    plot_sigma_eigenvalues(model, dm.calib_dataloader(), save_dir=artifact_dir)
    plot_state_pdfs(full_dataset, start_idx=calib_start, end_idx=calib_end, save_dir=artifact_dir, num_workers=args.state_pdf_workers)
    plot_ratio_distribution(all_ratios, args.alpha, save_dir=artifact_dir)
    plot_clustered_coverage(cluster_metrics, 1 - args.alpha, save_dir=artifact_dir)

    results = {
        "q_learned": float(q_learned),
        "q_fixed": float(q_fixed),
        "alpha": float(args.alpha),
        "checkpoint": str(best_ckpt),
        "data_path": str(data_path),
        "calib_range": [calib_start, calib_end],
        "holdout_metrics": {
            "learned_coverage": float(learned_cov),
            "fixed_baseline_coverage": float(fixed_cov),
            "mean_volume_ratio": float(np.mean(all_ratios)),
        },
    }
    config_path = artifact_dir / "conformal_config.json"
    with config_path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=4)

    print("\n" + "=" * 50)
    print(f"HOLDOUT SET METRICS SUMMARY (Target: {1 - args.alpha:.2f})")
    print(f"Learned Model Coverage:  {learned_cov:.4f}")
    print(f"Fixed Baseline Coverage: {fixed_cov:.4f}")
    print(f"Mean Volume Ratio:       {np.mean(all_ratios):.4f}")
    print(f"Configuration profile written to: '{config_path}'")
    print("=" * 50)


if __name__ == "__main__":
    run_pipeline(parse_args())
