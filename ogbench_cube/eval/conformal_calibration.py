import torch
import numpy as np
import json
import math
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Subset

ERROR_MODEL_ROOT = Path(__file__).resolve().parents[2] / "error_calib" / "error_model"
if str(ERROR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ERROR_MODEL_ROOT))

from datamodule import ErrorDataModule, ErrorDataset
from error_model import MGNLLPredictor
from conformal_prediction import conformal_calibration

# ==============================================================================
# 1. DIAGNOSTIC PLOTTING UTILITIES
# ==============================================================================

@torch.no_grad()
def plot_sigma_eigenvalues(model, dataloader, save_dir: Path):
    """
    Computes the eigenvalues of Sigma = L * L^T for each sample
    and plots a histogram of the eigenvalue distributions.
    """
    print("\nComputing eigenvalues of Sigma = L * L^T...")
    model.eval()
    all_eigvals = []
    
    for x, _ in dataloader:
        L_mat = model(x.to(model.device))
        Sigma = torch.bmm(L_mat, L_mat.transpose(1, 2))
        eigvals = torch.linalg.eigvalsh(Sigma)
        all_eigvals.append(eigvals.cpu())
        
    all_eigvals = torch.cat(all_eigvals, dim=0).numpy()
    state_dim = all_eigvals.shape[1]
    
    print(f"Plotting histograms for {state_dim} dimensions...")
    
    cols = 6
    rows = math.ceil(state_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()
    
    for i in range(state_dim):
        ax = axes[i]
        ax.hist(all_eigvals[:, i], bins=50, color='royalblue', alpha=0.7)
        ax.set_title(f"Eigval {i+1} (Ascending)")
        ax.tick_params(axis='both', which='major', labelsize=8)
        ax.set_yscale('log') 
        
    for j in range(state_dim, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    save_path = save_dir / "sigma_eigenvalues.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved eigenvalue histograms to '{save_path}'\n")


@torch.no_grad()
def plot_state_pdfs(dataset, start_idx, end_idx, save_dir: Path):
    """
    Extracts the targets (y) from the dataset over a specific index range 
    and plots their KDE smoothed PDFs.
    """
    print(f"\nExtracting data from index {start_idx} to {end_idx} for PDF plotting...")
    
    subset_indices = list(range(start_idx, end_idx))
    subset = Subset(dataset, subset_indices)
    loader = DataLoader(subset, batch_size=4096, num_workers=4, pin_memory=torch.cuda.is_available())
    
    all_targets = []
    for _, y in loader:
        all_targets.append(y.cpu())
        
    all_targets = torch.cat(all_targets, dim=0).numpy()
    state_dim = all_targets.shape[1]
    
    print(f"Plotting KDE PDFs for {state_dim} state dimensions...")
    
    cols = 6
    rows = math.ceil(state_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()
    
    for i in range(state_dim):
        ax = axes[i]
        data_dim = all_targets[:, i]
        
        try:
            kde = stats.gaussian_kde(data_dim)
            x_min, x_max = data_dim.min(), data_dim.max()
            padding = 0.1 * (x_max - x_min) if (x_max - x_min) > 0 else 0.1
            x_grid = np.linspace(x_min - padding, x_max + padding, 200)
            
            ax.plot(x_grid, kde(x_grid), color='darkorange', lw=2, label="KDE")
            ax.hist(data_dim, bins=50, density=True, color='gray', alpha=0.3)
        except Exception as e:
            print(f"  [Warning] KDE failed for dim {i}: {e}. Plotting histogram only.")
            ax.hist(data_dim, bins=50, density=True, color='gray', alpha=0.5)
            
        ax.set_title(f"State Dim {i+1}")
        ax.tick_params(axis='both', which='major', labelsize=8)
        
    for j in range(state_dim, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    save_path = save_dir / "state_pdfs_kde.png"
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved state PDF plots to '{save_path}'\n")


def plot_ratio_distribution(ratios, alpha, save_dir: Path):
    """
    Plots the log-scaled distribution of volume ratios.
    """
    plt.figure(figsize=(10, 6))
    log_ratios = np.log10(ratios)
    
    sns.histplot(log_ratios, kde=True, color='royalblue', bins=50)
    plt.axvline(np.mean(log_ratios), color='red', linestyle='--', label=f'Mean: $10^{{{np.mean(log_ratios):.2f}}}$')
    
    plt.title(f"Distribution of Volume Ratios (Learned/Fixed) | $\\alpha$={alpha}")
    plt.xlabel(r"$\log_{10}(\text{Volume Ratio})$")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = save_dir / "volume_ratio_distribution.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Distribution plot saved as '{save_path}'")


def plot_clustered_coverage(metrics, target_coverage, save_dir: Path):
    """
    Plots the conditional coverage across different clusters in state-action space.
    """
    clusters = [m['cluster'] for m in metrics]
    learned = [m['learned_cov'] for m in metrics]
    fixed = [m['fixed_cov'] for m in metrics]

    plt.figure(figsize=(12, 5))
    x = np.arange(len(clusters))
    width = 0.35

    plt.bar(x - width/2, learned, width, label='Learned (Adaptive)', color='royalblue')
    plt.bar(x + width/2, fixed, width, label='Fixed (Baseline)', color='lightgrey')
    
    plt.axhline(y=target_coverage, color='red', linestyle='--', label=f'Target {target_coverage:.2f}')
    plt.ylim(0.7, 1.0)
    plt.xlabel("State-Action Cluster ID")
    plt.ylabel("Empirical Coverage")
    plt.title("Conditional Coverage across State-Action Clusters")
    plt.legend()
    
    save_path = save_dir / "clustered_coverage.png"
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Clustered coverage plot saved as '{save_path}'")

# ==============================================================================
# 2. CONFORMAL EVALUATION ANALYSIS
# ==============================================================================

def calculate_fixed_baseline(dataloader, alpha):
    """
    Computes a fixed ellipsoidal bound based on the empirical covariance of errors.
    """
    all_errors = []
    for _, errors in dataloader:
        all_errors.append(errors)
    all_errors = torch.cat(all_errors, dim=0)
    
    n = all_errors.shape[0]
    sigma_emp = (all_errors.T @ all_errors) / (n - 1)
    
    L_fixed = torch.linalg.cholesky(sigma_emp + torch.eye(sigma_emp.shape[0]) * 1e-6)
    inv_L_errors = torch.linalg.solve_triangular(L_fixed, all_errors.T, upper=False).T
    scores = torch.norm(inv_L_errors, p=2, dim=1).numpy()
    
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q_fixed = np.sort(scores)[k - 1] if k <= n else np.max(scores)
    
    return L_fixed, q_fixed


@torch.no_grad()
def evaluate_uncertainty_metrics(model, dataloader, q_learned, L_fixed, q_fixed):
    """
    Computes empirical coverage over the evaluation set and full distribution of volume ratios.
    """
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
        inputs, errors = inputs.to(device), errors.to(device)
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
        
        batch_ratios = torch.exp(log_det_V_learned - log_det_V_fixed)
        volume_ratios.append(batch_ratios.cpu())
        total_points += batch_size

    all_ratios = torch.cat(volume_ratios).numpy()
    return (learned_covered / total_points), (fixed_covered / total_points), all_ratios


@torch.no_grad()
def evaluate_clustered_coverage(model, dataloader, q_learned, L_fixed, q_fixed, n_clusters=30):
    """
    Clusters the latent/input space to assess localized conditional coverage behaviors.
    """
    model.eval()
    device = model.device
    L_fixed = L_fixed.to(device)
    
    all_inputs = []
    all_learned_scores = []
    all_fixed_scores = []

    for inputs, errors in dataloader:
        inputs, errors = inputs.to(device), errors.to(device)
        
        L_learned = model(inputs)
        alpha_learned = torch.linalg.solve_triangular(L_learned, errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_learned = torch.norm(alpha_learned, p=2, dim=1)
        
        alpha_fixed = torch.linalg.solve_triangular(L_fixed.unsqueeze(0), errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_fixed = torch.norm(alpha_fixed, p=2, dim=1)

        all_inputs.append(inputs.cpu())
        all_learned_scores.append(scores_learned.cpu())
        all_fixed_scores.append(scores_fixed.cpu())

    all_inputs = torch.cat(all_inputs).numpy()
    all_learned_scores = torch.cat(all_learned_scores).numpy()
    all_fixed_scores = torch.cat(all_fixed_scores).numpy()

    scaler = StandardScaler()
    scaled_inputs = scaler.fit_transform(all_inputs)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_inputs)

    cluster_metrics = []
    for i in range(n_clusters):
        mask = (cluster_labels == i)
        if np.sum(mask) == 0: continue
        
        c_learned_cov = np.mean(all_learned_scores[mask] <= q_learned)
        c_fixed_cov = np.mean(all_fixed_scores[mask] <= q_fixed)
        
        cluster_metrics.append({
            'cluster': i,
            'size': np.sum(mask),
            'learned_cov': c_learned_cov,
            'fixed_cov': c_fixed_cov
        })

    return cluster_metrics

# ==============================================================================
# 3. PIPELINE ORCHESTRATION
# ==============================================================================

def run_pipeline(data_path: str, base_log_dir: str, alpha: float = 0.1, batch_size: int = 4096):
    lightning_dir = Path(base_log_dir)
    
    # 1. Setup DataModule
    dm = ErrorDataModule(data_path=data_path, batch_size=batch_size)
    dm.setup()
    
    model = MGNLLPredictor(
        input_dim=dm.input_dim, state_dim=dm.state_dim, 
        num_layers=2, hidden_dim=128, diagonal=False, 
        lr=0.0003, reg_scale=10, dropout_prob=0.3, use_spectral_norm=False
    )

    logger = TensorBoardLogger(save_dir=str(lightning_dir), name="cube_experiment_embed_8")

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss", mode="min", save_top_k=1, filename="best-error-model-{epoch:02d}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 2. Train Model
    trainer = L.Trainer(
        max_epochs=6400, 
        accelerator="auto", 
        devices=1,
        logger=logger,                        
        default_root_dir=lightning_dir,       
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0  
    )
    trainer.fit(model, datamodule=dm)

    # 3. Resolve output paths dynamically based on current Lightning Logger version directory
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt and Path(best_ckpt).exists():
        plot_save_dir = Path(best_ckpt).parent
        print(f"\n[Artifacts] Detected checkpoint folder: {plot_save_dir}")
    else:
        plot_save_dir = Path(logger.log_dir) / "checkpoints"
        plot_save_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[Warning] Best checkpoint path unresolvable. Defaulting to logger dir: {plot_save_dir}")

    # 4. Load optimized weights for Calibration
    print(f"\nLoading best model weights from: {best_ckpt}")
    model = MGNLLPredictor.load_from_checkpoint(best_ckpt)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # 5. Extract specific subset control parameters matching plot_stats partitioning
    full_dataset = ErrorDataset(torch.load(data_path))
    n_total = len(full_dataset)
    calib_start = int(0.5 * n_total)
    calib_end = int(0.9 * n_total)

    # 6. Conformal Calibration & Baselines
    print("\n--- Performing Conformal Calibration ---")
    q_learned = conformal_calibration(model, dm.calib_dataloader(), alpha=alpha)
    L_fixed, q_fixed = calculate_fixed_baseline(dm.calib_dataloader(), alpha=alpha)
    
    print(f"Resulting Learned Quantile (q_learned): {q_learned:.6f}")
    print(f"Resulting Fixed Quantile (q_fixed):    {q_fixed:.6f}")

    # 7. Evaluate Global Metrics and Localized Clusters
    print("\nEvaluating global metrics across holdout dataset...")
    learned_cov, fixed_cov, all_ratios = evaluate_uncertainty_metrics(
        model, dm.val_dataloader(), q_learned, L_fixed, q_fixed
    )
    cluster_metrics = evaluate_clustered_coverage(model, dm.val_dataloader(), q_learned, L_fixed, q_fixed)

    # 8. Output Visualizations straight into the checkpoint folder
    print(f"\nWriting diagnostic visualizations to: {plot_save_dir}")
    plot_sigma_eigenvalues(model, dm.calib_dataloader(), save_dir=plot_save_dir)
    plot_state_pdfs(full_dataset, start_idx=calib_start, end_idx=n_total, save_dir=plot_save_dir)
    plot_ratio_distribution(all_ratios, alpha, save_dir=plot_save_dir)
    plot_clustered_coverage(cluster_metrics, 1 - alpha, save_dir=plot_save_dir)

    # 9. Save Configuration Specs for Downstream Use
    conformal_config_path = plot_save_dir / "conformal_config.json"
    results = {
        "q_learned": float(q_learned),
        "q_fixed": float(q_fixed),
        "alpha": alpha,
        "checkpoint": str(best_ckpt),
        "calib_range": [calib_start, calib_end],
        "holdout_metrics": {
            "learned_coverage": float(learned_cov),
            "fixed_baseline_coverage": float(fixed_cov),
            "mean_volume_ratio": float(np.mean(all_ratios))
        }
    }
    with open(conformal_config_path, "w") as f:
        json.dump(results, f, indent=4)
        
    print(f"\n" + "="*50)
    print(f"HOLDOUT SET METRICS SUMMARY (Target: {1-alpha:.2f})")
    print(f"Learned Model Coverage:  {learned_cov:.4f}")
    print(f"Fixed Baseline Coverage: {fixed_cov:.4f}")
    print(f"Mean Volume Ratio:       {np.mean(all_ratios):.4f}")
    print(f"Configuration profile written to: '{conformal_config_path}'")
    print("="*50)


if __name__ == "__main__":
    DATA_PATH = "ogbench_cube/eval/ogbench_one_step_error_data_embed_8.pt"
    LOG_DIR = "ogbench_cube/models/err_pred_logs"
    ALPHA = 0.1
    
    run_pipeline(data_path=DATA_PATH, base_log_dir=LOG_DIR, alpha=ALPHA)
