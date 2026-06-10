import torch
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

ERROR_MODEL_ROOT = Path(__file__).resolve().parents[2] / "error_calib" / "error_model"
if str(ERROR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ERROR_MODEL_ROOT))

from datamodule import ErrorDataModule
from error_model import MGNLLPredictor
from conformal_prediction import conformal_calibration

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

@torch.no_grad()
def evaluate_clustered_coverage(model, dataloader, q_learned, L_fixed, q_fixed, n_clusters=30):
    model.eval()
    device = model.device
    L_fixed = L_fixed.to(device)
    
    all_inputs = []
    all_learned_scores = []
    all_fixed_scores = []

    # 1. Collect all scores and inputs from validation set
    for inputs, errors in dataloader:
        inputs, errors = inputs.to(device), errors.to(device)
        
        # Learned Model Scores
        L_learned = model(inputs)
        alpha_learned = torch.linalg.solve_triangular(L_learned, errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_learned = torch.norm(alpha_learned, p=2, dim=1)
        
        # Fixed Baseline Scores
        alpha_fixed = torch.linalg.solve_triangular(L_fixed.unsqueeze(0), errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_fixed = torch.norm(alpha_fixed, p=2, dim=1)

        all_inputs.append(inputs.cpu())
        all_learned_scores.append(scores_learned.cpu())
        all_fixed_scores.append(scores_fixed.cpu())

    all_inputs = torch.cat(all_inputs).numpy()
    all_learned_scores = torch.cat(all_learned_scores).numpy()
    all_fixed_scores = torch.cat(all_fixed_scores).numpy()

    # 2. Cluster the State-Action space
    # Standardizing is important for KMeans in high-D latent spaces (48D)
    scaler = StandardScaler()
    scaled_inputs = scaler.fit_transform(all_inputs)
    
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_inputs)

    # 3. Calculate per-cluster coverage
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

def plot_clustered_coverage(metrics, target_coverage):
    clusters = [m['cluster'] for m in metrics]
    learned = [m['learned_cov'] for m in metrics]
    fixed = [m['fixed_cov'] for m in metrics]

    plt.figure(figsize=(12, 5))
    x = np.arange(len(clusters))
    width = 0.35

    plt.bar(x - width/2, learned, width, label='Learned (Adaptive)', color='royalblue')
    plt.bar(x + width/2, fixed, width, label='Fixed (Baseline)', color='lightgrey')
    
    plt.axhline(y=target_coverage, color='red', linestyle='--', label='Target 0.90')
    plt.ylim(0.7, 1.0) # Zoom in to see the variance
    plt.xlabel("State-Action Cluster ID")
    plt.ylabel("Empirical Coverage")
    plt.title("Conditional Coverage across State-Action Clusters")
    plt.legend()
    plt.savefig("ogbench_cube/eval/clustered_coverage.png")
    print("Clustered coverage plot saved as 'clustered_coverage.png'")

def calculate_fixed_baseline(dataloader, alpha):
    """
    Computes a fixed ellipsoidal bound based on the empirical 
    covariance of the calibration set errors[cite: 215, 216].
    """
    all_errors = []
    for _, errors in dataloader:
        all_errors.append(errors)
    all_errors = torch.cat(all_errors, dim=0)
    
    n = all_errors.shape[0]
    sigma_emp = (all_errors.T @ all_errors) / (n - 1)
    
    # Cholesky Factor L_fixed [cite: 135, 641]
    L_fixed = torch.linalg.cholesky(sigma_emp + torch.eye(sigma_emp.shape[0]) * 1e-6)
    
    # Non-conformity scores [cite: 123, 126]
    inv_L_errors = torch.linalg.solve_triangular(L_fixed, all_errors.T, upper=False).T
    scores = torch.norm(inv_L_errors, p=2, dim=1).numpy()
    
    # Conformal Quantile q_fixed [cite: 71, 104]
    k = int(np.ceil((n + 1) * (1 - alpha)))
    q_fixed = np.sort(scores)[k - 1] if k <= n else np.max(scores)
    
    return L_fixed, q_fixed

@torch.no_grad()
def evaluate_uncertainty_metrics(model, dataloader, q_learned, L_fixed, q_fixed):
    """
    Computes coverage and full distribution of volume ratios[cite: 17, 150].
    """
    model.eval()
    learned_covered = 0
    fixed_covered = 0
    total_points = 0
    volume_ratios = []
    
    device = model.device
    L_fixed = L_fixed.to(device)
    state_dim = L_fixed.shape[-1]
    
    # Pre-calculate baseline log-volume component
    log_det_V_fixed = state_dim * torch.log(torch.tensor(q_fixed, device=device)) + torch.logdet(L_fixed)

    for inputs, errors in dataloader:
        inputs, errors = inputs.to(device), errors.to(device)
        batch_size = inputs.size(0)
        
        # 1. State-Dependent Prediction [cite: 125, 136]
        L_learned = model(inputs)
        
        # Coverage check
        alpha_learned = torch.linalg.solve_triangular(L_learned, errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_learned = torch.norm(alpha_learned, p=2, dim=1)
        learned_covered += (scores_learned <= q_learned).sum().item()

        # 2. Fixed Baseline Coverage
        alpha_fixed = torch.linalg.solve_triangular(L_fixed.unsqueeze(0), errors.unsqueeze(-1), upper=False).squeeze(-1)
        scores_fixed = torch.norm(alpha_fixed, p=2, dim=1)
        fixed_covered += (scores_fixed <= q_fixed).sum().item()

        # 3. Volume Ratio Calculation (Log-space for numerical stability)
        log_det_L_learned = torch.logdet(L_learned)
        log_det_V_learned = state_dim * torch.log(torch.tensor(q_learned, device=device)) + log_det_L_learned
        
        batch_ratios = torch.exp(log_det_V_learned - log_det_V_fixed)
        volume_ratios.append(batch_ratios.cpu())
        total_points += batch_size

    all_ratios = torch.cat(volume_ratios).numpy()
    return (learned_covered / total_points), (fixed_covered / total_points), all_ratios

def plot_ratio_distribution(ratios, alpha):
    """
    Plots the log-scaled distribution of volume ratios.
    """
    plt.figure(figsize=(10, 6))
    
    # Use log10 for the x-axis as suggested
    log_ratios = np.log10(ratios)
    
    sns.histplot(log_ratios, kde=True, color='royalblue', bins=50)
    
    # Add vertical line for the mean
    plt.axvline(np.mean(log_ratios), color='red', linestyle='--', label=f'Mean: $10^{{{np.mean(log_ratios):.2f}}}$')
    
    plt.title(f"Distribution of Volume Ratios (Learned/Fixed) | $\\alpha$={alpha}")
    plt.xlabel(r"$\log_{10}(\text{Volume Ratio})$") # Added 'r' and simplified \text
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save for PACE environment
    plt.savefig("ogbench_cube/eval/volume_ratio_distribution.png", dpi=300)
    print("\nDistribution plot saved as 'volume_ratio_distribution.png'")

def plot_visualizations(ratios, cluster_metrics, alpha):
    """
    Wrapper to generate both key diagnostic plots for the L4DC submission.
    """
    # 1. Plot the Volume Ratios
    plot_ratio_distribution(ratios, alpha)
    
    # 2. Plot the Conditional Coverage (Target is 1 - Alpha, e.g., 0.90)
    plot_clustered_coverage(cluster_metrics, 1 - alpha)

if __name__ == "__main__":
    PATH = "ogbench_cube/eval/ogbench_one_step_error_data_embed_8.pt"
    ALPHA = 0.1
    lightning_dir = Path("ogbench_cube/models/err_pred_logs")

    dm = ErrorDataModule(data_path=PATH, batch_size=4096)
    dm.setup() 
    
    # model = MGNLLPredictor(
    #     input_dim=dm.input_dim, state_dim=dm.state_dim, 
    #     num_layers=3, hidden_dim=128, diagonal=False, 
    #     lr=0.00025, reg_scale=10, dropout_prob=0.3, use_spectral_norm=False
    # )
    model = MGNLLPredictor(
        input_dim=dm.input_dim, state_dim=dm.state_dim, 
        num_layers=2, hidden_dim=128, diagonal=False, 
        lr=0.00025, reg_scale=10, dropout_prob=0.3, use_spectral_norm=False
    )


    # Configure Logger
    logger = TensorBoardLogger(
        save_dir=str(lightning_dir),
        name="cube_experiment_embed_8"
    )

    # 1. Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        filename="best-error-model-{epoch:02d}"
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # 2. FIXED: Added logger and default_root_dir
    trainer = L.Trainer(
        max_epochs=6400, 
        accelerator="auto", 
        devices=1,
        logger=logger,                        # <-- Explicitly pass your logger here
        default_root_dir=lightning_dir,       # <-- Dictates backup path for checkpoint tracking
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0  
    )
    
    trainer.fit(model, datamodule=dm)

    # 3. Load the best model weights before Calibration
    print(f"\nLoading best model from {checkpoint_callback.best_model_path}...")
    model = MGNLLPredictor.load_from_checkpoint(checkpoint_callback.best_model_path)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # --- CALIBRATION ---
    print("Performing Conformal Calibration...")
    q_learned = conformal_calibration(model, dm.calib_dataloader(), alpha=ALPHA)
    L_fixed, q_fixed = calculate_fixed_baseline(dm.calib_dataloader(), alpha=ALPHA)
    
    # 1. Perform Clustered/Conditional Coverage analysis
    cluster_metrics = evaluate_clustered_coverage(model, dm.val_dataloader(), q_learned, L_fixed, q_fixed)

    # 2. Perform Global Metric analysis
    print("\nEvaluating metrics across holdout set...")
    learned_cov, fixed_cov, all_ratios = evaluate_uncertainty_metrics(
        model, dm.val_dataloader(), q_learned, L_fixed, q_fixed
    )

    # 3. Generate both plots at once
    plot_visualizations(all_ratios, cluster_metrics, ALPHA)

    # 4. Print the final summary
    print(f"\n" + "="*45)
    print(f"HOLDOUT SET METRICS (Target Coverage: {1-ALPHA:.2f})")
    print(f"Learned Model Coverage:  {learned_cov:.4f}")
    print(f"Fixed Baseline Coverage: {fixed_cov:.4f}")
    print(f"Mean Volume Ratio:       {np.mean(all_ratios):.4f}")
    print(f"Volume Ratio Std Dev:    {np.std(all_ratios):.4f}")
    print("="*45)

    # Generate the plot
    plot_ratio_distribution(all_ratios, ALPHA)
