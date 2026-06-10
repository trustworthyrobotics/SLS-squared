import torch
import numpy as np
import json
import math
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import scipy.stats as stats  # Added for KDE
from torch.utils.data import DataLoader, Subset

ERROR_MODEL_ROOT = Path(__file__).resolve().parents[2] / "error_calib" / "error_model"
if str(ERROR_MODEL_ROOT) not in sys.path:
    sys.path.insert(0, str(ERROR_MODEL_ROOT))

from datamodule import ErrorDataset
from error_model import MGNLLPredictor
from conformal_prediction import conformal_calibration

@torch.no_grad()
def plot_sigma_eigenvalues(model, dataloader, save_path="rope/eval/sigma_eigenvalues.png"):
    """
    Computes the eigenvalues of Sigma = L * L^T for each sample
    and plots a histogram of the eigenvalue distributions.
    """
    print("\nComputing eigenvalues of Sigma = L * L^T...")
    model.eval()
    all_eigvals = []
    
    for x, _ in dataloader:
        # Get the Cholesky factor L from the model
        L = model(x.to(model.device))
        
        # Compute the covariance matrix Sigma = L @ L^T
        Sigma = torch.bmm(L, L.transpose(1, 2))
        
        # Compute eigenvalues (eigvalsh is optimized for symmetric matrices)
        # Returns eigenvalues in ascending order
        eigvals = torch.linalg.eigvalsh(Sigma)
        all_eigvals.append(eigvals.cpu())
        
    all_eigvals = torch.cat(all_eigvals, dim=0).numpy()
    state_dim = all_eigvals.shape[1]
    
    print(f"Plotting histograms for {state_dim} dimensions...")
    
    # Calculate grid size for the subplots
    cols = 6
    rows = math.ceil(state_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()
    
    for i in range(state_dim):
        ax = axes[i]
        # Plot the distribution of the i-th sorted eigenvalue
        ax.hist(all_eigvals[:, i], bins=50, color='royalblue', alpha=0.7)
        ax.set_title(f"Eigval {i+1} (Ascending)")
        ax.tick_params(axis='both', which='major', labelsize=8)
        # Log scale is typically required as eigenvalues can span multiple orders of magnitude
        ax.set_yscale('log') 
        
    # Hide any unused subplots in the grid
    for j in range(state_dim, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved eigenvalue histograms to '{save_path}'\n")

@torch.no_grad()
def plot_state_pdfs(dataset, start_idx, end_idx, save_path="rope/eval/state_pdfs_kde.png"):
    """
    Extracts the targets (y) from the dataset over a specific index range 
    (e.g., calib + test) and plots their KDE smoothed PDFs in a 6x6 grid.
    """
    print(f"\nExtracting data from index {start_idx} to {end_idx} for PDF plotting...")
    
    # Create subset for calib + test
    subset_indices = list(range(start_idx, end_idx))
    subset = Subset(dataset, subset_indices)
    
    # Use a dataloader to quickly iterate through and collect data
    loader = DataLoader(subset, batch_size=4096, num_workers=4, pin_memory=torch.cuda.is_available())
    
    all_targets = []
    for _, y in loader:
        all_targets.append(y.cpu())
        
    all_targets = torch.cat(all_targets, dim=0).numpy()
    state_dim = all_targets.shape[1]
    
    print(f"Plotting KDE PDFs for {state_dim} state dimensions...")
    
    # Calculate grid size for the subplots (defaults to 6x6 for 36 dims)
    cols = 6
    rows = math.ceil(state_dim / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 2.5 * rows))
    axes = axes.flatten()
    
    for i in range(state_dim):
        ax = axes[i]
        data_dim = all_targets[:, i]
        
        # Calculate KDE
        try:
            # Gaussian KDE smoothing
            kde = stats.gaussian_kde(data_dim)
            # Create a smooth x-axis grid spanning the data range + 10% padding
            x_min, x_max = data_dim.min(), data_dim.max()
            padding = 0.1 * (x_max - x_min) if (x_max - x_min) > 0 else 0.1
            x_grid = np.linspace(x_min - padding, x_max + padding, 200)
            
            # Plot KDE line
            ax.plot(x_grid, kde(x_grid), color='darkorange', lw=2, label="KDE")
            
            # Plot a light histogram underneath for density visualization
            ax.hist(data_dim, bins=50, density=True, color='gray', alpha=0.3)
        except Exception as e:
            # Fallback if KDE fails (e.g., variance is exactly zero)
            print(f"  [Warning] KDE failed for dim {i}: {e}. Plotting histogram only.")
            ax.hist(data_dim, bins=50, density=True, color='gray', alpha=0.5)
            
        ax.set_title(f"State Dim {i+1}")
        ax.tick_params(axis='both', which='major', labelsize=8)
        
    # Hide any unused subplots
    for j in range(state_dim, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"Saved state PDF plots to '{save_path}'\n")


def compute_and_save_quantile(
    data_path: str, 
    checkpoint_path: str, 
    alpha: float = 0.1, 
    batch_size: int = 4096
):
    # 1. Load the trained MGNLL model
    print(f"Loading model from {checkpoint_path}...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MGNLLPredictor.load_from_checkpoint(checkpoint_path)
    model.to(device)
    model.eval()

    # 2. Load dataset directly to control indexing
    full_dict = torch.load(data_path)
    dataset = ErrorDataset(full_dict)
    n_total = len(dataset)
    
    # Define ranges:
    # 0% - 50%: Training/Val (Already used)
    # 50% - 90%: Calibration (Target)
    # 90% - 100%: Planning/Test (Reserved)
    calib_start = int(0.5 * n_total)
    calib_end = int(0.9 * n_total)
    
    calib_indices = list(range(calib_start, calib_end))
    calib_ds = Subset(dataset, calib_indices)
    
    calib_loader = DataLoader(
        calib_ds, 
        batch_size=batch_size, 
        num_workers=4, 
        pin_memory=torch.cuda.is_available()
    )
    
    print(f"Total Transitions: {n_total}")
    print(f"Calibration Range: {calib_start} to {calib_end} ({len(calib_indices)} samples)")
    print(f"Planning Range:    {calib_end} to {n_total} ({n_total - calib_end} samples)")

    # 3. Compute and Plot Sigma Eigenvalues
    plot_sigma_eigenvalues(model, calib_loader, save_path="rope/eval/sigma_eigenvalues.png")

    # 4. Plot KDE distributions for the target state/error (Calib + Test)
    # We pass n_total as the end_idx to include the 90%-100% planning/test data.
    plot_state_pdfs(dataset, start_idx=calib_start, end_idx=n_total, save_path="rope/eval/state_pdfs_kde.png")

    # 5. Compute the Quantile
    # Uses the formula: k = ceil((n + 1) * (1 - alpha))
    print(f"Computing Conformal Quantile (alpha={alpha})...")
    q_learned = conformal_calibration(model, calib_loader, alpha=alpha)
    
    print(f"\nResulting Quantile (q_learned): {q_learned:.6f}")

    # 6. Save for use in plan_sls_mpc.py
    results = {
        "q_learned": float(q_learned),
        "alpha": alpha,
        "checkpoint": str(checkpoint_path),
        "calib_range": [calib_start, calib_end]
    }
    
    with open("conformal_config.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved to 'conformal_config.json'")

if __name__ == "__main__":
    # Update these paths to match your PACE or local directory structure
    DATA_PATH = "rope/eval/rope_one_step_error_data.pt"  # Path to the dataset generated by generate_error.py
    # Using the checkpoint filename visible in your screenshot
    CKPT_PATH = "/home/user/latent-brs/rope/models/err_pred_logs/rope_experiment/version_7/checkpoints/best-error-model-epoch=3781.ckpt"
    
    compute_and_save_quantile(DATA_PATH, CKPT_PATH)
