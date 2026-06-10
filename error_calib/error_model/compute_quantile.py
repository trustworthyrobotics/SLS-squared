import torch
import numpy as np
import json
import math
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Subset
from datamodule import ErrorDataset
from error_model import MGNLLPredictor
from conformal_prediction import conformal_calibration

@torch.no_grad()
def plot_sigma_eigenvalues(model, dataloader, save_path="sigma_eigenvalues.png"):
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
    breakpoint()
    
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
    plot_sigma_eigenvalues(model, calib_loader, save_path="sigma_eigenvalues.png")

    # 4. Compute the Quantile
    # Uses the formula: k = ceil((n + 1) * (1 - alpha))
    print(f"Computing Conformal Quantile (alpha={alpha})...")
    q_learned = conformal_calibration(model, calib_loader, alpha=alpha)
    
    print(f"\nResulting Quantile (q_learned): {q_learned:.6f}")

    # 5. Save for use in plan_sls_mpc.py
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
    DATA_PATH = "lewm_one_step_error_data_rand_8.pt"
    # Using the checkpoint filename visible in your screenshot
    CKPT_PATH = "lightning_logs/version_12/checkpoints/best-error-model-epoch=1599.ckpt"
    
    compute_and_save_quantile(DATA_PATH, CKPT_PATH)