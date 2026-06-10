import numpy as np
import torch

@torch.no_grad()
def conformal_calibration(model, dataloader, alpha=0.1):
    model.eval()
    scores = []
    for x, err in dataloader:
        L = model(x.to(model.device))
        # score s = ||L^-1 * error||_2 [cite: 123]
        s = torch.norm(torch.linalg.solve_triangular(L, err.to(model.device).unsqueeze(-1), upper=False).squeeze(-1), p=2, dim=1)
        scores.append(s.cpu())
    
    scores = torch.cat(scores).numpy()
    n = len(scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))
    
    if k > n:
        print(f"WARNING: Insufficient calibration data for alpha={alpha}. Returning max.")
        return np.max(scores) # Practical fallback requested
    
    return np.sort(scores)[k - 1]