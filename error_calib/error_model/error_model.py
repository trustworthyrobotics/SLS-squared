import torch
import torch.nn as nn
import lightning as L
from torch.nn.utils import spectral_norm

class MGNLLPredictor(L.LightningModule):
    def __init__(
        self, 
        input_dim: int, 
        state_dim: int, 
        num_layers: int = 3,
        hidden_dim: int = 512, 
        dropout_prob: float = 0.3,
        use_spectral_norm: bool = True,  # New boolean flag
        diagonal: bool = False, 
        lr: float = 1e-5,
        reg_scale: float = 1.0 
    ):
        super().__init__()
        self.save_hyperparameters()
        self.state_dim = state_dim
        self.diagonal = diagonal
        
        self.out_dim = state_dim if diagonal else (state_dim * (state_dim + 1)) // 2

        layers = []
        curr_in = input_dim
        
        for i in range(num_layers):
            is_last = (i == num_layers - 1)
            out_features = self.out_dim if is_last else hidden_dim
            
            lin = nn.Linear(curr_in, out_features)
            
            if self.hparams.use_spectral_norm:
                # Apply spectral norm (sets spectral radius to 1)
                lin = spectral_norm(lin)
                # Manual weight scaling to achieve the target reg_scale (Lipschitz bound)
                with torch.no_grad():
                    # Note: spectral_norm stores weights in weight_orig
                    lin.weight_orig.mul_(reg_scale)
            
            layers.append(lin)
            
            if not is_last:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(p=self.hparams.dropout_prob))
            
            curr_in = hidden_dim

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        raw = self.net(x)
        batch_size = x.shape[0]
        
        if self.diagonal:
            # Diagonals use exp to ensure positive variance. Added 1e-4 floor.
            return torch.diag_embed(torch.exp(raw) + 1e-4) 
        
        L = torch.zeros((batch_size, self.state_dim, self.state_dim), device=self.device)
        tril_indices = torch.tril_indices(self.state_dim, self.state_dim)
        L[:, tril_indices[0], tril_indices[1]] = raw
        
        # Diagonal floor to prevent numerical collapse during training
        diag_idx = torch.arange(self.state_dim)
        L[:, diag_idx, diag_idx] = torch.exp(L[:, diag_idx, diag_idx]) + 1e-4
        return L

    def _mgnll_loss(self, L, error):
        """
        Multivariate Gaussian Negative Log-Likelihood loss:
        L = 0.5 * (epsilon^T * Sigma^-1 * epsilon + ln(det(Sigma)))
        """
        # Quadratic term: ||L^-1 * e||^2
        dist = torch.linalg.solve_triangular(L, error.unsqueeze(-1), upper=False)
        quad_term = torch.sum(dist**2, dim=(1, 2))
        
        # Log-determinant: 2 * sum(ln(L_ii))
        log_det = 2 * torch.sum(torch.log(torch.diagonal(L, dim1=-2, dim2=-1)), dim=-1)
        
        return 0.5 * torch.mean(quad_term + log_det)

    def training_step(self, batch, batch_idx):
        inputs, errors = batch
        L_factor = self(inputs)
        loss = self._mgnll_loss(L_factor, errors)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        inputs, errors = batch
        L_factor = self(inputs)
        loss = self._mgnll_loss(L_factor, errors)
        self.log("val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=50, T_mult=1, eta_min=1e-7
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch", 
                "frequency": 1
            }
        }

    # def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        
        # # 1. First phase: Cosine Annealing (Epochs 0 to 1999)
        # scheduler1 = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        #     optimizer, T_0=50, T_mult=1, eta_min=1e-7
        # )
        
        # # 2. Second phase: Custom Exponential Decay starting at 1e-5 (Epoch 2000+)
        # # LambdaLR multiplies the *original* optimizer LR (self.hparams.lr) by its output.
        # # We compute a multiplier that forces the LR to 1e-5, then apply exponential decay.
        # target_start_lr = 2e-6
        # gamma = 0.99
        
        # def lr_lambda(epoch):
        #     # 'epoch' here resets to 0 when SequentialLR switches to this scheduler
        #     # We need: original_lr * multiplier = target_start_lr * (gamma ** epoch)
        #     # So: multiplier = (target_start_lr / original_lr) * (gamma ** epoch)
        #     multiplier = (target_start_lr / self.hparams.lr) * (gamma ** epoch)
        #     return multiplier

        # scheduler2 = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        # # 3. Sequence them together
        # main_scheduler = torch.optim.lr_scheduler.SequentialLR(
        #     optimizer, 
        #     schedulers=[scheduler1, scheduler2], 
        #     milestones=[3200]
        # )
        
        # return {
        #     "optimizer": optimizer,
        #     "lr_scheduler": {
        #         "scheduler": main_scheduler,
        #         "interval": "epoch", 
        #         "frequency": 1
        #     }
        # }