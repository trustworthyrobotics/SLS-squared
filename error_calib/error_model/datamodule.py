import os 
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader, Subset

class ErrorDataset(Dataset):
    def __init__(self, data_dict):
        # input: [z_t, a_t], output: error [cite: 516, 616]
        self.inputs = torch.cat([data_dict["x_t"], data_dict["a_t"]], dim=-1).float()
        self.errors = data_dict["error"].float()
        self.input_dim = self.inputs.shape[-1]
        self.state_dim = self.errors.shape[-1]

    def __len__(self): return self.inputs.size(0)
    def __getitem__(self, idx): return self.inputs[idx], self.errors[idx]

class ErrorDataModule(L.LightningDataModule):
    def __init__(self, data_path: str, batch_size: int = 2048):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        
        # Performance Tuning
        self.workers = os.cpu_count() or 1
        # Only use pin_memory if CUDA is available for training
        self.pin_memory = torch.cuda.is_available()

    def setup(self, stage=None):
        full_dict = torch.load(self.data_path)
        dataset = ErrorDataset(full_dict) # Logic from previous turn
        self.input_dim, self.state_dim = dataset.input_dim, dataset.state_dim
        
        mid = len(dataset) // 2
        train_val_indices = list(range(0, mid))
        split = int(0.8 * mid)
        
        self.train_ds = Subset(dataset, train_val_indices[:split])
        self.val_ds = Subset(dataset, train_val_indices[split:])
        self.calib_ds = Subset(dataset, list(range(mid, len(dataset))))

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.workers > 0 else False
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.batch_size, 
            num_workers=self.workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.workers > 0 else False
        )

    def calib_dataloader(self):
        return DataLoader(
            self.calib_ds, 
            batch_size=self.batch_size, 
            num_workers=self.workers,
            pin_memory=self.pin_memory
        )