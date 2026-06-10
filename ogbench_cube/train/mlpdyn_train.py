#!/usr/bin/env python3
"""Train an LE-WM-style JEPA with a Markov latent-state MLP dynamics predictor."""

from __future__ import annotations

import argparse
import json
import os
from functools import partial
from pathlib import Path, PosixPath

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import lightning as pl
import numpy as np
import stable_pretraining as spt
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, Dataset, random_split

from ogbench_cube.shared.models import JEPA, MLP, MLPDynamicsPredictor, SIGReg


DEFAULT_DATASET_PATH = "ogbench_cube/data/expert_data/ogbench_cube_expert.h5"
DEFAULT_RUN_DIR = "ogbench_cube/models/mlpdyn"
FINETUNE_DIR = None
FIXED_FRAMESKIP = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--finetune-dir", type=Path, default=FINETUNE_DIR)
    parser.add_argument("--output-model-name", default="lewm")
    parser.add_argument("--seed", type=int, default=3072)
    parser.add_argument("--train-split", type=float, default=1.0)

    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=14)
    parser.add_argument("--encoder-scale", default="tiny")
    parser.add_argument("--embed-dim", type=int, default=24)
    parser.add_argument("--markov-deriv", type=int, default=1)
    parser.add_argument("--num-preds", type=int, default=5, help="Autoregressive rollout horizon.")
    parser.add_argument("--action-dim", type=int, default=5)

    parser.add_argument("--predictor-hidden-width", type=int, default=512)
    parser.add_argument("--predictor-depth", type=int, default=2)
    parser.add_argument("--predictor-dropout", type=float, default=0.0)

    parser.add_argument("--sigreg-weight", type=float, default=0.005)
    parser.add_argument("--sigreg-knots", type=int, default=17)
    parser.add_argument("--sigreg-num-proj", type=int, default=1024)
    parser.add_argument("--straighten", action="store_true", default=True, help="Apply temporal straightening to encoder latents.")
    parser.add_argument("--straighten-weight", type=float, default=1e-2)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=120)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument(
        "--persistent-workers",
        action="store_true",
        default=True,
        help="Keep training dataloader workers alive across epochs. Validation workers stay non-persistent to avoid doubled RAM usage.",
    )
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="gpu")
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--save-object-every", type=int, default=1)
    args = parser.parse_args()
    args.frameskip = FIXED_FRAMESKIP
    args.markov_state_dim = (args.markov_deriv + 1) * args.embed_dim
    return args


class LeWMOGBenchCubeDataset(Dataset):
    def __init__(
        self,
        dataset_path: Path,
        *,
        markov_deriv: int,
        num_preds: int,
        frameskip: int,
        img_size: int,
        action_dim: int,
    ) -> None:
        self.dataset_path = dataset_path
        self.markov_deriv = int(markov_deriv)
        self.num_preds = int(num_preds)
        self.frameskip = int(frameskip)
        self.num_steps = 1 + self.num_preds
        self.action_steps = self.num_preds
        self.img_size = int(img_size)
        self.action_dim = int(action_dim)
        self.effective_action_dim = self.frameskip * self.action_dim
        self._h5: h5py.File | None = None

        if self.markov_deriv < 0:
            raise ValueError("markov_deriv must be non-negative.")
        if self.num_preds < 1:
            raise ValueError("num_preds must be positive.")
        if self.frameskip < 1:
            raise ValueError("frameskip must be positive.")

        with h5py.File(self.dataset_path, "r") as h5:
            self.ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64)
            self.ep_offset = np.asarray(h5["ep_offset"][:], dtype=np.int64)
            if int(h5["action"].shape[-1]) != self.action_dim:
                raise ValueError(f"Expected action_dim={self.action_dim}, got {h5['action'].shape[-1]}.")
            finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
            finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
            self.action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
            self.action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
            self.action_std = np.maximum(self.action_std, 1e-6)

        self.samples: list[tuple[int, int]] = []
        required_last_frame_offset = (self.num_steps - 1) * self.frameskip
        required_action_end_offset = self.action_steps * self.frameskip
        required_offset = max(required_last_frame_offset, required_action_end_offset)
        for ep_idx, ep_len in enumerate(self.ep_len.tolist()):
            max_start = ep_len - 1 - required_offset
            for start in range(max_start + 1):
                self.samples.append((ep_idx, start))
        if not self.samples:
            raise ValueError("No valid training windows found. Check frameskip/markov_deriv/num_preds.")

    def __len__(self) -> int:
        return len(self.samples)

    def _file(self) -> h5py.File:
        if self._h5 is None:
            self._h5 = h5py.File(self.dataset_path, "r")
        return self._h5

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        h5 = self._file()
        ep_idx, start = self.samples[index]
        base = int(self.ep_offset[ep_idx]) + start

        frame_offsets = np.arange(self.num_steps, dtype=np.int64) * self.frameskip
        pixel_rows = base + frame_offsets
        pixels_np = np.asarray(h5["pixels"][pixel_rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
        if self.markov_deriv > 0:
            prev_offsets = np.arange(self.markov_deriv, 0, -1, dtype=np.int64) * self.frameskip
            prev_rows = int(self.ep_offset[ep_idx]) + np.maximum(start - prev_offsets, 0)
            prev_pixels_np = np.stack([np.asarray(h5["pixels"][int(row)], dtype=np.uint8) for row in prev_rows], axis=0)
            prev_pixels = torch.from_numpy(prev_pixels_np).permute(0, 3, 1, 2).contiguous()
        else:
            prev_pixels = pixels[:0].clone()

        action_blocks = []
        for step in range(self.action_steps):
            action_start = base + step * self.frameskip
            action_stop = action_start + self.frameskip
            block = np.asarray(h5["action"][action_start:action_stop], dtype=np.float32)
            block = (np.nan_to_num(block, nan=0.0) - self.action_mean) / self.action_std
            action_blocks.append(torch.from_numpy(block.reshape(-1)))

        return {
            "pixels": pixels,
            "prev_pixels": prev_pixels,
            "action": torch.stack(action_blocks, dim=0).float(),
        }

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self) -> None:
        h5 = getattr(self, "_h5", None)
        if h5 is not None:
            try:
                h5.close()
            except Exception:
                pass
            self._h5 = None


class ModelObjectCallback(Callback):
    def __init__(self, dirpath: Path, filename: str, epoch_interval: int = 1) -> None:
        super().__init__()
        self.dirpath = dirpath
        self.filename = filename
        self.epoch_interval = int(epoch_interval)

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        epoch = trainer.current_epoch + 1
        if not trainer.is_global_zero:
            return
        if epoch % self.epoch_interval != 0 and epoch != trainer.max_epochs:
            return
        self.dirpath.mkdir(parents=True, exist_ok=True)
        torch.save(pl_module.model, self.dirpath / f"{self.filename}_epoch_{epoch}_object.ckpt")


def temporal_straightening_loss(emb: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    if emb.shape[1] < 3:
        return emb.new_zeros(())
    vel_prev = emb[:, 1:-1] - emb[:, :-2]
    vel_next = emb[:, 2:] - emb[:, 1:-1]
    cosine = F.cosine_similarity(vel_prev, vel_next, dim=-1, eps=eps)
    return (1.0 - cosine).mean()


def required_markov_history(markov_deriv: int) -> int:
    if markov_deriv < 0:
        raise ValueError("markov_deriv must be non-negative.")
    return markov_deriv + 1


def build_markov_state(history_emb: torch.Tensor, markov_deriv: int) -> torch.Tensor:
    squeeze = False
    if history_emb.ndim == 2:
        history_emb = history_emb.unsqueeze(0)
        squeeze = True
    context_len = required_markov_history(markov_deriv)
    if history_emb.ndim != 3 or history_emb.shape[1] < context_len:
        raise ValueError(
            f"Expected history_emb with shape [batch, >= {context_len}, dim], got {tuple(history_emb.shape)}."
        )
    deriv_seq = history_emb[:, -context_len:]
    components = [deriv_seq[:, -1]]
    for _ in range(markov_deriv):
        deriv_seq = deriv_seq[:, 1:] - deriv_seq[:, :-1]
        components.append(deriv_seq[:, -1])
    state = torch.cat(components, dim=-1)
    return state[0] if squeeze else state


def preprocess_pixels(pixels: torch.Tensor, img_size: int) -> torch.Tensor:
    pixels = pixels.float().div_(255.0)
    if pixels.shape[-2:] != (img_size, img_size):
        batch_size, time_steps = pixels.shape[:2]
        pixels = F.interpolate(
            pixels.view(batch_size * time_steps, *pixels.shape[2:]),
            size=(img_size, img_size),
            mode="bilinear",
            align_corners=False,
        ).view(batch_size, time_steps, *pixels.shape[2:3], img_size, img_size)
    pixel_mean = pixels.new_tensor([0.485, 0.456, 0.406]).view(1, 1, 3, 1, 1)
    pixel_std = pixels.new_tensor([0.229, 0.224, 0.225]).view(1, 1, 3, 1, 1)
    return (pixels - pixel_mean) / pixel_std


def lewm_forward(self, batch: dict[str, torch.Tensor], stage: str, args: argparse.Namespace):
    n_preds = args.num_preds
    lambd = args.sigreg_weight
    embed_dim = args.embed_dim
    markov_context_len = required_markov_history(args.markov_deriv)

    all_pixels = torch.cat((batch["prev_pixels"], batch["pixels"]), dim=1)
    all_pixels = preprocess_pixels(all_pixels, args.img_size)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = self.model.encode({"pixels": all_pixels, "action": batch["action"]})
    history_emb = output["emb"][:, :markov_context_len]
    emb = output["emb"][:, args.markov_deriv :]
    output["emb"] = emb
    act_emb = output["act_emb"]
    rollout_state = build_markov_state(history_emb, args.markov_deriv)
    pred_losses = []
    pred_embs = []
    for step in range(n_preds):
        ctx_state = rollout_state.unsqueeze(1)
        ctx_act = act_emb[:, step : step + 1]

        pred_state = self.model.predict(ctx_state, ctx_act)[:, 0]
        pred_next = pred_state[..., :embed_dim]
        tgt_next = emb[:, step + 1]
        if args.markov_deriv > 0:
            tgt_history = torch.cat((emb[:, step + 1 - args.markov_deriv : step + 1], tgt_next.unsqueeze(1)), dim=1)
        else:
            tgt_history = tgt_next.unsqueeze(1)
        tgt_state = build_markov_state(tgt_history, args.markov_deriv)
        pred_losses.append((pred_state - tgt_state).pow(2).mean())
        pred_embs.append(pred_next)
        rollout_state = pred_state

    output["pred_emb"] = torch.stack(pred_embs, dim=1)
    output["pred_loss"] = torch.stack(pred_losses).mean()
    output["sigreg_loss"] = self.sigreg(emb.transpose(0, 1))
    output["straighten_loss"] = temporal_straightening_loss(emb) if args.straighten else emb.new_zeros(())
    output["loss"] = (
        output["pred_loss"]
        + lambd * output["sigreg_loss"]
        + args.straighten_weight * output["straighten_loss"]
    )

    log_prefix = "train" if stage == "fit" else stage
    losses = {f"{log_prefix}/{key}": value.detach() for key, value in output.items() if "loss" in key}
    self.log_dict(losses, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
    return output


def build_model(args: argparse.Namespace) -> JEPA:
    encoder = spt.backbone.utils.vit_hf(
        args.encoder_scale,
        patch_size=args.patch_size,
        image_size=args.img_size,
        pretrained=False,
        use_mask_token=False,
    )
    hidden_dim = encoder.config.hidden_size
    embed_dim = args.embed_dim
    markov_state_dim = args.markov_state_dim
    effective_act_dim = args.frameskip * args.action_dim

    predictor = MLPDynamicsPredictor(
        embed_dim=markov_state_dim,
        action_dim=effective_act_dim,
        history_size=1,
        action_history_size=1,
        num_preds=1,
        hidden_width=args.predictor_hidden_width,
        depth=args.predictor_depth,
        dropout=args.predictor_dropout,
    )
    projector = MLP(input_dim=hidden_dim, output_dim=embed_dim, hidden_dim=2048, norm_fn=torch.nn.BatchNorm1d)
    return JEPA(
        encoder=encoder,
        predictor=predictor,
        action_encoder=torch.nn.Identity(),
        projector=projector,
        pred_proj=torch.nn.Identity(),
    )


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    *,
    shuffle: bool,
    drop_last: bool,
    persistent_workers: bool,
) -> DataLoader:
    loader_kwargs = {
        "batch_size": args.batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": args.num_workers,
        "pin_memory": args.accelerator == "gpu",
        "persistent_workers": args.num_workers > 0 and persistent_workers,
    }
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = args.prefetch_factor
    return DataLoader(dataset, **loader_kwargs)


def sanitize_hparams(args: argparse.Namespace) -> dict[str, object]:
    hparams = vars(args).copy()
    for key, value in hparams.items():
        if isinstance(value, Path):
            hparams[key] = str(value)
    return hparams


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out-file", type=Path, default="lewm_one_step_error_data_ogbench.pt")
    parser.add_argument("--frame-batch-size", type=int, default=128)
    args = parser.parse_args()

    with open(args.model_dir / "config.json") as f:
        config = json.load(f)
    
    # --- FIXED: Config injection with fallback defaults ---
    defaults = {
        "markov_deriv": 1,
        "num_preds": 1,
        "frameskip": 1,
        "img_size": 224,
        "action_dim": 5,
    }
    for k, fallback in defaults.items():
        val = config.get(k)
        setattr(args, k, val if val is not None else fallback)
    # ------------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(latest_object_checkpoint(args.model_dir), map_location=device, weights_only=False).eval()
    
    with h5py.File(args.dataset_path, "r") as h5:
        ep_len = h5["ep_len"][:]
        
    history_len = required_markov_history(args.markov_deriv)
    valid_indices = np.flatnonzero(ep_len - 1 - (history_len - 1 + args.num_preds) * args.frameskip >= 0)

    all_x, all_a, all_e = [], [], []
    for idx in tqdm(valid_indices, desc="Generating Errors"):
        px, act = load_episode_standalone(args.dataset_path, idx, args)
        data = extract_errors(model, px, act, args, device)
        if data is not None:
            all_x.append(data["x_t"])
            all_a.append(data["a_t"])
            all_e.append(data["error"])

    torch.save({"x_t": torch.cat(all_x), "a_t": torch.cat(all_a), "error": torch.cat(all_e)}, args.out_file)
    print(f"Saved {len(torch.cat(all_x))} transitions to {args.out_file}")

if __name__ == "__main__":
    main()
