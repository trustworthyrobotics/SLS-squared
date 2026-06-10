#!/usr/bin/env python3
"""Standalone generator for 1-step prediction errors."""

import argparse
import json
import os
import tempfile
import torch
import h5py
import re
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

if "MPLCONFIGDIR" not in os.environ or not os.access(os.environ["MPLCONFIGDIR"], os.W_OK):
    mpl_config_dir = Path(tempfile.gettempdir()) / f"matplotlib-{os.getuid()}"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(mpl_config_dir)

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Keep only external imports
from reacher.train.mlpdyn_train import LeWMReacherDataset

# --- Re-defined Constants from mlpdyn_eval to break circular import ---
DEFAULT_DATASET_PATH = "reacher/data/test_data_50hz/reacher_test.h5"
DEFAULT_ACTION_STATS_DATASET_PATH = "reacher/data/expert_data_50hz/reacher_expert.h5"
DEFAULT_MODEL_DIR = "reacher/models/mlpdyn_ft_5"

def latest_object_checkpoint(model_dir: Path) -> Path:
    pattern = re.compile(r".*_epoch_(\d+)_object\.ckpt$")
    candidates = []
    for path in model_dir.glob("*_epoch_*_object.ckpt"):
        match = pattern.match(path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {model_dir}")
    return max(candidates, key=lambda item: item[0])[1]

def load_action_normalization_stats(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    dataset = LeWMReacherDataset(
        dataset_path,
        history_size=1,
        num_preds=1,
        frameskip=1,
        img_size=224,
        action_dim=action_dim,
    )
    return dataset.action_mean.astype(np.float32), dataset.action_std.astype(np.float32)


def load_episode_standalone(dataset_path, episode_idx, args, action_mean, action_std):
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)
        pixels_np = np.asarray(h5["pixels"][rows], dtype=np.uint8)
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).float().div_(255.0)
        
        if pixels.shape[-2:] != (args.img_size, args.img_size):
            pixels = torch.nn.functional.interpolate(
                pixels, size=(args.img_size, args.img_size), mode="bilinear", align_corners=False
            )
        
        # Apply ImageNet normalization (Channel-wise for RGB)
        p_mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        p_std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        pixels = (pixels - p_mean) / p_std

        actions = np.asarray(h5["action"][rows], dtype=np.float32)
        actions = (np.nan_to_num(actions, nan=0.0) - action_mean) / action_std
        
    return pixels, torch.from_numpy(actions).float()

@torch.no_grad()
def extract_errors(model, pixels, actions, args, device):
    # Process images through the encoder
    latents = []
    for start in range(0, pixels.shape[0], args.frame_batch_size):
        chunk = pixels[start : start + args.frame_batch_size].to(device)
        output = model.encoder(chunk, interpolate_pos_encoding=True)
        emb = model.projector(output.last_hidden_state[:, 0])
        latents.append(emb)
    true_latents = torch.cat(latents, dim=0)

    rollout_steps = (true_latents.shape[0] - 1 - (args.history_size - 1) * args.frameskip) // args.frameskip
    states, acts, targets = [], [], []

    for step in range(rollout_steps):
        t_curr = step + args.history_size - 1
        curr_z = true_latents[t_curr]
        delta_z = curr_z - true_latents[t_curr - 1] if t_curr > 0 else torch.zeros_like(curr_z)
        
        # State = [z_t, delta_z_t] (Markov state)
        states.append(torch.cat((curr_z, delta_z), dim=-1))
        
        a_start = t_curr * args.frameskip
        acts.append(actions[a_start : a_start + args.frameskip].flatten())
        
        # TARGET: Full Markov State = [z_{t+1}, delta_z_{t+1}]
        next_z = true_latents[t_curr + 1]
        next_delta_z = next_z - curr_z
        targets.append(torch.cat((next_z, next_delta_z), dim=-1))

    s_tsr, a_tsr, target_tsr = torch.stack(states).to(device), torch.stack(acts).to(device), torch.stack(targets).to(device)
    
    # Dynamics prediction f(s, a)
    act_emb = model.action_encoder(a_tsr.unsqueeze(1))
    
    # Extract the full 36-dimensional state prediction
    pred_s = model.predict(s_tsr.unsqueeze(1), act_emb)[:, 0] 
    
    # Error calculated directly in the 36-dimensional space
    return {"x_t": s_tsr.cpu(), "a_t": a_tsr.cpu(), "error": (target_tsr - pred_s).cpu()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--action-stats-dataset-path", type=Path, default=DEFAULT_ACTION_STATS_DATASET_PATH)
    parser.add_argument("--out-file", type=Path, default="reacher/eval/reacher_one_step_error_data_9.pt")
    parser.add_argument("--frame-batch-size", type=int, default=128)
    
    # Add these as command-line arguments so they have safe defaults
    parser.add_argument("--history-size", type=int, default=1) 
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=1)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--action-dim", type=int, default=2)
    args = parser.parse_args()

    with open(args.model_dir / "config.json") as f:
        config = json.load(f)
    
    # Safe Config injection: only overwrite if the key exists AND isn't None
    for k in ["history_size", "num_preds", "frameskip", "img_size", "action_dim"]:
        val = config.get(k)
        if val is not None:
            setattr(args, k, val)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(latest_object_checkpoint(args.model_dir), map_location=device, weights_only=False).eval()
    action_mean, action_std = load_action_normalization_stats(args.action_stats_dataset_path, args.action_dim)
    
    with h5py.File(args.dataset_path, "r") as h5:
        ep_len = h5["ep_len"][:]
        
    valid_indices = np.flatnonzero(ep_len - 1 - (args.history_size + args.num_preds) * args.frameskip >= 0)

    all_x, all_a, all_e = [], [], []
    for idx in tqdm(valid_indices, desc="Generating Errors"):
        px, act = load_episode_standalone(args.dataset_path, idx, args, action_mean, action_std)
        data = extract_errors(model, px, act, args, device)
        all_x.append(data["x_t"]); all_a.append(data["a_t"]); all_e.append(data["error"])

    torch.save({"x_t": torch.cat(all_x), "a_t": torch.cat(all_a), "error": torch.cat(all_e)}, args.out_file)
    print(f"Saved {len(torch.cat(all_x))} transitions to {args.out_file}")


if __name__ == "__main__":
    main()
