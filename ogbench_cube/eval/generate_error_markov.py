#!/usr/bin/env python3
"""Standalone generator for 1-step prediction errors for OGBench MLP predictor."""

import argparse
import json
import os
import torch
import h5py
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])

import gymnasium
import mujoco
import ogbench.manipspace  # noqa: F401
from ogbench.manipspace import lie

# Keep only external imports for OGBench
from ogbench_cube.train.mlpdyn_train import (
    build_markov_state,
    preprocess_pixels,
    required_markov_history,
)

# --- Re-defined Constants from mlpdyn_eval to break circular import ---
DEFAULT_DATASET_PATH = "ogbench_cube/data/test_data/ogbench_cube_test.h5"
DEFAULT_EXPERT_DATASET_PATH = "ogbench_cube/data/expert_data/ogbench_cube_expert.h5"
DEFAULT_MODEL_DIR = "ogbench_cube/models/mlpdyn_embd_8"


def load_action_normalization_stats(dataset_path: Path, action_dim: int) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(dataset_path, "r") as h5:
        if int(h5["action"].shape[-1]) != action_dim:
            raise ValueError(f"Expected action_dim={action_dim}, got {h5['action'].shape[-1]}.")
        finite_actions = np.asarray(h5["action"][:], dtype=np.float32)
        finite_actions = finite_actions[~np.isnan(finite_actions).any(axis=1)]
        action_mean = finite_actions.mean(axis=0, keepdims=True).astype(np.float32)
        action_std = finite_actions.std(axis=0, keepdims=True).astype(np.float32)
        action_std = np.maximum(action_std, 1e-6)
    return action_mean, action_std

def hide_target_cube(env: gymnasium.Env) -> None:
    for geom_ids in env.unwrapped._cube_target_geom_ids_list:
        for gid in geom_ids:
            env.unwrapped._model.geom(gid).rgba[3] = 0.0

def restore_target_pose(env: gymnasium.Env, target_block_pos: np.ndarray, target_block_yaw: float) -> None:
    unwrapped = env.unwrapped
    unwrapped._target_block = 0
    target_mocap_id = unwrapped._cube_target_mocap_ids[0]
    unwrapped._data.mocap_pos[target_mocap_id] = np.asarray(target_block_pos, dtype=np.float64)
    unwrapped._data.mocap_quat[target_mocap_id] = np.asarray(
        lie.SO3.from_z_radians(float(target_block_yaw)).wxyz,
        dtype=np.float64,
    )
    hide_target_cube(env)

def render_episode_without_target_cube(
    env: gymnasium.Env,
    *,
    seed: int,
    qpos: np.ndarray,
    qvel: np.ndarray,
    target_block_pos: np.ndarray,
    target_block_yaw: np.ndarray,
    camera: str,
) -> np.ndarray:
    env.reset(seed=seed)
    frames = []
    for step in range(qpos.shape[0]):
        env.unwrapped._data.qpos[: qpos.shape[1]] = np.asarray(qpos[step], dtype=np.float64)
        env.unwrapped._data.qvel[: qvel.shape[1]] = np.asarray(qvel[step], dtype=np.float64)
        restore_target_pose(
            env,
            target_block_pos=target_block_pos[step],
            target_block_yaw=float(np.asarray(target_block_yaw[step]).reshape(-1)[0]),
        )
        env.unwrapped.pre_step()
        mujoco.mj_forward(env.unwrapped._model, env.unwrapped._data)
        env.unwrapped.post_step()
        hide_target_cube(env)
        frames.append(np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8))
    return np.stack(frames, axis=0)

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

def load_episode_standalone(dataset_path, episode_idx, args, action_mean, action_std, env, camera):
    with h5py.File(dataset_path, "r") as h5:
        ep_len = int(h5["ep_len"][episode_idx])
        ep_offset = int(h5["ep_offset"][episode_idx])
        rows = np.arange(ep_offset, ep_offset + ep_len, dtype=np.int64)

        pixels_np = render_episode_without_target_cube(
            env,
            seed=int(h5["episode_seed"][episode_idx]) if "episode_seed" in h5 else 0,
            qpos=np.asarray(h5["qpos"][rows], dtype=np.float32),
            qvel=np.asarray(h5["qvel"][rows], dtype=np.float32),
            target_block_pos=np.asarray(h5["target_block_pos"][rows], dtype=np.float32),
            target_block_yaw=np.asarray(h5["target_block_yaw"][rows], dtype=np.float32),
            camera=camera,
        )
        pixels = torch.from_numpy(pixels_np).permute(0, 3, 1, 2).contiguous()
        pixels = preprocess_pixels(pixels.unsqueeze(0), args.img_size)[0]

        actions = np.asarray(h5["action"][rows], dtype=np.float32)
        actions = (np.nan_to_num(actions, nan=0.0) - action_mean) / action_std
        actions = torch.from_numpy(actions).float()
        
    return pixels, actions

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

    history_len = required_markov_history(args.markov_deriv)
    rollout_steps = (true_latents.shape[0] - 1 - (history_len - 1) * args.frameskip) // args.frameskip
    
    if rollout_steps < 1:
        return None

    states, acts, targets = [], [], []

    for step in range(rollout_steps):
        t_curr = step * args.frameskip + (history_len - 1) * args.frameskip
        
        # Current State: build markov state dynamically using native OGBench logic
        hist_indices = [t_curr - i * args.frameskip for i in range(history_len - 1, -1, -1)]
        hist_z = true_latents[hist_indices] 
        curr_state = build_markov_state(hist_z.unsqueeze(0), args.markov_deriv)[0]
        states.append(curr_state)
        
        a_start = t_curr
        acts.append(actions[a_start : a_start + args.frameskip].flatten())
        
        # Target: Full Markov State
        t_next = t_curr + args.frameskip
        next_hist_indices = [t_next - i * args.frameskip for i in range(history_len - 1, -1, -1)]
        next_hist_z = true_latents[next_hist_indices]
        next_state = build_markov_state(next_hist_z.unsqueeze(0), args.markov_deriv)[0]
        targets.append(next_state)

    s_tsr = torch.stack(states).to(device)
    a_tsr = torch.stack(acts).to(device)
    target_tsr = torch.stack(targets).to(device)
    
    # Dynamics prediction f(s, a)
    act_emb = model.action_encoder(a_tsr.unsqueeze(1))
    
    # Extract the full dimensional state prediction
    pred_s = model.predict(s_tsr.unsqueeze(1), act_emb)[:, 0] 
    
    # Error calculated directly in the n-dimensional derivative space
    return {"x_t": s_tsr.cpu(), "a_t": a_tsr.cpu(), "error": (target_tsr - pred_s).cpu()}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--expert-dataset-path", type=Path, default=DEFAULT_EXPERT_DATASET_PATH)
    parser.add_argument("--out-file", type=Path, default="ogbench_cube/eval/ogbench_one_step_error_data_embed_8.pt")
    parser.add_argument("--frame-batch-size", type=int, default=128)
    args = parser.parse_args()

    with open(args.model_dir / "config.json") as f:
        config = json.load(f)
    
    # --- FIXED: Config injection with robust fallbacks ---
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
    # -----------------------------------------------------

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = torch.load(latest_object_checkpoint(args.model_dir), map_location=device, weights_only=False).eval()
    
    with h5py.File(args.dataset_path, "r") as h5:
        ep_len = h5["ep_len"][:]
        env_name = str(h5.attrs.get("env_name", "cube-single-v0"))
        camera = str(h5.attrs.get("camera", "front_pixels"))
        width = int(h5["pixels"].shape[2])
        height = int(h5["pixels"].shape[1])
        
    history_len = required_markov_history(args.markov_deriv)
    valid_indices = np.flatnonzero(ep_len - 1 - (history_len - 1 + args.num_preds) * args.frameskip >= 0)

    action_mean, action_std = load_action_normalization_stats(args.expert_dataset_path, args.action_dim)
    env = gymnasium.make(env_name, terminate_at_goal=False, mode="data_collection", width=width, height=height)

    all_x, all_a, all_e = [], [], []
    try:
        for idx in tqdm(valid_indices, desc="Generating Errors"):
            px, act = load_episode_standalone(
                args.dataset_path,
                idx,
                args,
                action_mean,
                action_std,
                env,
                camera,
            )
            data = extract_errors(model, px, act, args, device)
            if data is not None:
                all_x.append(data["x_t"])
                all_a.append(data["a_t"])
                all_e.append(data["error"])
    finally:
        env.close()

    torch.save({"x_t": torch.cat(all_x), "a_t": torch.cat(all_a), "error": torch.cat(all_e)}, args.out_file)
    print(f"Saved {len(torch.cat(all_x))} transitions to {args.out_file}")

if __name__ == "__main__":
    main()
