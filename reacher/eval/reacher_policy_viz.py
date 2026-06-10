#!/usr/bin/env python3
"""Render rollouts from the DM Control Reacher SAC policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import imageio.v2 as imageio
import numpy as np
import torch as th
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from tqdm.auto import tqdm

from reacher.train.reacher_policy_train import DmControlGymEnv

DEFAULT_OUTPUT_DIR = "reacher/models/reacher-dm-control-sac"
DEFAULT_MODEL_PATH = "reacher/models/reacher-dm-control-sac/best_model/best_model.zip"
DEFAULT_VECNORMALIZE_PATH = "reacher/models/reacher-dm-control-sac/vecnormalize.pkl"
DEFAULT_OUTDIR = "reacher/eval/reacher_videos"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to the trained SAC zip file.",
    )
    parser.add_argument(
        "--vecnormalize-path",
        type=Path,
        default=DEFAULT_VECNORMALIZE_PATH,
        help="Path to VecNormalize statistics from training.",
    )
    parser.add_argument(
        "--task",
        choices=("easy", "hard"),
        default="hard",
        help="DM Control task variant used during training.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=DEFAULT_OUTDIR,
        help="Directory where MP4s will be written.",
    )
    parser.add_argument(
        "--videos",
        type=int,
        default=20,
        help="Number of episodes to render.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for evaluation episodes.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=500,
        help="Maximum control steps per episode.",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="Episode time limit in seconds.",
    )
    parser.add_argument(
        "--action-cost-weight",
        type=float,
        default=None,
        help=(
            "Quadratic action penalty used by the environment. Defaults to the "
            "value in train_config.json when available, otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--action-rate-cost-weight",
        type=float,
        default=None,
        help=(
            "Quadratic action-change penalty used by the environment. Defaults to the "
            "value in train_config.json when available, otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--velocity-cost-weight",
        type=float,
        default=None,
        help=(
            "Quadratic velocity penalty used by the environment. Defaults to the "
            "value in train_config.json when available, otherwise 0.0."
        ),
    )
    parser.add_argument(
        "--width",
        type=int,
        default=224,
        help="Render width in pixels.",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=224,
        help="Render height in pixels.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=50,
        help="Output video FPS. DM Control reacher runs at 50 Hz by default.",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=8,
        help="ImageIO/ffmpeg quality for MP4 output.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Torch device for policy inference.",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using deterministic policy output.",
    )
    parser.add_argument(
        "--control-noise",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add Gaussian noise to policy actions before stepping the environment.",
    )
    parser.add_argument(
        "--control-noise-std",
        type=float,
        default=2.0,
        help="Per-dimension standard deviation for Gaussian action noise.",
    )
    return parser.parse_args()


def require_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if th.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not th.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available.")
    return device_arg


def load_train_config(model_path: Path, vecnormalize_path: Path) -> dict[str, Any]:
    config_candidates = [
        model_path.parent.parent / "train_config.json",
        model_path.parent / "train_config.json",
        vecnormalize_path.parent / "train_config.json",
    ]
    for config_path in config_candidates:
        if not config_path.exists():
            continue
        with config_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    return {}


def resolve_vecnormalize_path(requested_path: Path, model_path: Path) -> Path:
    if requested_path.exists():
        return requested_path

    run_dir = model_path.parent.parent if model_path.parent.name == "best_model" else model_path.parent
    candidates = [
        run_dir / "vecnormalize.pkl",
        run_dir / "eval_vecnormalize.pkl",
    ]
    candidates.extend(sorted((run_dir / "checkpoints").glob("*vecnormalize*.pkl")))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        raise FileNotFoundError(
            f"VecNormalize stats not found: {requested_path}. "
            "Pass --vecnormalize-path for the run that produced this model."
        )

    best_path = max(existing, key=lambda path: path.stat().st_mtime)
    print(f"VecNormalize stats not found at {requested_path}; using {best_path}")
    return best_path


def make_eval_env(
    task: str,
    seed: int,
    time_limit: float,
    vecnormalize_path: Path,
    action_cost_weight: float = 0.0,
    action_rate_cost_weight: float = 0.0,
    velocity_cost_weight: float = 0.0,
) -> VecNormalize:
    def _factory() -> Monitor:
        env = DmControlGymEnv(
            domain_name="reacher",
            task_name=task,
            seed=seed,
            time_limit=time_limit,
            action_cost_weight=action_cost_weight,
            action_rate_cost_weight=action_rate_cost_weight,
            velocity_cost_weight=velocity_cost_weight,
        )
        env.reset(seed=seed)
        return Monitor(env)

    base_env = DummyVecEnv([_factory])
    vec_env = VecNormalize.load(str(vecnormalize_path), base_env)
    vec_env.training = False
    vec_env.norm_reward = False
    return vec_env


def get_render_env(vec_env: VecNormalize) -> DmControlGymEnv:
    monitor_env = vec_env.venv.envs[0]
    raw_env = getattr(monitor_env, "env", None)
    if raw_env is None or not isinstance(raw_env, DmControlGymEnv):
        raise TypeError("Expected VecNormalize -> DummyVecEnv -> Monitor -> DmControlGymEnv")
    return raw_env


def configure_offscreen_framebuffer(render_env: DmControlGymEnv, width: int, height: int) -> None:
    """Expand the MuJoCo offscreen framebuffer to fit the requested dimensions.

    Must be called after every env reset (which rebuilds physics) and before
    the first physics.render() call (which lazily creates the GL context).
    """
    global_ = render_env._env.physics.model.vis.global_
    global_.offheight = max(height, int(global_.offheight))
    global_.offwidth = max(width, int(global_.offwidth))


def clip_action_to_space(action: np.ndarray, vec_env: VecNormalize) -> np.ndarray:
    action_space = getattr(vec_env, "action_space", None)
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    action = np.asarray(action, dtype=np.float32)
    if high.shape != action.shape or low.shape != action.shape:
        return action
    return np.clip(action, low, high).astype(np.float32)


def render_episode_frames(
    *,
    model: SAC,
    vec_env: VecNormalize,
    render_env: DmControlGymEnv,
    episode_seed: int,
    deterministic: bool,
    max_steps: int,
    width: int,
    height: int,
    control_noise: bool,
    control_noise_std: float,
) -> tuple[np.ndarray, float, int]:
    vec_env.seed(episode_seed)
    obs = vec_env.reset()
    rng = np.random.default_rng(episode_seed)
    configure_offscreen_framebuffer(render_env, width, height)
    frames = [render_env._env.physics.render(height=height, width=width, camera_id=0)]

    total_reward = 0.0
    num_steps = 0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        if control_noise:
            noise = rng.normal(loc=0.0, scale=control_noise_std, size=action.shape).astype(np.float32)
            action = clip_action_to_space(action + noise, vec_env)
        obs, rewards, dones, infos = vec_env.step(action)
        total_reward += float(rewards[0])
        num_steps += 1
        frames.append(render_env._env.physics.render(height=height, width=width, camera_id=0))
        if bool(dones[0]):
            break

    return np.stack(frames, axis=0), total_reward, num_steps


def main() -> None:
    args = parse_args()
    device = require_device(args.device)

    model_path = args.model_path.expanduser().resolve()
    vecnormalize_path = args.vecnormalize_path.expanduser().resolve()
    outdir = args.outdir.expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    vecnormalize_path = resolve_vecnormalize_path(vecnormalize_path, model_path)
    train_config = load_train_config(model_path, vecnormalize_path)
    action_cost_weight = (
        float(train_config.get("action_cost_weight", 0.0))
        if args.action_cost_weight is None
        else float(args.action_cost_weight)
    )
    action_rate_cost_weight = (
        float(train_config.get("action_rate_cost_weight", 0.0))
        if args.action_rate_cost_weight is None
        else float(args.action_rate_cost_weight)
    )
    velocity_cost_weight = (
        float(train_config.get("velocity_cost_weight", 0.0))
        if args.velocity_cost_weight is None
        else float(args.velocity_cost_weight)
    )
    if action_cost_weight < 0.0:
        raise ValueError("--action-cost-weight must be non-negative")
    if action_rate_cost_weight < 0.0:
        raise ValueError("--action-rate-cost-weight must be non-negative")
    if velocity_cost_weight < 0.0:
        raise ValueError("--velocity-cost-weight must be non-negative")
    if args.control_noise_std < 0.0:
        raise ValueError("--control-noise-std must be non-negative")

    vec_env = make_eval_env(
        task=args.task,
        seed=args.seed,
        time_limit=args.time_limit,
        vecnormalize_path=vecnormalize_path,
        action_cost_weight=action_cost_weight,
        action_rate_cost_weight=action_rate_cost_weight,
        velocity_cost_weight=velocity_cost_weight,
    )
    render_env = get_render_env(vec_env)
    model = SAC.load(str(model_path), env=vec_env, device=device)

    episode_rewards: list[float] = []
    episode_lengths: list[int] = []
    digits = max(2, len(str(args.videos - 1)))

    for episode_index in tqdm(range(args.videos), desc="Rendering", unit="video"):
        episode_seed = args.seed + episode_index
        frames, total_reward, num_steps = render_episode_frames(
            model=model,
            vec_env=vec_env,
            render_env=render_env,
            episode_seed=episode_seed,
            deterministic=not args.stochastic,
            max_steps=args.max_steps,
            width=args.width,
            height=args.height,
            control_noise=args.control_noise,
            control_noise_std=args.control_noise_std,
        )
        output_path = outdir / f"reacher_{episode_index:0{digits}d}.mp4"
        imageio.mimwrite(output_path, frames, fps=args.fps, quality=args.quality)
        episode_rewards.append(total_reward)
        episode_lengths.append(num_steps)
        print(
            f"saved {output_path} "
            f"(reward={total_reward:.3f}, steps={num_steps}, frames={frames.shape[0]})"
        )

    summary = {
        "model_path": str(model_path),
        "vecnormalize_path": str(vecnormalize_path),
        "task": args.task,
        "videos": args.videos,
        "outdir": str(outdir),
        "device": device,
        "deterministic": not args.stochastic,
        "control_noise": bool(args.control_noise),
        "control_noise_std": float(args.control_noise_std),
        "action_cost_weight": action_cost_weight,
        "action_rate_cost_weight": action_rate_cost_weight,
        "velocity_cost_weight": velocity_cost_weight,
        "mean_reward": float(np.mean(episode_rewards)) if episode_rewards else 0.0,
        "mean_length": float(np.mean(episode_lengths)) if episode_lengths else 0.0,
        "width": args.width,
        "height": args.height,
        "fps": args.fps,
    }
    print(json.dumps(summary, indent=2))
    vec_env.close()


if __name__ == "__main__":
    main()
