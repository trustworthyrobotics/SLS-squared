#!/usr/bin/env python3
"""Generate PushT rollouts with the diffusion policy for latent dynamics training."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import gymnasium as gym
import h5py
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

try:
    import hdf5plugin
except ModuleNotFoundError:
    hdf5plugin = None

from pusht.shared.pusht_env import (
    DEFAULT_PUSHT_ENV_ID,
    make_pusht_env,
    reset_pusht_env_to_obstacle_init as reset_pusht_env_to_insertion_init,
    reset_pusht_env_to_state,
)
from pusht.shared.utils import env_action_from_policy_action, load_expert_policy_bundle

DEFAULT_MODEL_DIR = Path("pusht/models")
DEFAULT_OUTPUT_PATH = Path("pusht/data/pusht_diffusion_insertion.h5")
ROLLOUT_MODES = (
    "expert",
    "expert_plus_noise",
    "expert_edge_sample",
    "biased_random",
    "insertion",
    "noised_insertion",
)
RATIOS = (0.0, 0.0, 0.0, 0.0, 0.6, 0.4)  # Matches ROLLOUT_MODES.
DEFAULT_MAX_ENV_STEPS_BY_MODE = (500, 500, 500, 50, 150, 150)
PUSHT_WALL_MIN = 5.0
PUSHT_WALL_MAX = 506.0
PUSHT_WALL_RADIUS = 2.0
PUSHT_AGENT_RADIUS = 15.0
PUSHT_TEE_CAP_TOP_Y = 0.0
ENV_ACTION_SCALE = 100.0
TEE_SCALE = 30.0
TEE_LENGTH = 4.0
TEE_BAR_X_MIN = -TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_X_MAX = TEE_LENGTH * TEE_SCALE / 2.0
TEE_BAR_Y_MIN = 0.0
TEE_BAR_Y_MAX = TEE_SCALE
TEE_STEM_X_MIN = -TEE_SCALE / 2.0
TEE_STEM_X_MAX = TEE_SCALE / 2.0
TEE_STEM_Y_MIN = TEE_SCALE
TEE_STEM_Y_MAX = TEE_LENGTH * TEE_SCALE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--env-id", default=DEFAULT_PUSHT_ENV_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--num-episodes", type=int, default=50_000)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument(
        "--max-steps-expert",
        type=int,
        default=DEFAULT_MAX_ENV_STEPS_BY_MODE[0],
        help="Max env steps for expert rollouts.",
    )
    parser.add_argument(
        "--max-steps-expert-noisy",
        type=int,
        default=DEFAULT_MAX_ENV_STEPS_BY_MODE[1],
        help="Max env steps for expert_plus_noise rollouts.",
    )
    parser.add_argument(
        "--max-steps-expert-edge-sample",
        type=int,
        default=DEFAULT_MAX_ENV_STEPS_BY_MODE[2],
        help="Max env steps for expert_edge_sample rollouts.",
    )
    parser.add_argument(
        "--max-steps-biased-random",
        type=int,
        default=DEFAULT_MAX_ENV_STEPS_BY_MODE[3],
        help="Max env steps for biased_random rollouts.",
    )
    parser.add_argument(
        "--max-steps-insertion",
        type=int,
        default=DEFAULT_MAX_ENV_STEPS_BY_MODE[4],
        help="Max env steps for insertion rollouts.",
    )
    parser.add_argument(
        "--max-steps-noised-insertion",
        type=int,
        default=DEFAULT_MAX_ENV_STEPS_BY_MODE[5],
        help="Max env steps for noised_insertion rollouts.",
    )
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument(
        "--control-interval",
        type=int,
        default=3,
        help="Query the policy every N env steps and store samples at that same interval.",
    )
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--async-envs", action="store_true", default=True)
    parser.add_argument("--keep-failures", action="store_true", default=True)
    parser.add_argument("--hide-target", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--expert-noise-std", type=float, default=8.5)
    parser.add_argument(
        "--insertion-init-block-offset",
        type=float,
        nargs=2,
        default=(130.0, 165.0),
        dest="insertion_init_block_offset",
        metavar=("MIN", "MAX"),
        help="Range to sample the insertion block offset from the goal along the negative local vertical axis.",
    )
    parser.add_argument(
        "--insertion-init-max-tilt-deg",
        type=float,
        default=30.0,
        dest="insertion_init_max_tilt_deg",
        help="Maximum absolute tilt sampled for insertion rollouts, in degrees.",
    )
    parser.add_argument(
        "--insertion-init-axis-threshold",
        type=float,
        default=10.0,
        dest="insertion_init_axis_threshold",
        help="Success threshold for insertion rollouts along the sampled tilted axis.",
    )
    parser.add_argument(
        "--insertion-init-pusher-face-offset",
        type=float,
        default=15.0,
        dest="insertion_init_pusher_face_offset",
        help="Distance from the T top face center to the pusher center for insertion rollouts.",
    )
    parser.add_argument(
        "--insertion-pusher-top-edge-margin",
        type=float,
        default=5.0,
        dest="insertion_pusher_top_edge_margin",
        help="Failure margin when the pusher crosses below the T cap top edge in insertion rollouts.",
    )
    parser.add_argument(
        "--insertion-pusher-top-edge-fail-steps",
        type=int,
        default=5,
        dest="insertion_pusher_top_edge_fail_steps",
        help="Consecutive env steps below the T cap top edge before insertion rollouts terminate as failure.",
    )
    parser.add_argument(
        "--biased-random-direction-kappa",
        type=float,
        default=1.5,
        help="Von Mises concentration around the direction to the T for biased_random rollouts.",
    )
    parser.add_argument(
        "--biased-random-magnitude-min",
        type=float,
        default=15.0,
        help="Minimum sampled target displacement magnitude in pixels for biased_random rollouts.",
    )
    parser.add_argument(
        "--biased-random-magnitude-max",
        type=float,
        default=35.0,
        help="Maximum sampled target displacement magnitude in pixels for biased_random rollouts.",
    )
    parser.add_argument(
        "--biased-random-aim-mode",
        choices=("center", "surface"),
        default="surface",
        help="Whether biased_random rollouts aim toward the block center or a random point on the T surface.",
    )
    parser.add_argument(
        "--pixel-compression",
        choices=("blosc", "lzf", "gzip", "none"),
        default="lzf",
        help="Compression for stored pixel frames.",
    )
    parser.add_argument(
        "--pixel-chunk-frames",
        type=int,
        default=100,
        help="Number of frames per pixels chunk in the output HDF5.",
    )
    return parser.parse_args()


def _as_xy(value: Any) -> np.ndarray:
    if hasattr(value, "x") and hasattr(value, "y"):
        return np.asarray([value.x, value.y], dtype=np.float32)
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size < 2:
        raise ValueError(f"Expected an xy-like value, got {value!r}")
    return arr[:2]


def _extract_pixels(raw_observation: dict[str, Any]) -> np.ndarray:
    if "pixels" not in raw_observation:
        raise KeyError("Expected PushT observation to contain 'pixels'.")
    pixels = raw_observation["pixels"]
    if isinstance(pixels, dict):
        pixels = next(iter(pixels.values()))
    pixels = np.asarray(pixels)
    if pixels.dtype != np.uint8:
        pixels = np.clip(pixels, 0, 255).astype(np.uint8)
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError(f"Expected HWC RGB pixels, got shape {pixels.shape}.")
    return pixels


def _extract_agent_xy(raw_observation: dict[str, Any], env) -> np.ndarray:
    if "agent_pos" in raw_observation:
        return np.asarray(raw_observation["agent_pos"], dtype=np.float32).reshape(-1)[:2]
    base_env = getattr(env, "unwrapped", env)
    return _as_xy(base_env.agent.position)


def _extract_block_pose(env) -> tuple[float, float, float]:
    base_env = getattr(env, "unwrapped", env)
    block_xy = _as_xy(base_env.block.position)
    block_theta = float(np.asarray(base_env.block.angle, dtype=np.float32).reshape(-1)[0])
    return float(block_xy[0]), float(block_xy[1]), block_theta


def _extract_goal_pose(env) -> tuple[float, float, float]:
    base_env = getattr(env, "unwrapped", env)
    goal = np.asarray(base_env.goal_pose, dtype=np.float32).reshape(-1)
    return float(goal[0]), float(goal[1]), float(goal[2])


def _pusht_agent_bounds() -> tuple[np.ndarray, np.ndarray]:
    min_coord = PUSHT_WALL_MIN + PUSHT_WALL_RADIUS + PUSHT_AGENT_RADIUS
    max_coord = PUSHT_WALL_MAX - PUSHT_WALL_RADIUS - PUSHT_AGENT_RADIUS
    low = np.full((2,), min_coord, dtype=np.float32)
    high = np.full((2,), max_coord, dtype=np.float32)
    return low, high


def _sample_agent_pos_on_edge(rng: np.random.Generator) -> np.ndarray:
    low, high = _pusht_agent_bounds()
    edge = int(rng.integers(4))
    coord = float(rng.uniform(float(low[0]), float(high[0])))
    if edge == 0:
        return np.asarray([low[0], coord], dtype=np.float32)
    if edge == 1:
        return np.asarray([high[0], coord], dtype=np.float32)
    if edge == 2:
        return np.asarray([coord, low[1]], dtype=np.float32)
    return np.asarray([coord, high[1]], dtype=np.float32)


class PushTPrivilegedObsWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, *, vector_env_index: int):
        super().__init__(env)
        self.vector_env_index = int(vector_env_index)
        self.observation_space = gym.spaces.Dict(
            {
                **self.observation_space.spaces,
                "block_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
                "goal_pose": gym.spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32),
            }
        )

    def _augment(self, observation: dict[str, Any]) -> dict[str, Any]:
        observation = dict(observation)
        observation["block_pose"] = np.asarray(_extract_block_pose(self.unwrapped), dtype=np.float32)
        observation["goal_pose"] = np.asarray(_extract_goal_pose(self.unwrapped), dtype=np.float32)
        return observation

    def reset(self, **kwargs):
        observation, info = self.env.reset(**kwargs)
        return self._augment(observation), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return self._augment(observation), reward, terminated, truncated, info

    def maybe_edge_sample_reset(
        self,
        enabled_by_env: np.ndarray | list[bool],
        seed_by_env: np.ndarray | list[int | None],
    ) -> dict[str, Any]:
        base_env = getattr(self.env, "unwrapped", self.env)
        enabled = bool(np.asarray(enabled_by_env)[self.vector_env_index])
        if not enabled:
            raw_observation = base_env.get_obs()
            return self._augment(raw_observation)

        seed = np.asarray(seed_by_env, dtype=object)[self.vector_env_index]
        rng = np.random.default_rng(None if seed is None else int(seed))
        sampled_agent_pos = _sample_agent_pos_on_edge(rng)
        block_x, block_y, block_theta = _extract_block_pose(self.unwrapped)
        state = np.asarray(
            [sampled_agent_pos[0], sampled_agent_pos[1], block_x, block_y, block_theta, 0.0, 0.0],
            dtype=np.float64,
        )
        reset_pusht_env_to_state(base_env, state)
        raw_observation = base_env.get_obs()
        return self._augment(raw_observation)

    def maybe_insertion_reset(
        self,
        enabled_by_env: np.ndarray | list[bool],
        seed_by_env: np.ndarray | list[int | None],
        block_offset_by_env: np.ndarray | list[float],
        max_tilt_deg: float,
        pusher_face_offset: float,
    ) -> dict[str, Any]:
        base_env = getattr(self.env, "unwrapped", self.env)
        enabled = bool(np.asarray(enabled_by_env)[self.vector_env_index])
        if not enabled:
            raw_observation = base_env.get_obs()
            return self._augment(raw_observation)

        seed = np.asarray(seed_by_env, dtype=object)[self.vector_env_index]
        rng = np.random.default_rng(None if seed is None else int(seed))
        tilt_deg = float(rng.uniform(-float(max_tilt_deg), float(max_tilt_deg)))
        block_offset = float(np.asarray(block_offset_by_env, dtype=np.float64)[self.vector_env_index])
        reset_pusht_env_to_insertion_init(
            base_env,
            block_offset=block_offset,
            tilt_deg=tilt_deg,
            pusher_face_offset=float(pusher_face_offset),
        )
        raw_observation = base_env.get_obs()
        return self._augment(raw_observation)


def _make_state(raw_observation: dict[str, Any], env) -> np.ndarray:
    if "block_pose" in raw_observation:
        block_x, block_y, block_theta = np.asarray(raw_observation["block_pose"], dtype=np.float32).reshape(-1)[:3]
    else:
        block_x, block_y, block_theta = _extract_block_pose(env)
    if "goal_pose" in raw_observation:
        goal_x, goal_y, goal_theta = np.asarray(raw_observation["goal_pose"], dtype=np.float32).reshape(-1)[:3]
    else:
        goal_x, goal_y, goal_theta = _extract_goal_pose(env)
    return np.asarray(
        [
            block_x,
            block_y,
            np.cos(block_theta),
            np.sin(block_theta),
            goal_x,
            goal_y,
            goal_theta,
        ],
        dtype=np.float32,
    )


def _make_proprio(raw_observation: dict[str, Any], env, previous_action: np.ndarray) -> np.ndarray:
    agent_xy = _extract_agent_xy(raw_observation, env)
    return np.asarray(
        [agent_xy[0], agent_xy[1], previous_action[0], previous_action[1]],
        dtype=np.float32,
    )


def _extract_success(terminated: bool, reward: float, info: dict[str, Any]) -> bool:
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).item())
    if "success" in info:
        return bool(np.asarray(info["success"]).item())
    return bool(terminated or reward >= 0.95)


def _policy_action_to_relative_action(policy_action: np.ndarray, agent_pos: np.ndarray) -> np.ndarray:
    return np.clip((policy_action - agent_pos) / 100.0, -1.0, 1.0).astype(np.float32)


def _clip_action_to_space_single(action: np.ndarray, action_space: gym.Space | None) -> np.ndarray:
    if action_space is None:
        return np.asarray(action, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    if high.shape != action.shape or low.shape != action.shape:
        return np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high).astype(np.float32)


def _target_xy_to_env_action(action_space: gym.Space | None, agent_xy: np.ndarray, target_xy: np.ndarray) -> np.ndarray:
    if action_space is not None:
        high = np.asarray(getattr(action_space, "high", None))
        low = np.asarray(getattr(action_space, "low", None))
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return np.clip((target_xy - agent_xy) / ENV_ACTION_SCALE, low, high).astype(np.float32)
    return target_xy.astype(np.float32)


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float32)


def _is_insertion_mode(mode: str) -> bool:
    return mode in {"insertion", "noised_insertion"}


def _uses_expert_policy(mode: str) -> bool:
    return mode in {"expert", "expert_plus_noise", "expert_edge_sample", "insertion", "noised_insertion"}


def _uses_noised_expert(mode: str) -> bool:
    return mode in {"expert_plus_noise", "noised_insertion"}


def _insertion_init_axis(block_theta: float) -> np.ndarray:
    return np.asarray([-np.sin(block_theta), np.cos(block_theta)], dtype=np.float64)


def _insertion_axis_distance(block_pose: np.ndarray, goal_pose: np.ndarray, axis: np.ndarray) -> float:
    block_xy = np.asarray(block_pose, dtype=np.float64).reshape(-1)[:2]
    goal_xy = np.asarray(goal_pose, dtype=np.float64).reshape(-1)[:2]
    axis = np.asarray(axis, dtype=np.float64)
    norm = float(np.linalg.norm(axis))
    if norm < 1e-9:
        raise ValueError("insertion-init axis must have nonzero norm.")
    return abs(float(np.dot(block_xy - goal_xy, axis / norm)))


def _pusher_local_y(block_pose: np.ndarray, agent_pos: np.ndarray) -> float:
    block_pose = np.asarray(block_pose, dtype=np.float64).reshape(-1)
    agent_pos = np.asarray(agent_pos, dtype=np.float64).reshape(-1)[:2]
    rotation = _rotation_matrix(float(block_pose[2])).astype(np.float64)
    local_xy = rotation.T @ (agent_pos - block_pose[:2])
    return float(local_xy[1])


def _sample_point_on_t(block_pose: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    bar_area = (TEE_BAR_X_MAX - TEE_BAR_X_MIN) * (TEE_BAR_Y_MAX - TEE_BAR_Y_MIN)
    stem_area = (TEE_STEM_X_MAX - TEE_STEM_X_MIN) * (TEE_STEM_Y_MAX - TEE_STEM_Y_MIN)
    if float(rng.uniform()) < bar_area / (bar_area + stem_area):
        local_xy = np.asarray(
            [
                rng.uniform(TEE_BAR_X_MIN, TEE_BAR_X_MAX),
                rng.uniform(TEE_BAR_Y_MIN, TEE_BAR_Y_MAX),
            ],
            dtype=np.float32,
        )
    else:
        local_xy = np.asarray(
            [
                rng.uniform(TEE_STEM_X_MIN, TEE_STEM_X_MAX),
                rng.uniform(TEE_STEM_Y_MIN, TEE_STEM_Y_MAX),
            ],
            dtype=np.float32,
        )
    return ((_rotation_matrix(float(block_pose[2])) @ local_xy) + block_pose[:2]).astype(np.float32)


def _rollout_probabilities() -> np.ndarray:
    ratios = np.asarray(RATIOS, dtype=np.float64)
    if ratios.shape != (len(ROLLOUT_MODES),):
        raise ValueError(f"RATIOS must have length {len(ROLLOUT_MODES)}.")
    if np.any(ratios < 0.0):
        raise ValueError("Rollout ratios cannot be negative.")
    total = float(ratios.sum())
    if total <= 0.0:
        raise ValueError("At least one rollout ratio must be positive.")
    return ratios / total


def _sample_rollout_modes(args: argparse.Namespace, rng: np.random.Generator, count: int) -> list[str]:
    return [str(mode) for mode in rng.choice(ROLLOUT_MODES, size=count, p=_rollout_probabilities())]


def _mode_max_steps(args: argparse.Namespace, mode: str) -> int:
    mapping = {
        "expert": args.max_steps_expert,
        "expert_plus_noise": args.max_steps_expert_noisy,
        "expert_edge_sample": args.max_steps_expert_edge_sample,
        "biased_random": args.max_steps_biased_random,
        "insertion": args.max_steps_insertion,
        "noised_insertion": args.max_steps_noised_insertion,
    }
    return int(mapping[mode])


def _max_episode_steps(args: argparse.Namespace) -> int:
    return max(_mode_max_steps(args, mode) for mode in ROLLOUT_MODES)


def _clip_action_to_space(actions: np.ndarray, action_space: gym.Space | None) -> np.ndarray:
    if action_space is None:
        return np.asarray(actions, dtype=np.float32)
    high = np.asarray(getattr(action_space, "high", None))
    low = np.asarray(getattr(action_space, "low", None))
    if high.shape != actions.shape[1:] or low.shape != actions.shape[1:]:
        return np.asarray(actions, dtype=np.float32)
    return np.clip(actions, low, high).astype(np.float32)


def _sample_biased_random_actions(
    raw_observations: dict[str, Any],
    action_space: gym.Space | None,
    rollout_modes: list[str],
    active: np.ndarray,
    episode_rngs: list[np.random.Generator],
    *,
    direction_kappa: float,
    magnitude_min: float,
    magnitude_max: float,
    aim_mode: str,
) -> tuple[np.ndarray, np.ndarray]:
    num_envs = len(rollout_modes)
    env_actions = np.zeros((num_envs, 2), dtype=np.float32)
    logged_actions = np.zeros((num_envs, 2), dtype=np.float32)
    if "block_pose" not in raw_observations:
        raise KeyError("biased_random rollout requires block_pose observations.")

    agent_positions = np.asarray(raw_observations["agent_pos"], dtype=np.float32)
    block_poses = np.asarray(raw_observations["block_pose"], dtype=np.float32)
    for env_idx in range(num_envs):
        if not active[env_idx] or rollout_modes[env_idx] != "biased_random":
            continue
        agent_xy = agent_positions[env_idx]
        block_pose = block_poses[env_idx]
        rng = episode_rngs[env_idx]
        if aim_mode == "center":
            aim_xy = block_pose[:2]
        else:
            aim_xy = _sample_point_on_t(block_pose, rng)
        aim_delta = aim_xy - agent_xy
        aim_angle = float(np.arctan2(aim_delta[1], aim_delta[0]))
        sampled_angle = float(rng.vonmises(aim_angle, direction_kappa))
        magnitude = float(rng.uniform(magnitude_min, magnitude_max))
        target_xy = agent_xy + magnitude * np.asarray([np.cos(sampled_angle), np.sin(sampled_angle)], dtype=np.float32)
        env_action = _clip_action_to_space_single(
            _target_xy_to_env_action(action_space, agent_xy, target_xy),
            action_space,
        )
        env_actions[env_idx] = env_action
        if action_space is not None:
            high = np.asarray(getattr(action_space, "high", None))
            low = np.asarray(getattr(action_space, "low", None))
            if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
                logged_actions[env_idx] = env_action.copy()
                continue
        logged_actions[env_idx] = _policy_action_to_relative_action(target_xy, agent_xy)
    return env_actions, logged_actions


def _apply_expert_noise(
    expert_env_action: np.ndarray,
    agent_pos: np.ndarray,
    action_space: gym.Space | None,
    rng: np.random.Generator,
    noise_std: float,
) -> tuple[np.ndarray, np.ndarray]:
    noise = rng.normal(loc=0.0, scale=noise_std, size=expert_env_action.shape).astype(np.float32)
    noisy_env_action = _clip_action_to_space(expert_env_action[None, :] + noise[None, :], action_space)[0]

    if action_space is not None:
        high = np.asarray(getattr(action_space, "high", None))
        low = np.asarray(getattr(action_space, "low", None))
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return noisy_env_action, noisy_env_action.copy()

    noisy_logged_action = _policy_action_to_relative_action(
        noisy_env_action,
        np.asarray(agent_pos, dtype=np.float32),
    )
    return noisy_env_action, noisy_logged_action


class H5EpisodeWriter:
    def __init__(self, out_path: Path, *, pixel_compression: str, pixel_chunk_frames: int):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path = out_path
        self.pixel_compression = pixel_compression
        self.pixel_chunk_frames = pixel_chunk_frames
        self.num_episodes = 0
        self.num_frames = 0
        self.h5 = h5py.File(out_path, "w")
        self.h5.create_dataset("ep_len", shape=(0,), maxshape=(None,), dtype=np.int32, chunks=True)
        self.h5.create_dataset("ep_offset", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("action", shape=(0, 2), maxshape=(None, 2), dtype=np.float32, chunks=True)
        self.h5.create_dataset("state", shape=(0, 7), maxshape=(None, 7), dtype=np.float32, chunks=True)
        self.h5.create_dataset("proprio", shape=(0, 4), maxshape=(None, 4), dtype=np.float32, chunks=True)
        self.h5.create_dataset("episode_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("step_idx", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.create_dataset("rollout_mode", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        self.h5.attrs["rollout_modes"] = np.asarray(ROLLOUT_MODES, dtype=h5py.string_dtype(encoding="utf-8"))

    def _pixel_create_kwargs(self, image_shape: tuple[int, int, int]) -> dict[str, Any]:
        kwargs: dict[str, Any] = {"chunks": (self.pixel_chunk_frames, *image_shape)}
        if self.pixel_compression == "none":
            return kwargs
        if self.pixel_compression == "lzf":
            return {**kwargs, "compression": "lzf"}
        if self.pixel_compression == "gzip":
            return {**kwargs, "compression": "gzip", "compression_opts": 4, "shuffle": True}
        if hdf5plugin is None:
            raise ModuleNotFoundError("Install hdf5plugin or use --pixel-compression lzf/gzip/none.")
        return {
            **kwargs,
            **hdf5plugin.Blosc(
                cname="lz4",
                clevel=5,
                shuffle=hdf5plugin.Blosc.SHUFFLE,
            ),
        }

    def _ensure_pixels_dataset(self, image_shape: tuple[int, int, int]) -> None:
        if "pixels" in self.h5:
            return
        self.h5.create_dataset(
            "pixels",
            shape=(0, *image_shape),
            maxshape=(None, *image_shape),
            dtype=np.uint8,
            **self._pixel_create_kwargs(image_shape),
        )

    def append_episodes(
        self,
        episodes: list[tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], int]],
    ) -> None:
        episodes = [episode for episode in episodes if len(episode[1]) > 0]
        if not episodes:
            return

        ep_lengths = np.asarray([len(ep_actions) for _, ep_actions, _, _, _ in episodes], dtype=np.int32)
        flat_pixels = [frame for ep_pixels, _, _, _, _ in episodes for frame in ep_pixels]
        flat_actions = [frame for _, ep_actions, _, _, _ in episodes for frame in ep_actions]
        flat_states = [frame for _, _, ep_states, _, _ in episodes for frame in ep_states]
        flat_proprios = [frame for _, _, _, ep_proprios, _ in episodes for frame in ep_proprios]

        pixel_array = np.stack(flat_pixels).astype(np.uint8, copy=False)
        action_array = np.stack(flat_actions).astype(np.float32, copy=False)
        state_array = np.stack(flat_states).astype(np.float32, copy=False)
        proprio_array = np.stack(flat_proprios).astype(np.float32, copy=False)
        rollout_mode_array = np.concatenate(
            [
                np.full(length, rollout_mode, dtype=np.int64)
                for length, (_, _, _, _, rollout_mode) in zip(ep_lengths, episodes, strict=True)
            ]
        )
        episode_idx_array = np.repeat(
            np.arange(self.num_episodes, self.num_episodes + len(episodes), dtype=np.int64),
            ep_lengths,
        )
        step_idx_array = np.concatenate([np.arange(length, dtype=np.int64) for length in ep_lengths])
        ep_offset_array = self.num_frames + np.concatenate(
            [np.asarray([0], dtype=np.int64), np.cumsum(ep_lengths[:-1], dtype=np.int64)]
        )

        num_new_frames = int(ep_lengths.sum())
        frame_start = self.num_frames
        frame_end = frame_start + num_new_frames
        ep_start = self.num_episodes
        ep_end = ep_start + len(episodes)

        self._ensure_pixels_dataset(tuple(pixel_array.shape[1:]))

        self.h5["ep_len"].resize((ep_end,))
        self.h5["ep_offset"].resize((ep_end,))
        self.h5["ep_len"][ep_start:ep_end] = ep_lengths
        self.h5["ep_offset"][ep_start:ep_end] = ep_offset_array

        arrays = {
            "pixels": pixel_array,
            "action": action_array,
            "state": state_array,
            "proprio": proprio_array,
            "episode_idx": episode_idx_array,
            "step_idx": step_idx_array,
            "rollout_mode": rollout_mode_array,
        }
        for name, array in arrays.items():
            dataset = self.h5[name]
            dataset.resize((frame_end, *dataset.shape[1:]))
            dataset[frame_start:frame_end] = array

        self.num_episodes = ep_end
        self.num_frames = frame_end

    def close(self) -> None:
        if "pixels" not in self.h5:
            self.h5.create_dataset(
                "pixels",
                shape=(0, 0, 0, 3),
                maxshape=(None, 0, 0, 3),
                dtype=np.uint8,
            )
        self.h5.close()


def make_env(args: argparse.Namespace, *, vector_env_index: int):
    env = make_pusht_env(
        args.env_id,
        obs_type="pixels_agent_pos",
        render_mode="rgb_array",
        max_episode_steps=_max_episode_steps(args),
        observation_width=args.image_width,
        observation_height=args.image_height,
        visualization_width=args.image_width,
        visualization_height=args.image_height,
        hide_target=args.hide_target,
    )
    return PushTPrivilegedObsWrapper(env, vector_env_index=vector_env_index)


def _resize_hwc_uint8(image: np.ndarray, height: int, width: int) -> np.ndarray:
    if image.shape[:2] == (height, width):
        return image
    return np.asarray(Image.fromarray(image).resize((width, height), Image.Resampling.BILINEAR), dtype=np.uint8)


def _slice_observation(raw_observations: dict[str, Any], env_index: int) -> dict[str, Any]:
    observation = {}
    for key, value in raw_observations.items():
        if isinstance(value, dict):
            observation[key] = {subkey: subvalue[env_index] for subkey, subvalue in value.items()}
        else:
            observation[key] = value[env_index]
    return observation


def _merge_vector_observations(
    raw_observations: dict[str, Any],
    refreshed_observations: list[dict[str, Any]] | tuple[dict[str, Any], ...],
    enabled_mask: np.ndarray,
) -> None:
    for env_idx, observation in enumerate(refreshed_observations):
        if not enabled_mask[env_idx]:
            continue
        for key, value in observation.items():
            if key not in raw_observations:
                continue
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    raw_observations[key][subkey][env_idx] = subvalue
            else:
                raw_observations[key][env_idx] = value


def _select_expert_actions(
    bundle,
    raw_observations: dict[str, Any],
    action_space: gym.Space | None,
) -> tuple[np.ndarray, np.ndarray]:
    image_shape = tuple(bundle.policy.config.input_features["observation.image"].shape)
    channels, height, width = image_shape
    if channels != 3:
        raise ValueError(f"Expected 3-channel policy input, got {image_shape=}.")

    pixels = raw_observations["pixels"]
    if isinstance(pixels, dict):
        pixels = next(iter(pixels.values()))
    pixels = np.asarray(pixels)
    resized = np.stack([_resize_hwc_uint8(frame, height, width) for frame in pixels], axis=0)
    images = torch.from_numpy(resized).permute(0, 3, 1, 2).contiguous().float() / 255.0
    states = torch.from_numpy(np.asarray(raw_observations["agent_pos"], dtype=np.float32))
    batch = {
        "observation.image": images,
        "observation.state": states,
    }
    batch = bundle.preprocessor(batch)
    action = bundle.policy.select_action(batch)
    action = bundle.postprocessor(action)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()

    policy_actions = np.asarray(action, dtype=np.float32).reshape(states.shape[0], 2)
    agent_pos = states.numpy()
    logged_actions = np.stack(
        [
            _policy_action_to_relative_action(policy_action, agent_pos[idx])
            for idx, policy_action in enumerate(policy_actions)
        ],
        axis=0,
    )

    if action_space is not None:
        high = np.asarray(getattr(action_space, "high", None))
        low = np.asarray(getattr(action_space, "low", None))
        if high.shape == (2,) and low.shape == (2,) and np.all(high <= 1.0) and np.all(low >= -1.0):
            return logged_actions, logged_actions

    env_actions = np.stack(
        [
            env_action_from_policy_action(
                policy_action,
                env=None,
                observation=_slice_observation(raw_observations, env_idx),
                action_mode="absolute",
            )
            for env_idx, policy_action in enumerate(policy_actions)
        ],
        axis=0,
    ).astype(np.float32)
    return env_actions, logged_actions


def _extract_vector_success(
    terminated: np.ndarray,
    rewards: np.ndarray,
    info: dict[str, Any],
    env_index: int,
) -> bool:
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, (list, tuple)) and final_info[env_index] is not None:
            env_info = final_info[env_index]
            if "is_success" in env_info:
                return bool(env_info["is_success"])
            if "success" in env_info:
                return bool(env_info["success"])
    if "is_success" in info:
        return bool(np.asarray(info["is_success"])[env_index])
    if "success" in info:
        return bool(np.asarray(info["success"])[env_index])
    return bool(terminated[env_index] or rewards[env_index] >= 0.95)


def make_vector_env(args: argparse.Namespace) -> gym.vector.VectorEnv:
    env_fns = [
        (lambda env_idx=env_idx, args=args: make_env(args, vector_env_index=env_idx))
        for env_idx in range(args.num_envs)
    ]
    env_cls = gym.vector.AsyncVectorEnv if args.async_envs else gym.vector.SyncVectorEnv
    try:
        from gymnasium.vector import AutoresetMode

        return env_cls(env_fns, autoreset_mode=AutoresetMode.SAME_STEP)
    except (ImportError, TypeError):
        return env_cls(env_fns)


def main() -> None:
    args = parse_args()
    if args.num_episodes < 1:
        raise ValueError("--num-episodes must be >= 1.")
    if args.control_interval < 1:
        raise ValueError("--control-interval must be >= 1.")
    if args.num_envs < 1:
        raise ValueError("--num-envs must be >= 1.")
    if args.pixel_chunk_frames < 1:
        raise ValueError("--pixel-chunk-frames must be >= 1.")
    if args.max_steps_expert < 1:
        raise ValueError("--max-steps-expert must be >= 1.")
    if args.max_steps_expert_noisy < 1:
        raise ValueError("--max-steps-expert-noisy must be >= 1.")
    if args.max_steps_expert_edge_sample < 1:
        raise ValueError("--max-steps-expert-edge-sample must be >= 1.")
    if args.max_steps_biased_random < 1:
        raise ValueError("--max-steps-biased-random must be >= 1.")
    if args.max_steps_insertion < 1:
        raise ValueError("--max-steps-insertion must be >= 1.")
    if args.max_steps_noised_insertion < 1:
        raise ValueError("--max-steps-noised-insertion must be >= 1.")
    if args.expert_noise_std < 0.0:
        raise ValueError("--expert-noise-std must be >= 0.")
    insertion_init_block_offset_range = np.asarray(args.insertion_init_block_offset, dtype=np.float64).reshape(-1)
    if insertion_init_block_offset_range.shape != (2,):
        raise ValueError("--insertion-init-block-offset expects exactly two values: MIN MAX.")
    if np.any(insertion_init_block_offset_range < 0.0):
        raise ValueError("--insertion-init-block-offset values must be >= 0.")
    if insertion_init_block_offset_range[1] < insertion_init_block_offset_range[0]:
        raise ValueError("--insertion-init-block-offset MAX must be >= MIN.")
    if args.insertion_init_pusher_face_offset < 0.0:
        raise ValueError("--insertion-init-pusher-face-offset must be >= 0.")
    if args.insertion_init_max_tilt_deg < 0.0:
        raise ValueError("--insertion-init-max-tilt-deg must be >= 0.")
    if args.insertion_init_axis_threshold < 0.0:
        raise ValueError("--insertion-init-axis-threshold must be >= 0.")
    if args.insertion_pusher_top_edge_margin < 0.0:
        raise ValueError("--insertion-pusher-top-edge-margin must be >= 0.")
    if args.insertion_pusher_top_edge_fail_steps < 1:
        raise ValueError("--insertion-pusher-top-edge-fail-steps must be >= 1.")
    if args.biased_random_direction_kappa < 0.0:
        raise ValueError("--biased-random-direction-kappa must be >= 0.")
    if args.biased_random_magnitude_min < 0.0:
        raise ValueError("--biased-random-magnitude-min must be >= 0.")
    if args.biased_random_magnitude_max < args.biased_random_magnitude_min:
        raise ValueError("--biased-random-magnitude-max must be >= --biased-random-magnitude-min.")
    if args.out.exists():
        raise FileExistsError(f"Output already exists: {args.out}")

    rollout_probabilities = _rollout_probabilities()
    uses_expert_policy = any(
        prob > 0.0 and _uses_expert_policy(mode)
        for prob, mode in zip(rollout_probabilities, ROLLOUT_MODES, strict=True)
    )
    bundle = load_expert_policy_bundle(args.model_dir, device=args.device) if uses_expert_policy else None
    writer = H5EpisodeWriter(
        args.out,
        pixel_compression=args.pixel_compression,
        pixel_chunk_frames=args.pixel_chunk_frames,
    )
    env = make_vector_env(args)
    action_space = getattr(env, "single_action_space", getattr(env, "action_space", None))
    mode_rng = np.random.default_rng(args.start_seed)
    saved = 0
    attempts = 0

    try:
        with tqdm(total=args.num_episodes, desc="Collecting episodes", unit="ep") as pbar:
            while saved < args.num_episodes:
                reset_seeds = [args.start_seed + attempts + env_idx for env_idx in range(args.num_envs)]
                attempts += args.num_envs
                if bundle is not None:
                    bundle.policy.reset()
                raw_observations, _ = env.reset(seed=reset_seeds)
                rollout_modes = _sample_rollout_modes(args, mode_rng, args.num_envs)
                edge_sample_mask = np.asarray(
                    [mode == "expert_edge_sample" for mode in rollout_modes],
                    dtype=bool,
                )
                if edge_sample_mask.any():
                    refreshed_observations = env.call(
                        "maybe_edge_sample_reset",
                        edge_sample_mask,
                        reset_seeds,
                    )
                    _merge_vector_observations(raw_observations, refreshed_observations, edge_sample_mask)
                insertion_mask = np.asarray(
                    [_is_insertion_mode(mode) for mode in rollout_modes],
                    dtype=bool,
                )
                insertion_block_offsets = np.zeros(args.num_envs, dtype=np.float64)
                for env_idx, seed in enumerate(reset_seeds):
                    if insertion_mask[env_idx]:
                        insertion_rng = np.random.default_rng(seed)
                        insertion_block_offsets[env_idx] = float(
                            insertion_rng.uniform(
                                insertion_init_block_offset_range[0],
                                insertion_init_block_offset_range[1],
                            )
                        )
                if insertion_mask.any():
                    refreshed_observations = env.call(
                        "maybe_insertion_reset",
                        insertion_mask,
                        reset_seeds,
                        insertion_block_offsets,
                        float(args.insertion_init_max_tilt_deg),
                        float(args.insertion_init_pusher_face_offset),
                    )
                    _merge_vector_observations(raw_observations, refreshed_observations, insertion_mask)
                previous_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                active = np.ones(args.num_envs, dtype=bool)
                env_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                logged_actions = np.zeros((args.num_envs, 2), dtype=np.float32)
                max_steps_by_env = np.asarray([_mode_max_steps(args, mode) for mode in rollout_modes], dtype=np.int32)
                episode_rngs = [np.random.default_rng(reset_seeds[env_idx]) for env_idx in range(args.num_envs)]
                insertion_axes: list[np.ndarray | None] = [None] * args.num_envs
                insertion_pusher_top_edge_violation_steps = np.zeros(args.num_envs, dtype=np.int32)
                if insertion_mask.any():
                    block_poses = np.asarray(raw_observations["block_pose"], dtype=np.float32)
                    for env_idx in np.flatnonzero(insertion_mask):
                        insertion_axes[env_idx] = _insertion_init_axis(float(block_poses[env_idx, 2]))
                batch_pixels: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_actions: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_states: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_proprios: list[list[np.ndarray]] = [[] for _ in range(args.num_envs)]
                batch_success = np.zeros(args.num_envs, dtype=bool)

                for step_idx in range(_max_episode_steps(args)):
                    reached_cap = active & (step_idx >= max_steps_by_env)
                    active[reached_cap] = False
                    if not active.any():
                        break

                    if step_idx % args.control_interval == 0:
                        env_actions.fill(0.0)
                        logged_actions.fill(0.0)
                        if bundle is not None:
                            expert_env_actions, expert_logged_actions = _select_expert_actions(
                                bundle,
                                raw_observations,
                                action_space,
                            )
                            expert_mask = np.asarray(
                                [
                                    active[idx] and rollout_modes[idx] in {"expert", "insertion"}
                                    for idx in range(args.num_envs)
                                ],
                                dtype=bool,
                            )
                            env_actions[expert_mask] = expert_env_actions[expert_mask]
                            logged_actions[expert_mask] = expert_logged_actions[expert_mask]
                            noisy_mask = np.asarray(
                                [active[idx] and _uses_noised_expert(rollout_modes[idx]) for idx in range(args.num_envs)],
                                dtype=bool,
                            )
                            noisy_indices = np.flatnonzero(noisy_mask)
                            for env_idx in noisy_indices:
                                noisy_env_action, noisy_logged_action = _apply_expert_noise(
                                    expert_env_actions[env_idx],
                                    np.asarray(raw_observations["agent_pos"][env_idx], dtype=np.float32),
                                    action_space,
                                    episode_rngs[env_idx],
                                    args.expert_noise_std,
                                )
                                env_actions[env_idx] = noisy_env_action
                                logged_actions[env_idx] = noisy_logged_action
                            edge_mask = np.asarray(
                                [active[idx] and rollout_modes[idx] == "expert_edge_sample" for idx in range(args.num_envs)],
                                dtype=bool,
                            )
                            env_actions[edge_mask] = expert_env_actions[edge_mask]
                            logged_actions[edge_mask] = expert_logged_actions[edge_mask]
                        random_env_actions, random_logged_actions = _sample_biased_random_actions(
                            raw_observations,
                            action_space,
                            rollout_modes,
                            active,
                            episode_rngs,
                            direction_kappa=args.biased_random_direction_kappa,
                            magnitude_min=args.biased_random_magnitude_min,
                            magnitude_max=args.biased_random_magnitude_max,
                            aim_mode=args.biased_random_aim_mode,
                        )
                        random_mask = np.asarray(
                            [active[idx] and rollout_modes[idx] == "biased_random" for idx in range(args.num_envs)],
                            dtype=bool,
                        )
                        env_actions[random_mask] = random_env_actions[random_mask]
                        logged_actions[random_mask] = random_logged_actions[random_mask]

                    for env_idx in range(args.num_envs):
                        if not active[env_idx]:
                            continue
                        if step_idx % args.control_interval == 0:
                            raw_observation = _slice_observation(raw_observations, env_idx)
                            batch_pixels[env_idx].append(_extract_pixels(raw_observation))
                            batch_actions[env_idx].append(logged_actions[env_idx].copy())
                            batch_states[env_idx].append(_make_state(raw_observation, env=None))
                            batch_proprios[env_idx].append(
                                _make_proprio(raw_observation, env=None, previous_action=previous_actions[env_idx])
                            )

                    raw_observations, rewards, terminated, truncated, info = env.step(env_actions)
                    for env_idx in range(args.num_envs):
                        if not active[env_idx]:
                            continue
                        if _is_insertion_mode(rollout_modes[env_idx]):
                            current_block_pose = np.asarray(raw_observations["block_pose"][env_idx], dtype=np.float32)
                            current_agent_pos = np.asarray(raw_observations["agent_pos"][env_idx], dtype=np.float32)
                            goal_pose = np.asarray(raw_observations["goal_pose"][env_idx], dtype=np.float32)
                            axis = insertion_axes[env_idx]
                            if axis is None:
                                raise RuntimeError("Missing insertion axis for insertion rollout.")
                            axis_distance = _insertion_axis_distance(current_block_pose, goal_pose, axis)
                            success = bool(axis_distance <= args.insertion_init_axis_threshold)
                            pusher_top_edge_violated = (
                                _pusher_local_y(current_block_pose, current_agent_pos)
                                > PUSHT_TEE_CAP_TOP_Y + args.insertion_pusher_top_edge_margin
                            )
                            if pusher_top_edge_violated:
                                insertion_pusher_top_edge_violation_steps[env_idx] += 1
                            else:
                                insertion_pusher_top_edge_violation_steps[env_idx] = 0
                            pusher_crossed_top_edge = (
                                insertion_pusher_top_edge_violation_steps[env_idx]
                                >= args.insertion_pusher_top_edge_fail_steps
                            )
                            batch_success[env_idx] = batch_success[env_idx] or success
                            if success or pusher_crossed_top_edge or truncated[env_idx]:
                                active[env_idx] = False
                        else:
                            batch_success[env_idx] = batch_success[env_idx] or _extract_vector_success(
                                terminated=terminated,
                                rewards=rewards,
                                info=info,
                                env_index=env_idx,
                            )
                        if active[env_idx] and (terminated[env_idx] or truncated[env_idx]):
                            active[env_idx] = False
                    if step_idx % args.control_interval == 0:
                        previous_actions = logged_actions.copy()
                    if not active.any():
                        break

                episodes_to_write = []
                last_saved_seed = None
                discarded = 0
                for env_idx, seed in enumerate(reset_seeds):
                    if saved + len(episodes_to_write) >= args.num_episodes:
                        break
                    if not batch_success[env_idx] and not args.keep_failures:
                        discarded += 1
                        continue
                    episodes_to_write.append(
                        (
                            batch_pixels[env_idx],
                            batch_actions[env_idx],
                            batch_states[env_idx],
                            batch_proprios[env_idx],
                            ROLLOUT_MODES.index(rollout_modes[env_idx]),
                        )
                    )
                    last_saved_seed = seed

                if episodes_to_write:
                    writer.append_episodes(episodes_to_write)
                    saved += len(episodes_to_write)
                    pbar.update(len(episodes_to_write))
                    pbar.set_postfix(
                        last_seed=last_saved_seed,
                        saved_batch=len(episodes_to_write),
                        stored_steps=sum(len(ep_actions) for _, ep_actions, _, _, _ in episodes_to_write),
                        discarded=discarded,
                        refresh=False,
                    )
                else:
                    pbar.set_postfix(
                        last_seed=reset_seeds[-1],
                        saved_batch=0,
                        stored_steps=0,
                        discarded=discarded,
                        refresh=False,
                    )
    finally:
        writer.close()
        env.close()

    if saved == 0:
        raise RuntimeError("No episodes were saved. Pass --keep-failures if you want to save unsuccessful rollouts.")

    print(f"Wrote {args.out.resolve()} with {saved} episodes and {writer.num_frames} frames.")


if __name__ == "__main__":
    main()
