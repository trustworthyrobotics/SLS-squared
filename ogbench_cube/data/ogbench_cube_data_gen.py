#!/usr/bin/env python3
"""Collect single-goal OGBench cube expert trajectories into LE-WM HDF5."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", os.environ["MUJOCO_GL"])
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import h5py
import imageio.v2 as imageio
import gymnasium
import mujoco
import numpy as np
import ogbench.manipspace  # noqa: F401
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from tqdm.auto import tqdm
from ogbench.manipspace import lie

DEFAULT_OUTDIR = "ogbench_cube/data/train_data_2"
DEFAULT_OUTPUT_NAME = "ogbench_cube_train_2.h5"
DEFAULT_ENV_NAME = "cube-single-v0"
DEFAULT_SIM_FREQ_HZ = 500.0
DEFAULT_CONTROL_DECIMATION = 25
VISUALIZE_TARGET = False
DETERMINISTIC_ARM_START = True
SAVE_DEPTH = False
XY_SAMPLING_BOUNDS = np.asarray([[0.30, -0.25], [0.5, 0.25]], dtype=np.float32) # x (front back), y (left right), [x_min, y_min] and [x_max, y_max]
Z_SAMPLING_BOUNDS = np.asarray([0.02, 0.30], dtype=np.float32)
THETA_SAMPLING_BOUNDS = np.asarray([0.0, 2.0 * np.pi], dtype=np.float32)
MIN_START_GOAL_SAMPLING_DIST = 0.10
ACTION_DIM = 5
DEFAULT_GOAL_YAW_THRESHOLD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-name", default=DEFAULT_ENV_NAME)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--target-transitions",
        type=int,
        default=600_000,
        help="Collect variable-length trajectories until this many action transitions are stored.",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        default=1500,
        help="Fallback collection target when --target-transitions is omitted.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-episode-steps", type=int, default=100)
    parser.add_argument("--min-steps", type=int, default=3)
    parser.add_argument("--sim-freq-hz", type=float, default=DEFAULT_SIM_FREQ_HZ)
    parser.add_argument("--control-decimation", type=int, default=DEFAULT_CONTROL_DECIMATION)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--quality", type=int, default=8)
    parser.add_argument("--compression", choices=("none", "lzf", "gzip"), default="lzf")
    parser.add_argument("--noise", type=float, default=0.0)
    parser.add_argument("--noise-smoothing", type=float, default=0.5)
    parser.add_argument("--segment-dt", type=float, default=0.4)
    parser.add_argument("--goal-threshold", type=float, default=0.04)
    parser.add_argument(
        "--goal-yaw-threshold",
        type=float,
        default=DEFAULT_GOAL_YAW_THRESHOLD,
        help="Maximum wrapped yaw error in radians for considering the cube at the goal orientation.",
    )
    parser.add_argument(
        "--post-goal-steps",
        type=int,
        default=10,
        help="Number of extra control steps to record after the cube first reaches the goal threshold.",
    )
    parser.add_argument("--camera", default="front_pixels")
    return parser.parse_args()


def create_resizable_dataset(
    h5: h5py.File,
    name: str,
    shape_tail: tuple[int, ...],
    dtype: np.dtype | type,
    *,
    compression: str | None = None,
    chunks: tuple[int, ...] | bool | None = True,
) -> h5py.Dataset:
    return h5.create_dataset(
        name,
        shape=(0, *shape_tail),
        maxshape=(None, *shape_tail),
        dtype=dtype,
        compression=compression,
        chunks=chunks,
    )


def append_rows(dataset: h5py.Dataset, values: np.ndarray) -> tuple[int, int]:
    start = int(dataset.shape[0])
    end = start + int(values.shape[0])
    dataset.resize((end, *dataset.shape[1:]))
    dataset[start:end] = values
    return start, end


def valid_training_windows(ep_len: np.ndarray, *, history_size: int = 3, num_preds: int = 1, frameskip: int = 1) -> int:
    num_steps = history_size + num_preds
    required_last_frame_offset = (num_steps - 1) * frameskip
    required_action_end_offset = history_size * frameskip
    required_offset = max(required_last_frame_offset, required_action_end_offset)
    return int(np.maximum(ep_len - 1 - required_offset + 1, 0).sum())


def should_continue(args: argparse.Namespace, num_trajectories: int, total_transitions: int) -> bool:
    if args.target_transitions is not None:
        return total_transitions < args.target_transitions
    return num_trajectories < args.num_trajectories


def compute_env_timing(sim_freq_hz: float, control_decimation: int) -> tuple[float, float, float]:
    if sim_freq_hz <= 0.0:
        raise ValueError("--sim-freq-hz must be positive.")
    if control_decimation < 1:
        raise ValueError("--control-decimation must be positive.")
    physics_timestep = 1.0 / sim_freq_hz
    control_timestep = physics_timestep * control_decimation
    control_freq_hz = 1.0 / control_timestep
    return physics_timestep, control_timestep, control_freq_hz


def apply_xy_sampling_bounds(env: object) -> None:
    xy_bounds = np.asarray(XY_SAMPLING_BOUNDS, dtype=np.float64)
    if xy_bounds.shape != (2, 2):
        raise ValueError(f"XY_SAMPLING_BOUNDS must have shape (2, 2), got {xy_bounds.shape}.")
    env.unwrapped._object_sampling_bounds = xy_bounds
    env.unwrapped._target_sampling_bounds = xy_bounds


def apply_deterministic_arm_start(
    env: object,
    *,
    mujoco_module: object,
    lie_module: object,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    arm_bounds = np.asarray(env.unwrapped._arm_sampling_bounds, dtype=np.float64)
    center_pos = arm_bounds.mean(axis=0)
    center_yaw = 0.0

    target_effector_orientation = lie_module.SO3.from_z_radians(center_yaw) @ env.unwrapped._effector_down_rotation
    pinch_pose = lie_module.SE3.from_rotation_and_translation(
        rotation=target_effector_orientation,
        translation=center_pos,
    )
    attach_pose = pinch_pose @ env.unwrapped._T_pa
    qpos_init = env.unwrapped._ik.solve(
        pos=attach_pose.translation(),
        quat=attach_pose.rotation().wxyz,
        curr_qpos=env.unwrapped._home_qpos,
    )

    env.unwrapped._data.qpos[env.unwrapped._arm_joint_ids] = qpos_init
    env.unwrapped.pre_step()
    mujoco_module.mj_forward(env.unwrapped._model, env.unwrapped._data)
    env.unwrapped.post_step()

    ob = np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
    reset_info = env.unwrapped.get_reset_info()
    return ob, reset_info


def sample_z(z_rng: np.random.Generator) -> float:
    z_bounds = np.asarray(Z_SAMPLING_BOUNDS, dtype=np.float64)
    if z_bounds.shape != (2,):
        raise ValueError(f"Z_SAMPLING_BOUNDS must have shape (2,), got {z_bounds.shape}.")
    z_min = float(z_bounds[0])
    z_max = float(z_bounds[1])
    if z_max < z_min:
        raise ValueError(
            "Z_SAMPLING_BOUNDS must satisfy z_max >= z_min, "
            f"got {z_min} and {z_max}."
        )
    return float(z_rng.uniform(z_min, z_max))


def sample_theta(theta_rng: np.random.Generator) -> float:
    theta_bounds = np.asarray(THETA_SAMPLING_BOUNDS, dtype=np.float64)
    if theta_bounds.shape != (2,):
        raise ValueError(f"THETA_SAMPLING_BOUNDS must have shape (2,), got {theta_bounds.shape}.")
    theta_min = float(theta_bounds[0])
    theta_max = float(theta_bounds[1])
    if theta_max < theta_min:
        raise ValueError(
            "THETA_SAMPLING_BOUNDS must satisfy theta_max >= theta_min, "
            f"got {theta_min} and {theta_max}."
        )
    return float(theta_rng.uniform(theta_min, theta_max))


def apply_theta_sampling_bounds(
    env: object,
    info: dict[str, np.ndarray],
    rng: np.random.Generator,
    *,
    mujoco_module: object,
    lie_module: object,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    target_block = int(info["privileged/target_block"])
    target_mocap_id = env.unwrapped._cube_target_mocap_ids[target_block]

    block_theta = sample_theta(rng)
    target_theta = sample_theta(rng)
    target_z = sample_z(rng)
    block_quat = np.asarray(lie_module.SO3.from_z_radians(block_theta).wxyz, dtype=np.float64)
    target_quat = np.asarray(lie_module.SO3.from_z_radians(target_theta).wxyz, dtype=np.float64)

    env.unwrapped._data.joint(f"object_joint_{target_block}").qpos[3:] = block_quat
    env.unwrapped._data.mocap_pos[target_mocap_id, 2] = target_z
    env.unwrapped._data.mocap_quat[target_mocap_id] = target_quat
    env.unwrapped.pre_step()
    mujoco_module.mj_forward(env.unwrapped._model, env.unwrapped._data)
    env.unwrapped.post_step()

    ob = np.asarray(env.unwrapped.compute_observation(), dtype=np.float32)
    bounded_info = env.unwrapped.get_reset_info()
    return ob, bounded_info


def sample_valid_reset(
    env: object,
    trajectory_seed: int,
    min_sampling_dist: float,
    *,
    mujoco_module: object,
    lie_module: object,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if min_sampling_dist < 0.0:
        raise ValueError(f"MIN_SAMPLING_DIST cannot be negative, got {min_sampling_dist}.")

    seed_offset = 0
    while True:
        ob, info = env.reset(seed=trajectory_seed + seed_offset)
        if DETERMINISTIC_ARM_START:
            ob, info = apply_deterministic_arm_start(
                env,
                mujoco_module=mujoco_module,
                lie_module=lie_module,
            )
        rng = np.random.default_rng(trajectory_seed + seed_offset)
        ob, info = apply_theta_sampling_bounds(
            env,
            info,
            rng,
            mujoco_module=mujoco_module,
            lie_module=lie_module,
        )
        start_xy = np.asarray(info["privileged/block_0_pos"][:2], dtype=np.float64)
        goal_xy = np.asarray(info["privileged/target_block_pos"][:2], dtype=np.float64)
        if np.linalg.norm(start_xy - goal_xy) >= min_sampling_dist:
            return ob, info
        seed_offset += 1


def angular_distance(a: float, b: float) -> float:
    return float(np.abs(np.arctan2(np.sin(a - b), np.cos(a - b))))


def symmetry_aware_yaw_distance(a: float, b: float, n: int = 4) -> float:
    symmetries = [b + i * 2 * np.pi / n for i in range(n)]
    return min(angular_distance(a, sym_yaw) for sym_yaw in symmetries)


class LocalCubePlanOracle:
    def __init__(self, env: object, segment_dt: float = 0.4, noise: float = 0.1, noise_smoothing: float = 0.5):
        self._env = env
        self._env_dt = self._env.unwrapped._control_timestep
        self._dt = segment_dt
        self._noise = noise
        self._noise_smoothing = noise_smoothing

        self._done = False
        self._t_init: float | None = None
        self._t_max: float | None = None
        self._plan: np.ndarray | None = None

    def above(self, pose: lie.SE3, z: float) -> lie.SE3:
        return (
            lie.SE3.from_rotation_and_translation(
                rotation=lie.SO3.identity(),
                translation=np.array([0.0, 0.0, z], dtype=np.float64),
            )
            @ pose
        )

    def to_pose(self, pos: np.ndarray, yaw: float) -> lie.SE3:
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(yaw),
            translation=np.asarray(pos, dtype=np.float64),
        )

    def get_yaw(self, pose: lie.SE3) -> float:
        yaw = float(pose.rotation().compute_yaw_radians())
        if yaw < 0.0:
            return yaw + 2 * np.pi
        return yaw

    def shortest_yaw(self, eff_yaw: float, obj_yaw: float, translation: np.ndarray, n: int = 4) -> lie.SE3:
        symmetries = np.array([i * 2 * np.pi / n + obj_yaw for i in range(-n, n + 1)], dtype=np.float64)
        closest_idx = int(np.argmin(np.abs(eff_yaw - symmetries)))
        return lie.SE3.from_rotation_and_translation(
            rotation=lie.SO3.from_z_radians(symmetries[closest_idx]),
            translation=np.asarray(translation, dtype=np.float64),
        )

    def compute_plan(self, times: list[float], poses: list[lie.SE3], grasps: list[float]) -> np.ndarray:
        grasp_interp = interp1d(times, grasps, kind="linear", axis=0, assume_sorted=True)

        xyzs = [p.translation() for p in poses]
        xyz_interp = interp1d(times, xyzs, kind="linear", axis=0, assume_sorted=True)

        quats = [p.rotation() for p in poses]

        def quat_interp(t: float) -> lie.SO3:
            seg_idx = int(np.searchsorted(times, t, side="right") - 1)
            seg_idx = int(np.clip(seg_idx, 0, len(times) - 2))
            interp_time = (t - times[seg_idx]) / (times[seg_idx + 1] - times[seg_idx])
            interp_time = float(np.clip(interp_time, 0.0, 1.0))
            return lie.interpolate(quats[seg_idx], quats[seg_idx + 1], interp_time)

        plan = []
        t = 0.0
        assert self._t_max is not None
        while t < self._t_max:
            action = np.zeros(ACTION_DIM, dtype=np.float64)
            action[:3] = xyz_interp(t)
            action[3] = quat_interp(t).compute_yaw_radians()
            action[4] = grasp_interp(t)
            plan.append(action)
            t += self._env_dt

        plan_array = np.asarray(plan, dtype=np.float64)

        if self._noise > 0:
            noise = (
                np.random.normal(0.0, 1.0, size=plan_array.shape)
                * np.array([0.05, 0.05, 0.05, 0.3, 1.0], dtype=np.float64)
                * self._noise
            )
            noise = gaussian_filter1d(noise, axis=0, sigma=self._noise_smoothing)
            plan_array += noise
            plan_array[:, 4] = np.clip(plan_array[:, 4], 0.0, 1.0)

        return plan_array

    def compute_keyframes(
        self,
        *,
        effector_initial: lie.SE3,
        block_initial: lie.SE3,
        block_goal: lie.SE3,
    ) -> tuple[dict[str, float], dict[str, lie.SE3], dict[str, float]]:
        poses: dict[str, lie.SE3] = {}

        block_pick = self.shortest_yaw(
            eff_yaw=self.get_yaw(effector_initial),
            obj_yaw=self.get_yaw(block_initial),
            translation=block_initial.translation(),
        )
        poses["initial"] = effector_initial
        poses["pick"] = self.above(block_pick, 0.1 + np.random.uniform(0.0, 0.1))
        poses["pick_start"] = block_pick
        poses["pick_end"] = block_pick
        poses["postpick"] = poses["pick"]

        block_goal_aligned = self.shortest_yaw(
            eff_yaw=self.get_yaw(poses["postpick"]),
            obj_yaw=self.get_yaw(block_goal),
            translation=block_goal.translation(),
        )
        poses["clearance"] = lie.interpolate(poses["postpick"], self.above(block_goal_aligned, 0.1), 0.5)
        poses["goal_approach"] = self.above(block_goal_aligned, 0.05)
        poses["goal_hold"] = block_goal_aligned
        poses["final"] = block_goal_aligned

        times = {
            "initial": 0.0,
            "pick": self._dt,
            "pick_start": self._dt * 2.5,
            "pick_end": self._dt * 3.5,
            "postpick": self._dt * 4.5,
            "clearance": self._dt * 5.5,
            "goal_approach": self._dt * 6.5,
            "goal_hold": self._dt * 8.0,
            "final": self._dt * 9.0,
        }
        for name in list(times.keys()):
            if name != "initial":
                times[name] += float(np.random.uniform(-1.0, 1.0) * self._dt * 0.2)

        grasps = {}
        grasp = 0.0
        for name in times.keys():
            if name == "pick_end":
                grasp = 1.0
            grasps[name] = grasp

        return times, poses, grasps

    def reset(self, ob: np.ndarray, info: dict[str, np.ndarray]) -> None:
        target_block = int(info["privileged/target_block"])
        effector_initial = self.to_pose(
            pos=info["proprio/effector_pos"],
            yaw=float(info["proprio/effector_yaw"][0]),
        )
        block_initial = self.to_pose(
            pos=info[f"privileged/block_{target_block}_pos"],
            yaw=float(info[f"privileged/block_{target_block}_yaw"][0]),
        )
        block_goal = self.to_pose(
            pos=info["privileged/target_block_pos"],
            yaw=float(info["privileged/target_block_yaw"][0]),
        )

        times, poses, grasps = self.compute_keyframes(
            effector_initial=effector_initial,
            block_initial=block_initial,
            block_goal=block_goal,
        )
        ordered_names = list(times.keys())
        ordered_poses = [poses[name] for name in ordered_names]
        ordered_grasps = [grasps[name] for name in ordered_names]
        ordered_times = [times[name] for name in ordered_names]

        self._t_init = float(info["time"][0])
        self._t_max = float(ordered_times[-1])
        self._done = False
        self._plan = self.compute_plan(ordered_times, ordered_poses, ordered_grasps)

    def select_action(self, ob: np.ndarray, info: dict[str, np.ndarray]) -> np.ndarray:
        if self._plan is None or self._t_init is None:
            raise RuntimeError("Oracle must be reset before select_action is called.")

        cur_plan_idx = int((float(info["time"][0]) - self._t_init + 1e-7) // self._env_dt)
        if cur_plan_idx >= len(self._plan) - 1:
            cur_plan_idx = len(self._plan) - 1
            self._done = True

        absolute_target = self._plan[cur_plan_idx]
        action = np.zeros(ACTION_DIM, dtype=np.float64)
        action[:3] = absolute_target[:3] - info["proprio/effector_pos"]
        action[3] = absolute_target[3] - float(info["proprio/effector_yaw"][0])
        action[4] = absolute_target[4] - float(info["proprio/gripper_opening"][0])
        return np.asarray(self._env.unwrapped.normalize_action(action), dtype=np.float32)


def extract_step_info(info: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "qpos": np.asarray(info["qpos"], dtype=np.float32),
        "qvel": np.asarray(info["qvel"], dtype=np.float32),
        "control": np.asarray(info["control"], dtype=np.float32),
        "effector_pos": np.asarray(info["proprio/effector_pos"], dtype=np.float32),
        "effector_yaw": np.asarray(info["proprio/effector_yaw"], dtype=np.float32),
        "gripper_opening": np.asarray(info["proprio/gripper_opening"], dtype=np.float32),
        "gripper_contact": np.asarray(info["proprio/gripper_contact"], dtype=np.float32),
        "block_pos": np.asarray(info["privileged/block_0_pos"], dtype=np.float32),
        "block_quat": np.asarray(info["privileged/block_0_quat"], dtype=np.float32),
        "block_yaw": np.asarray(info["privileged/block_0_yaw"], dtype=np.float32),
        "target_block_pos": np.asarray(info["privileged/target_block_pos"], dtype=np.float32),
        "target_block_yaw": np.asarray(info["privileged/target_block_yaw"], dtype=np.float32),
        "time": np.asarray(info["time"], dtype=np.float32),
    }


def reached_goal(info: dict[str, np.ndarray], threshold: float, yaw_threshold: float) -> bool:
    pos_ok = np.linalg.norm(info["privileged/block_0_pos"] - info["privileged/target_block_pos"]) <= threshold
    yaw_ok = symmetry_aware_yaw_distance(
        float(info["privileged/block_0_yaw"][0]),
        float(info["privileged/target_block_yaw"][0]),
    ) <= yaw_threshold
    return bool(pos_ok and yaw_ok)


def collect_trajectory(
    *,
    env: object,
    oracle: object,
    trajectory_seed: int,
    camera: str,
    save_depth: bool,
    goal_threshold: float,
    goal_yaw_threshold: float,
    post_goal_steps: int,
    mujoco_module: object,
    lie_module: object,
) -> tuple[dict[str, np.ndarray], float, bool, bool]:
    ob, info = sample_valid_reset(
        env,
        trajectory_seed,
        MIN_START_GOAL_SAMPLING_DIST,
        mujoco_module=mujoco_module,
        lie_module=lie_module,
    )
    oracle.reset(ob, info)

    observations = [np.asarray(ob, dtype=np.float32)]
    frames = [np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8)]
    depth_frames = [np.asarray(env.unwrapped.render(camera=camera, depth=True), dtype=np.float32)] if save_depth else []
    actions: list[np.ndarray] = []
    qpos = [np.asarray(info["qpos"], dtype=np.float32)]
    qvel = [np.asarray(info["qvel"], dtype=np.float32)]
    control = [np.asarray(info["control"], dtype=np.float32)]
    effector_pos = [np.asarray(info["proprio/effector_pos"], dtype=np.float32)]
    effector_yaw = [np.asarray(info["proprio/effector_yaw"], dtype=np.float32)]
    gripper_opening = [np.asarray(info["proprio/gripper_opening"], dtype=np.float32)]
    gripper_contact = [np.asarray(info["proprio/gripper_contact"], dtype=np.float32)]
    block_pos = [np.asarray(info["privileged/block_0_pos"], dtype=np.float32)]
    block_quat = [np.asarray(info["privileged/block_0_quat"], dtype=np.float32)]
    block_yaw = [np.asarray(info["privileged/block_0_yaw"], dtype=np.float32)]
    target_block_pos = [np.asarray(info["privileged/target_block_pos"], dtype=np.float32)]
    target_block_yaw = [np.asarray(info["privileged/target_block_yaw"], dtype=np.float32)]
    time = [np.asarray(info["time"], dtype=np.float32)]

    total_reward = 0.0
    terminated = False
    truncated = False
    success = reached_goal(info, goal_threshold, goal_yaw_threshold)
    goal_seen = success
    post_goal_step_count = 0

    while not (terminated or truncated or (goal_seen and post_goal_step_count >= post_goal_steps)):
        action = np.asarray(oracle.select_action(ob, info), dtype=np.float32)
        next_ob, reward, terminated, truncated, next_info = env.step(action)
        total_reward += float(reward)

        actions.append(action)
        observations.append(np.asarray(next_ob, dtype=np.float32))
        frames.append(np.asarray(env.unwrapped.render(camera=camera), dtype=np.uint8))
        if save_depth:
            depth_frames.append(np.asarray(env.unwrapped.render(camera=camera, depth=True), dtype=np.float32))

        step_info = extract_step_info(next_info)
        qpos.append(step_info["qpos"])
        qvel.append(step_info["qvel"])
        control.append(step_info["control"])
        effector_pos.append(step_info["effector_pos"])
        effector_yaw.append(step_info["effector_yaw"])
        gripper_opening.append(step_info["gripper_opening"])
        gripper_contact.append(step_info["gripper_contact"])
        block_pos.append(step_info["block_pos"])
        block_quat.append(step_info["block_quat"])
        block_yaw.append(step_info["block_yaw"])
        target_block_pos.append(step_info["target_block_pos"])
        target_block_yaw.append(step_info["target_block_yaw"])
        time.append(step_info["time"])

        ob = next_ob
        info = next_info
        success = reached_goal(info, goal_threshold, goal_yaw_threshold)
        if goal_seen:
            post_goal_step_count += 1
        elif success:
            goal_seen = True

    data = {
        "observation": np.stack(observations, axis=0),
        "pixels": np.stack(frames, axis=0),
        "action": np.stack(actions, axis=0) if actions else np.zeros((0, ACTION_DIM), dtype=np.float32),
        "qpos": np.stack(qpos, axis=0),
        "qvel": np.stack(qvel, axis=0),
        "control": np.stack(control, axis=0),
        "effector_pos": np.stack(effector_pos, axis=0),
        "effector_yaw": np.stack(effector_yaw, axis=0),
        "gripper_opening": np.stack(gripper_opening, axis=0),
        "gripper_contact": np.stack(gripper_contact, axis=0),
        "block_pos": np.stack(block_pos, axis=0),
        "block_quat": np.stack(block_quat, axis=0),
        "block_yaw": np.stack(block_yaw, axis=0),
        "target_block_pos": np.stack(target_block_pos, axis=0),
        "target_block_yaw": np.stack(target_block_yaw, axis=0),
        "time": np.stack(time, axis=0),
    }
    if save_depth:
        data["depth"] = np.stack(depth_frames, axis=0)
    return data, total_reward, bool(terminated), bool(truncated), bool(goal_seen)


def main() -> None:
    args = parse_args()
    if args.target_transitions is not None and args.target_transitions < 1:
        raise ValueError("--target-transitions must be positive when provided.")
    if args.num_trajectories < 1:
        raise ValueError("--num-trajectories must be positive.")
    if args.max_episode_steps < 1:
        raise ValueError("--max-episode-steps must be positive.")
    if args.min_steps < 1:
        raise ValueError("--min-steps must be positive.")
    if args.goal_yaw_threshold < 0.0:
        raise ValueError("--goal-yaw-threshold cannot be negative.")
    if args.post_goal_steps < 0:
        raise ValueError("--post-goal-steps cannot be negative.")
    physics_timestep, control_timestep, control_freq_hz = compute_env_timing(
        args.sim_freq_hz,
        args.control_decimation,
    )

    outdir = args.outdir.expanduser().resolve()
    video_dir = outdir / "videos"
    output_path = outdir / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Output file already exists: {output_path}. Pass --overwrite to replace it.")
    if output_path.exists():
        output_path.unlink()

    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=VISUALIZE_TARGET,
        max_episode_steps=args.max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=args.width,
        height=args.height,
    )
    apply_xy_sampling_bounds(env)
    oracle = LocalCubePlanOracle(
        env=env,
        segment_dt=args.segment_dt,
        noise=args.noise,
        noise_smoothing=args.noise_smoothing,
    )

    compression = None if args.compression == "none" else args.compression
    rewards: list[float] = []
    step_counts: list[int] = []
    terminated_flags: list[bool] = []
    truncated_flags: list[bool] = []
    failed_flags: list[bool] = []
    skipped_short = 0
    seed_offset = 0

    sample_ob, sample_info = env.reset(seed=args.seed)
    if DETERMINISTIC_ARM_START:
        sample_ob, sample_info = apply_deterministic_arm_start(
            env,
            mujoco_module=mujoco,
            lie_module=lie,
        )
    sample_frame = np.asarray(env.unwrapped.render(camera=args.camera), dtype=np.uint8)
    sample_depth = np.asarray(env.unwrapped.render(camera=args.camera, depth=True), dtype=np.float32) if SAVE_DEPTH else None
    obs_dim = int(np.asarray(sample_ob).shape[0])
    qpos_dim = int(np.asarray(sample_info["qpos"]).shape[0])
    qvel_dim = int(np.asarray(sample_info["qvel"]).shape[0])
    control_dim = int(np.asarray(sample_info["control"]).shape[0])
    env.close()

    env = gymnasium.make(
        args.env_name,
        terminate_at_goal=False,
        mode="data_collection",
        visualize_info=VISUALIZE_TARGET,
        max_episode_steps=args.max_episode_steps,
        physics_timestep=physics_timestep,
        control_timestep=control_timestep,
        width=args.width,
        height=args.height,
    )
    apply_xy_sampling_bounds(env)
    oracle = LocalCubePlanOracle(
        env=env,
        segment_dt=args.segment_dt,
        noise=args.noise,
        noise_smoothing=args.noise_smoothing,
    )

    with h5py.File(output_path, "w") as h5:
        h5.attrs["format"] = "stable_worldmodel_hdf5"
        h5.attrs["source"] = "ogbench_cube/data/ogbench_cube_data_gen.py"
        h5.attrs["env_name"] = args.env_name
        h5.attrs["seed"] = args.seed
        h5.attrs["goal_threshold"] = args.goal_threshold
        h5.attrs["goal_yaw_threshold"] = args.goal_yaw_threshold
        h5.attrs["post_goal_steps"] = args.post_goal_steps
        h5.attrs["video_dir"] = str(video_dir)
        h5.attrs["video_resolution"] = json.dumps([args.height, args.width])
        h5.attrs["camera"] = args.camera
        h5.attrs["save_depth"] = SAVE_DEPTH
        h5.attrs["target_visualization"] = VISUALIZE_TARGET
        h5.attrs["deterministic_arm_start"] = DETERMINISTIC_ARM_START
        h5.attrs["video_fps"] = control_freq_hz
        h5.attrs["sim_freq_hz"] = args.sim_freq_hz
        h5.attrs["control_freq_hz"] = control_freq_hz
        h5.attrs["control_decimation"] = args.control_decimation
        h5.attrs["physics_timestep"] = physics_timestep
        h5.attrs["control_timestep"] = control_timestep
        h5.attrs["max_episode_steps"] = args.max_episode_steps
        h5.attrs["xy_sampling_bounds"] = json.dumps(XY_SAMPLING_BOUNDS.tolist())
        h5.attrs["z_sampling_bounds"] = json.dumps(Z_SAMPLING_BOUNDS.tolist())
        h5.attrs["theta_sampling_bounds"] = json.dumps(THETA_SAMPLING_BOUNDS.tolist())
        h5.attrs["min_sampling_dist"] = MIN_START_GOAL_SAMPLING_DIST
        h5.attrs["noise"] = args.noise
        h5.attrs["noise_smoothing"] = args.noise_smoothing
        h5.attrs["segment_dt"] = args.segment_dt
        h5.attrs["observation_dim"] = obs_dim
        h5.attrs["action_dim"] = ACTION_DIM
        h5.attrs["qpos_dim"] = qpos_dim
        h5.attrs["qvel_dim"] = qvel_dim
        h5.attrs["control_dim"] = control_dim

        ep_len_ds = create_resizable_dataset(h5, "ep_len", (), np.int64, chunks=True)
        ep_offset_ds = create_resizable_dataset(h5, "ep_offset", (), np.int64, chunks=True)
        reward_ds = create_resizable_dataset(h5, "reward", (), np.float32, chunks=True)
        seed_ds = create_resizable_dataset(h5, "episode_seed", (), np.int64, chunks=True)
        terminated_ds = create_resizable_dataset(h5, "terminated", (), np.bool_, chunks=True)
        truncated_ds = create_resizable_dataset(h5, "truncated", (), np.bool_, chunks=True)
        pixels_ds = create_resizable_dataset(
            h5,
            "pixels",
            sample_frame.shape,
            np.uint8,
            compression=compression,
            chunks=(1, *sample_frame.shape),
        )
        depth_ds = (
            create_resizable_dataset(
                h5,
                "depth",
                sample_depth.shape,
                np.float32,
                compression=compression,
                chunks=(1, *sample_depth.shape),
            )
            if SAVE_DEPTH
            else None
        )
        action_ds = create_resizable_dataset(h5, "action", (ACTION_DIM,), np.float32, chunks=True)
        obs_ds = create_resizable_dataset(h5, "observation", (obs_dim,), np.float32, chunks=True)
        qpos_ds = create_resizable_dataset(h5, "qpos", (qpos_dim,), np.float32, chunks=True)
        qvel_ds = create_resizable_dataset(h5, "qvel", (qvel_dim,), np.float32, chunks=True)
        control_ds = create_resizable_dataset(h5, "control", (control_dim,), np.float32, chunks=True)
        effector_pos_ds = create_resizable_dataset(h5, "effector_pos", (3,), np.float32, chunks=True)
        effector_yaw_ds = create_resizable_dataset(h5, "effector_yaw", (1,), np.float32, chunks=True)
        gripper_opening_ds = create_resizable_dataset(h5, "gripper_opening", (1,), np.float32, chunks=True)
        gripper_contact_ds = create_resizable_dataset(h5, "gripper_contact", (1,), np.float32, chunks=True)
        block_pos_ds = create_resizable_dataset(h5, "block_pos", (3,), np.float32, chunks=True)
        block_quat_ds = create_resizable_dataset(h5, "block_quat", (4,), np.float32, chunks=True)
        block_yaw_ds = create_resizable_dataset(h5, "block_yaw", (1,), np.float32, chunks=True)
        target_block_pos_ds = create_resizable_dataset(h5, "target_block_pos", (3,), np.float32, chunks=True)
        target_block_yaw_ds = create_resizable_dataset(h5, "target_block_yaw", (1,), np.float32, chunks=True)
        time_ds = create_resizable_dataset(h5, "time", (1,), np.float32, chunks=True)
        episode_idx_ds = create_resizable_dataset(h5, "episode_idx", (), np.int64, chunks=True)
        step_idx_ds = create_resizable_dataset(h5, "step_idx", (), np.int64, chunks=True)

        progress_total = args.target_transitions if args.target_transitions is not None else args.num_trajectories
        progress_desc = "Collecting transitions" if args.target_transitions is not None else "Collecting trajectories"
        progress_unit = "step" if args.target_transitions is not None else "traj"

        with tqdm(total=progress_total, desc=progress_desc, unit=progress_unit) as progress:
            while should_continue(args, len(step_counts), int(np.sum(step_counts, dtype=np.int64))):
                trajectory_seed = args.seed + seed_offset
                seed_offset += 1

                trajectory, total_reward, terminated, truncated, reached_goal_once = collect_trajectory(
                    env=env,
                    oracle=oracle,
                    trajectory_seed=trajectory_seed,
                    camera=args.camera,
                    save_depth=SAVE_DEPTH,
                    goal_threshold=args.goal_threshold,
                    goal_yaw_threshold=args.goal_yaw_threshold,
                    post_goal_steps=args.post_goal_steps,
                    mujoco_module=mujoco,
                    lie_module=lie,
                )
                num_actions = int(trajectory["action"].shape[0])
                if num_actions < args.min_steps:
                    skipped_short += 1
                    continue

                episode_idx = len(step_counts)
                video_path = video_dir / f"trajectory_{episode_idx:07d}.mp4"
                imageio.mimwrite(
                    video_path,
                    trajectory["pixels"],
                    fps=control_freq_hz,
                    quality=args.quality,
                    macro_block_size=1,
                )

                padded_actions = np.empty((trajectory["observation"].shape[0], ACTION_DIM), dtype=np.float32)
                padded_actions[:-1] = trajectory["action"]
                padded_actions[-1] = np.nan

                offset, _ = append_rows(pixels_ds, trajectory["pixels"])
                if depth_ds is not None:
                    append_rows(depth_ds, trajectory["depth"])
                append_rows(action_ds, padded_actions)
                append_rows(obs_ds, trajectory["observation"])
                append_rows(qpos_ds, trajectory["qpos"])
                append_rows(qvel_ds, trajectory["qvel"])
                append_rows(control_ds, trajectory["control"])
                append_rows(effector_pos_ds, trajectory["effector_pos"])
                append_rows(effector_yaw_ds, trajectory["effector_yaw"])
                append_rows(gripper_opening_ds, trajectory["gripper_opening"])
                append_rows(gripper_contact_ds, trajectory["gripper_contact"])
                append_rows(block_pos_ds, trajectory["block_pos"])
                append_rows(block_quat_ds, trajectory["block_quat"])
                append_rows(block_yaw_ds, trajectory["block_yaw"])
                append_rows(target_block_pos_ds, trajectory["target_block_pos"])
                append_rows(target_block_yaw_ds, trajectory["target_block_yaw"])
                append_rows(time_ds, trajectory["time"])
                append_rows(episode_idx_ds, np.full((trajectory["observation"].shape[0],), episode_idx, dtype=np.int64))
                append_rows(step_idx_ds, np.arange(trajectory["observation"].shape[0], dtype=np.int64))
                append_rows(ep_len_ds, np.asarray([trajectory["observation"].shape[0]], dtype=np.int64))
                append_rows(ep_offset_ds, np.asarray([offset], dtype=np.int64))
                append_rows(reward_ds, np.asarray([total_reward], dtype=np.float32))
                append_rows(seed_ds, np.asarray([trajectory_seed], dtype=np.int64))
                append_rows(terminated_ds, np.asarray([terminated], dtype=np.bool_))
                append_rows(truncated_ds, np.asarray([truncated], dtype=np.bool_))

                rewards.append(total_reward)
                step_counts.append(num_actions)
                terminated_flags.append(terminated)
                truncated_flags.append(truncated)
                failed_flags.append(not reached_goal_once)
                if truncated:
                    final_pos_error = float(
                        np.linalg.norm(trajectory["block_pos"][-1] - trajectory["target_block_pos"][-1])
                    )
                    final_raw_yaw_error = angular_distance(
                        float(trajectory["block_yaw"][-1][0]),
                        float(trajectory["target_block_yaw"][-1][0]),
                    )
                    final_symmetry_yaw_error = symmetry_aware_yaw_distance(
                        float(trajectory["block_yaw"][-1][0]),
                        float(trajectory["target_block_yaw"][-1][0]),
                    )
                    print(
                        f"Temporarily logging truncated episode: episode_idx={episode_idx}, "
                        f"trajectory_seed={trajectory_seed}, num_actions={num_actions}, "
                        f"final_pos_error={final_pos_error:.4f}, "
                        f"final_raw_yaw_error={final_raw_yaw_error:.4f}, "
                        f"final_symmetry_yaw_error={final_symmetry_yaw_error:.4f}"
                    )
                progress.update(num_actions if args.target_transitions is not None else 1)
                progress.set_postfix(
                    episodes=len(step_counts),
                    transitions=int(np.sum(step_counts, dtype=np.int64)),
                    fail=int(np.sum(failed_flags, dtype=np.int64)),
                )

        ep_len = np.asarray(ep_len_ds[:], dtype=np.int64)
        total_transitions = int(np.sum(step_counts, dtype=np.int64))
        h5.attrs["num_episodes"] = len(step_counts)
        h5.attrs["total_frames"] = int(pixels_ds.shape[0])
        h5.attrs["total_transitions"] = total_transitions
        h5.attrs["skipped_short_episodes"] = skipped_short
        h5.attrs["mean_reward"] = float(np.mean(rewards)) if rewards else 0.0
        h5.attrs["mean_episode_steps"] = float(np.mean(step_counts)) if step_counts else 0.0
        h5.attrs["usable_train_windows_default"] = valid_training_windows(ep_len)

    env.close()

    summary = {
        "output_path": str(output_path),
        "video_dir": str(video_dir),
        "num_episodes": len(step_counts),
        "total_transitions": int(np.sum(step_counts, dtype=np.int64)),
        "total_frames": int(np.sum(step_counts, dtype=np.int64) + len(step_counts)),
        "min_episode_steps": int(np.min(step_counts)) if step_counts else 0,
        "mean_episode_steps": float(np.mean(step_counts)) if step_counts else 0.0,
        "max_episode_steps": int(np.max(step_counts)) if step_counts else 0,
        "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
        "terminated_episodes": int(np.sum(terminated_flags, dtype=np.int64)),
        "truncated_episodes": int(np.sum(truncated_flags, dtype=np.int64)),
        "skipped_short_episodes": skipped_short,
        "usable_train_windows_default": valid_training_windows(np.asarray(step_counts, dtype=np.int64) + 1)
        if step_counts
        else 0,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
