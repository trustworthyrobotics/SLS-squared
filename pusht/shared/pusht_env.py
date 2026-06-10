from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_PUSHT_ENV_ID = "gym_pusht/PushT-v0"
PUSHT_AGENT_RADIUS = 15.0
PUSHT_TEE_SCALE = 30.0
PUSHT_WALL_MIN = 5.0
PUSHT_WALL_MAX = 506.0
PUSHT_WALL_RADIUS = 2.0


def _import_gymnasium():
    try:
        import gymnasium as gym
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("Install gymnasium in the active environment.") from exc
    return gym


def _ensure_env_package(env_id: str) -> None:
    package_name = env_id.split("/", maxsplit=1)[0] if "/" in env_id else None
    if package_name is None:
        return
    try:
        __import__(package_name)
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            f"Environment package '{package_name}' is not installed. "
            "For the default PushT env, install gym-pusht."
        ) from exc


def _import_pusht_no_target_env():
    import pygame
    import pymunk
    from gym_pusht.envs.pusht import PushTEnv
    from gym_pusht.envs.pymunk_override import DrawOptions

    class PushTNoTargetEnv(PushTEnv):
        def _setup(self):
            self.space = pymunk.Space()
            self.space.gravity = 0, 0
            self.space.damping = self.damping if self.damping is not None else 0.0
            self.teleop = False

            walls = [
                self.add_segment(self.space, (5, 506), (5, 5), 2),
                self.add_segment(self.space, (5, 5), (506, 5), 2),
                self.add_segment(self.space, (506, 5), (506, 506), 2),
                self.add_segment(self.space, (5, 506), (506, 506), 2),
            ]
            self.space.add(*walls)

            self.agent = self.add_circle(self.space, (256, 400), 15)
            self.block, self._block_shapes = self.add_tee(self.space, (256, 300), 0)
            self.goal_pose = np.array([256, 256, np.pi / 4])
            if self.block_cog is not None:
                self.block.center_of_gravity = self.block_cog
            self.n_contact_points = 0

        def _draw(self):
            screen = pygame.Surface((512, 512))
            screen.fill((255, 255, 255))
            draw_options = DrawOptions(screen)
            self.space.debug_draw(draw_options)
            return screen

    return PushTNoTargetEnv


def make_pusht_env(
    env_id: str = DEFAULT_PUSHT_ENV_ID,
    *,
    obs_type: str = "pixels_agent_pos",
    render_mode: str = "rgb_array",
    max_episode_steps: int = 300,
    observation_width: int | None = None,
    observation_height: int | None = None,
    visualization_width: int | None = 384,
    visualization_height: int | None = 384,
    hide_target: bool = False,
):
    """Create a PushT environment with the repo's shared defaults."""
    if hide_target:
        if env_id != DEFAULT_PUSHT_ENV_ID:
            raise ValueError(f"hide_target=True only supports env_id={DEFAULT_PUSHT_ENV_ID!r}, got {env_id!r}.")

        env_cls = _import_pusht_no_target_env()
        env = env_cls(
            obs_type=obs_type,
            render_mode=render_mode,
            observation_width=observation_width,
            observation_height=observation_height,
            visualization_width=visualization_width if visualization_width is not None else observation_width,
            visualization_height=visualization_height if visualization_height is not None else observation_height,
        )
        env.reset(seed=0)
        return env

    gym = _import_gymnasium()
    _ensure_env_package(env_id)

    kwargs = {
        "obs_type": obs_type,
        "render_mode": render_mode,
        "max_episode_steps": max_episode_steps,
    }
    if observation_width is not None:
        kwargs["observation_width"] = observation_width
    if observation_height is not None:
        kwargs["observation_height"] = observation_height
    if visualization_width is not None:
        kwargs["visualization_width"] = visualization_width
    if visualization_height is not None:
        kwargs["visualization_height"] = visualization_height

    try:
        return gym.make(env_id, disable_env_checker=True, **kwargs)
    except TypeError:
        for key in (
            "observation_width",
            "observation_height",
            "visualization_width",
            "visualization_height",
        ):
            kwargs.pop(key, None)
        return gym.make(env_id, disable_env_checker=True, **kwargs)


def make_no_target_env(*, height: int, width: int, max_episode_steps: int = 300):
    return make_pusht_env(
        obs_type="pixels",
        render_mode="rgb_array",
        max_episode_steps=max_episode_steps,
        observation_width=width,
        observation_height=height,
        visualization_width=width,
        visualization_height=height,
        hide_target=True,
    )


def set_pusht_state(env: Any, state: np.ndarray) -> None:
    base_env = getattr(env, "unwrapped", env)
    base_env.agent.velocity = [0.0, 0.0]
    base_env.block.velocity = [0.0, 0.0]
    base_env.block.angular_velocity = 0.0
    state = np.asarray(state, dtype=np.float64)
    base_env.agent.position = list(state[:2])
    # The T body has an offset center of gravity, so setting position and then
    # angle shifts the rendered pose. Apply angle first, then place the body.
    base_env.block.angle = float(state[4])
    base_env.block.position = list(state[2:4])
    base_env.space.step(base_env.dt)
    base_env.agent.velocity = [float(state[5]), float(state[6])] if state.shape[0] >= 7 else [0.0, 0.0]
    base_env.block.velocity = [0.0, 0.0]
    base_env.block.angular_velocity = 0.0
    base_env._last_action = None


def reset_pusht_env_to_state(env: Any, state: np.ndarray) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    set_pusht_state(base_env, np.asarray(state, dtype=np.float64))
    return np.asarray(base_env._render(visualize=False), dtype=np.uint8)


def get_pusht_goal_pose(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    goal_pose = getattr(base_env, "goal_pose", None)
    if goal_pose is None:
        raise AttributeError("PushT env does not expose goal_pose.")
    return np.asarray(goal_pose, dtype=np.float32).reshape(-1)[:3]


def _rotation_matrix(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s], [s, c]], dtype=np.float64)


def make_obstacle_pusht_init_state(
    goal_pose: np.ndarray,
    *,
    block_offset: float = 120.0,
    tilt_deg: float = 10.0,
    pusher_face_offset: float = PUSHT_AGENT_RADIUS,
) -> np.ndarray:
    """Place the T above its goal and the pusher centered on the T's top face."""
    goal_pose = np.asarray(goal_pose, dtype=np.float64).reshape(-1)
    if goal_pose.shape[0] < 3:
        raise ValueError(f"Expected goal_pose with at least 3 values, got shape {goal_pose.shape}.")

    goal_theta = float(goal_pose[2])
    tilt_rad = float(np.deg2rad(tilt_deg))
    block_theta = goal_theta + tilt_rad
    goal_rotation = _rotation_matrix(goal_theta)
    block_offset = float(block_offset)
    horizontal_offset = block_offset * float(np.tan(tilt_rad))
    block_xy = goal_pose[:2] + goal_rotation @ np.asarray([horizontal_offset, -block_offset], dtype=np.float64)

    block_rotation = _rotation_matrix(block_theta)
    agent_xy = block_xy + block_rotation @ np.asarray([0.0, -float(pusher_face_offset)], dtype=np.float64)

    min_agent_coord = PUSHT_WALL_MIN + PUSHT_WALL_RADIUS + PUSHT_AGENT_RADIUS
    max_agent_coord = PUSHT_WALL_MAX - PUSHT_WALL_RADIUS - PUSHT_AGENT_RADIUS
    agent_xy = np.clip(agent_xy, min_agent_coord, max_agent_coord)

    return np.asarray(
        [agent_xy[0], agent_xy[1], block_xy[0], block_xy[1], block_theta, 0.0, 0.0],
        dtype=np.float64,
    )


def reset_pusht_env_to_obstacle_init(
    env: Any,
    *,
    block_offset: float = 120.0,
    tilt_deg: float = 10.0,
    pusher_face_offset: float = PUSHT_AGENT_RADIUS,
) -> tuple[np.ndarray, np.ndarray]:
    goal_pose = get_pusht_goal_pose(env)
    state = make_obstacle_pusht_init_state(
        goal_pose,
        block_offset=block_offset,
        tilt_deg=tilt_deg,
        pusher_face_offset=pusher_face_offset,
    )
    pixels = reset_pusht_env_to_state(env, state)
    return pixels, state


def get_pusht_block_pose(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    return np.asarray([base_env.block.position.x, base_env.block.position.y, base_env.block.angle], dtype=np.float32)


def get_pusht_agent_pos(env: Any) -> np.ndarray:
    base_env = getattr(env, "unwrapped", env)
    return np.asarray([base_env.agent.position.x, base_env.agent.position.y], dtype=np.float32)
