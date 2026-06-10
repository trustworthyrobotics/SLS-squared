from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time

import imageio.v2 as imageio
import mujoco
import mujoco.viewer
import numpy as np
from scipy.interpolate import CubicSpline


SHARED_DIR = Path(__file__).resolve().parent
LAB_SCENE_XML = SHARED_DIR / "lab_scene_base.xml"
IIWA_MODEL_XML = SHARED_DIR / "kuka_iiwa_14" / "iiwa14_base.xml"
DEFAULT_VIDEO_PATH = SHARED_DIR / "lab_env.mp4"

TABLE_SIZE = np.array([0.50, 0.75, 0.375], dtype=float)
TABLE_CENTER = np.array([0.0, 0.0, TABLE_SIZE[2]], dtype=float)
TABLE_TOP_Z = float(2.0 * TABLE_SIZE[2])

EE_FIXED_ROT = np.diag([1.0, -1.0, -1.0])
NOMINAL_TASK_STATE = np.array([0.08, 0.98, 0.42], dtype=float)
TASK_REACH_BOUNDS = (-0.1, 0.3)
TASK_HEIGHT_BOUNDS = (1.16, 1.35)
TASK_WIDTH_BOUNDS = (0.2, 0.75)
LEFT_ARM_SEED = np.array([0.42, 0.95, 0.0, -1.55, 0.0, 0.75, 0.0], dtype=float)
RIGHT_ARM_SEED = np.array([-0.42, 0.95, 0.0, -1.55, 0.0, 0.75, 0.0], dtype=float)
DEFAULT_RENDER_WIDTH = 224
DEFAULT_RENDER_HEIGHT = 224
DEFAULT_RENDER_FPS = 20
DEFAULT_VIEWER_LOOKAT = np.array([0.0, 0.0, 0.95], dtype=float)
DEFAULT_VIEWER_DISTANCE = 1.6
DEFAULT_VIEWER_AZIMUTH = 0.0
DEFAULT_VIEWER_ELEVATION = -20.0
RANDOM_CUBIC_SPLINE_MODE = "RANDOM_CUBIC_SPLINE"
RANDOM_WAYPOINT_MODE = "RANDOM_WAYPOINT"
WS_EDGE_SAMPLE_MODE = "WS_EDGE_SAMPLE"
MODE = RANDOM_CUBIC_SPLINE_MODE


@dataclass(frozen=True)
class TaskState:
    reach: float
    height: float
    width: float

    def as_array(self) -> np.ndarray:
        return np.array([self.reach, self.height, self.width], dtype=float)

    @classmethod
    def from_array(cls, values: np.ndarray | list[float] | tuple[float, float, float]) -> TaskState:
        array = np.asarray(values, dtype=float)
        if array.shape != (3,):
            raise ValueError(f"Expected a 3D task state, got shape {array.shape}")
        return cls(reach=float(array[0]), height=float(array[1]), width=float(array[2]))


@dataclass(frozen=True)
class TaskBounds:
    reach: tuple[float, float] = TASK_REACH_BOUNDS
    height: tuple[float, float] = TASK_HEIGHT_BOUNDS
    width: tuple[float, float] = TASK_WIDTH_BOUNDS

    def clip(self, state: TaskState | np.ndarray | list[float]) -> TaskState:
        if isinstance(state, TaskState):
            values = state.as_array()
        else:
            values = np.asarray(state, dtype=float)
        lower = np.array([self.reach[0], self.height[0], self.width[0]], dtype=float)
        upper = np.array([self.reach[1], self.height[1], self.width[1]], dtype=float)
        return TaskState.from_array(np.clip(values, lower, upper))


@dataclass(frozen=True)
class RopeSpec:
    segments: int = 22
    radius: float = 0.008
    density: float = 180.0
    damping: float = 0.6
    armature: float = 0.01
    friction: tuple[float, float, float] = (0.9, 0.03, 0.01)
    sag: float = 0.03
    slack_scale: float = 1.10
    rgba: tuple[float, float, float, float] = (0.58, 0.44, 0.27, 1.0)
    proxy_mass: float = 0.002
    proxy_stiffness: float = 4000.0
    proxy_damping: float = 40.0


@dataclass(frozen=True)
class BaseEnvConfig:
    task_bounds: TaskBounds = field(default_factory=TaskBounds)
    nominal_task_state: TaskState = field(default_factory=lambda: TaskState.from_array(NOMINAL_TASK_STATE))
    rope_spec: RopeSpec = field(default_factory=RopeSpec)
    enable_proxy_rope: bool = False
    asset_extra_xml: str = ""
    worldbody_extra_xml: str = ""
    offscreen_width: int = DEFAULT_RENDER_WIDTH
    offscreen_height: int = DEFAULT_RENDER_HEIGHT


@dataclass(frozen=True)
class ControllerConfig:
    task_bounds: TaskBounds = field(default_factory=TaskBounds)
    target_rot: np.ndarray = field(default_factory=lambda: EE_FIXED_ROT.copy())
    position_weight: float = 1.0
    orientation_weight: float = 0.35
    posture_weight: float = 0.025
    damping: float = 1e-3
    step_scale: float = 0.7
    max_iterations: int = 80


def get_lab_scene_xml_path() -> Path:
    return LAB_SCENE_XML


def format_vec(values: np.ndarray | tuple[float, ...] | list[float]) -> str:
    return " ".join(f"{float(value):.6f}" for value in values)


def build_task_targets(task_state: TaskState) -> tuple[np.ndarray, np.ndarray]:
    left = np.array([task_state.reach, 0.5 * task_state.width, task_state.height], dtype=float)
    right = np.array([task_state.reach, -0.5 * task_state.width, task_state.height], dtype=float)
    return left, right


def rope_rest_length(task_bounds: TaskBounds, rope_spec: RopeSpec) -> float:
    return rope_spec.slack_scale * task_bounds.width[1]


def build_rope_tendon_xml(spec: RopeSpec, task_bounds: TaskBounds) -> str:
    return (
        f'    <spatial name="rope_tendon" width="{spec.radius:.6f}" '
        f'range="0 {rope_rest_length(task_bounds, spec):.6f}" limited="true" '
        'stiffness="0" damping="0" rgba="0.85 0.2 0.2 1">'
        '\n      <site site="arm1_attachment_site"/>'
        '\n      <site site="arm2_attachment_site"/>'
        "\n    </spatial>"
    )


def proxy_segment_count(spec: RopeSpec) -> int:
    return max(2, int(spec.segments))


def proxy_node_count(spec: RopeSpec) -> int:
    return proxy_segment_count(spec) - 1


def proxy_segment_rest_length(task_bounds: TaskBounds, spec: RopeSpec) -> float:
    return rope_rest_length(task_bounds, spec) / float(proxy_segment_count(spec))


def build_proxy_curve_points(left: np.ndarray, right: np.ndarray, *, node_count: int, sag_depth: float) -> np.ndarray:
    points: list[np.ndarray] = [left.astype(np.float64).copy()]
    for index in range(node_count):
        t = float(index + 1) / float(node_count + 1)
        position = (1.0 - t) * left + t * right
        position[2] -= sag_depth * 4.0 * t * (1.0 - t)
        points.append(position.astype(np.float64, copy=False))
    points.append(right.astype(np.float64).copy())
    return np.stack(points, axis=0)


def polyline_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def solve_proxy_sag_depth(
    left: np.ndarray,
    right: np.ndarray,
    *,
    node_count: int,
    target_length: float,
    min_sag_depth: float,
) -> float:
    chord = float(np.linalg.norm(right - left))
    if target_length <= chord + 1e-9:
        return 0.0

    lower = max(0.0, float(min_sag_depth))
    lower_points = build_proxy_curve_points(left, right, node_count=node_count, sag_depth=lower)
    if polyline_length(lower_points) >= target_length:
        return lower

    upper = max(lower if lower > 0.0 else 0.01, 0.01)
    for _ in range(64):
        upper_points = build_proxy_curve_points(left, right, node_count=node_count, sag_depth=upper)
        if polyline_length(upper_points) >= target_length:
            break
        upper *= 2.0

    for _ in range(80):
        mid = 0.5 * (lower + upper)
        mid_points = build_proxy_curve_points(left, right, node_count=node_count, sag_depth=mid)
        if polyline_length(mid_points) < target_length:
            lower = mid
        else:
            upper = mid
    return upper


def build_proxy_rope_xml(spec: RopeSpec, task_bounds: TaskBounds) -> tuple[str, str]:
    width = max(0.001, 0.35 * spec.radius)
    site_size = max(0.0005, 0.25 * width)
    body_lines = [
        '    <body name="proxy_left_anchor" mocap="true" pos="0 0 0">',
        f'      <site name="proxy_left_anchor_site" pos="0 0 0" size="{site_size:.6f}" rgba="0 0 0 0"/>',
        "    </body>",
        '    <body name="proxy_right_anchor" mocap="true" pos="0 0 0">',
        f'      <site name="proxy_right_anchor_site" pos="0 0 0" size="{site_size:.6f}" rgba="0 0 0 0"/>',
        "    </body>",
    ]

    for index in range(proxy_node_count(spec)):
        node_name = f"proxy_node_{index:02d}"
        joint_name = f"{node_name}_joint"
        site_name = f"{node_name}_site"
        geom_name = f"{node_name}_geom"
        body_lines.extend(
            [
                f'    <body name="{node_name}" pos="0 0 0">',
                f'      <freejoint name="{joint_name}"/>',
                (
                    f'      <geom name="{geom_name}" type="sphere" size="{width:.6f}" '
                    f'mass="{spec.proxy_mass:.6f}" rgba="0 0 0 0" contype="0" conaffinity="0"/>'
                ),
                f'      <site name="{site_name}" pos="0 0 0" size="{site_size:.6f}" rgba="0 0 0 0"/>',
                "    </body>",
            ]
        )

    point_sites = ["proxy_left_anchor_site"]
    point_sites.extend(f"proxy_node_{index:02d}_site" for index in range(proxy_node_count(spec)))
    point_sites.append("proxy_right_anchor_site")

    tendon_lines = []
    segment_rest = proxy_segment_rest_length(task_bounds, spec)
    for index, (site_a, site_b) in enumerate(zip(point_sites[:-1], point_sites[1:], strict=True)):
        tendon_lines.extend(
            [
                (
                    f'    <spatial name="proxy_segment_{index:02d}" width="{width:.6f}" '
                    f'range="0 {segment_rest:.6f}" limited="true" '
                    f'stiffness="{spec.proxy_stiffness:.6f}" damping="{spec.proxy_damping:.6f}" '
                    'rgba="0 0 0 0">'
                ),
                f'      <site site="{site_a}"/>',
                f'      <site site="{site_b}"/>',
                "    </spatial>",
            ]
        )

    return "\n".join(body_lines), "\n".join(tendon_lines)


def build_lab_scene_xml(
    base_config: BaseEnvConfig | None = None,
) -> str:
    config = BaseEnvConfig() if base_config is None else base_config
    rope_tendon_xml = build_rope_tendon_xml(config.rope_spec, config.task_bounds)
    proxy_body_xml = ""
    proxy_tendon_xml = ""
    if config.enable_proxy_rope:
        proxy_body_xml, proxy_tendon_xml = build_proxy_rope_xml(config.rope_spec, config.task_bounds)
    asset_extra_xml = config.asset_extra_xml.rstrip()
    worldbody_extra_xml = config.worldbody_extra_xml.rstrip()
    return f"""
<mujoco model="lab_scene_control">
  <compiler angle="radian"/>

  <statistic center="0 0 0.9" extent="2.5"/>

  <option gravity="0 0 -9.81" timestep="0.002" integrator="implicitfast"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-20" offwidth="{int(config.offscreen_width)}" offheight="{int(config.offscreen_height)}"/>
  </visual>

  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge"
      rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    <model name="kuka_iiwa_model" file="{IIWA_MODEL_XML.as_posix()}"/>
{asset_extra_xml}
  </asset>

  <worldbody>
    <light name="main_light" pos="0 0 2.5" dir="0 0 -1"/>

    <geom name="ground" type="plane" size="3 3 0.1" material="groundplane"/>

    <geom name="table"
          type="box"
          pos="0 0 0.375"
          size="0.50 0.75 0.375"
          rgba="0.75 0.75 0.75 1"/>

    <geom name="arm1_plate"
          type="box"
          pos="0.40 0.65 0.75645"
          size="0.10 0.10 0.00645"
          rgba="0.3 0.3 0.3 1"/>

    <body name="arm1_mount" pos="0.40 0.65 0.7629" euler="0 0 180">
      <attach model="kuka_iiwa_model" body="base" prefix="arm1_"/>
    </body>

    <geom name="arm2_plate"
          type="box"
          pos="0.40 -0.65 0.75645"
          size="0.10 0.10 0.00645"
          rgba="0.3 0.3 0.3 1"/>

    <body name="arm2_mount" pos="0.40 -0.65 0.7629" euler="0 0 180">
      <attach model="kuka_iiwa_model" body="base" prefix="arm2_"/>
    </body>

    <camera name="video_cam"
            mode="fixed"
            pos="-1.5 0.0 1.6"
            xyaxes="0 -1 0  0.35 0 0.94"/>
    <!-- pos = Front/Back, Left/Right, Up/Down -->

    <camera name="ceiling_cam"
            mode="fixed"
            pos="0 0 3.0"
            xyaxes="1 0 0  0 1 0"/>
{worldbody_extra_xml}
{proxy_body_xml}
  </worldbody>

  <tendon>
{rope_tendon_xml}
{proxy_tendon_xml}
  </tendon>
</mujoco>
""".strip()


def load_model(scene_xml: str | None = None) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(scene_xml or build_lab_scene_xml())


def make_data(model: mujoco.MjModel) -> mujoco.MjData:
    return mujoco.MjData(model)


def quat_conjugate(quat: np.ndarray) -> np.ndarray:
    result = quat.copy()
    result[1:] *= -1.0
    return result


def orientation_error(target_rot: np.ndarray, current_rot: np.ndarray) -> np.ndarray:
    target_quat = np.zeros(4, dtype=float)
    current_quat = np.zeros(4, dtype=float)
    quat_error = np.zeros(4, dtype=float)
    rotvec = np.zeros(3, dtype=float)

    mujoco.mju_mat2Quat(target_quat, target_rot.reshape(-1))
    mujoco.mju_mat2Quat(current_quat, current_rot.reshape(-1))
    if np.dot(target_quat, current_quat) < 0.0:
        current_quat *= -1.0
    mujoco.mju_mulQuat(quat_error, target_quat, quat_conjugate(current_quat))
    mujoco.mju_quat2Vel(rotvec, quat_error, 1.0)
    return rotvec


def clip_joint_targets(joint_targets: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    return np.clip(joint_targets, lower, upper)


@dataclass
class JointPositionController:
    lower: np.ndarray
    upper: np.ndarray
    target: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.target = np.zeros_like(self.lower)

    def set_target(self, joint_targets: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        array = np.asarray(joint_targets, dtype=float)
        if array.shape != self.lower.shape:
            raise ValueError(f"Expected joint targets with shape {self.lower.shape}, got {array.shape}")
        self.target = clip_joint_targets(array, self.lower, self.upper)
        return self.target.copy()

    def apply_delta(self, delta: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        array = np.asarray(delta, dtype=float)
        if array.shape != self.lower.shape:
            raise ValueError(f"Expected joint delta with shape {self.lower.shape}, got {array.shape}")
        return self.set_target(self.target + array)


@dataclass
class SymmetricTaskController:
    env: LabEnv
    config: ControllerConfig = field(default_factory=ControllerConfig)
    desired_state: TaskState = field(default_factory=lambda: TaskState.from_array(NOMINAL_TASK_STATE))

    def clamp(self, state: TaskState | np.ndarray | list[float]) -> TaskState:
        return self.config.task_bounds.clip(state)

    def set_target(self, state: TaskState | np.ndarray | list[float]) -> TaskState:
        self.desired_state = self.clamp(state)
        return self.desired_state

    def apply_delta(self, delta: np.ndarray | list[float] | tuple[float, float, float]) -> TaskState:
        delta_array = np.asarray(delta, dtype=float)
        if delta_array.shape != (3,):
            raise ValueError(f"Expected a 3D task-space delta, got shape {delta_array.shape}")
        return self.set_target(self.desired_state.as_array() + delta_array)

    def solve(self, state: TaskState | None = None) -> np.ndarray:
        desired = self.desired_state if state is None else self.set_target(state)
        left_target, right_target = build_task_targets(desired)
        current = self.env.get_arm_joint_positions()
        left = self._solve_single_arm(
            joint_qposadr=self.env.arm1_qposadr,
            joint_dofadr=self.env.arm1_dofadr,
            site_id=self.env.arm1_site_id,
            target_pos=left_target,
            seed=current[:7] if np.any(current[:7]) else LEFT_ARM_SEED,
            posture_seed=LEFT_ARM_SEED,
        )
        right = self._solve_single_arm(
            joint_qposadr=self.env.arm2_qposadr,
            joint_dofadr=self.env.arm2_dofadr,
            site_id=self.env.arm2_site_id,
            target_pos=right_target,
            seed=current[7:] if np.any(current[7:]) else RIGHT_ARM_SEED,
            posture_seed=RIGHT_ARM_SEED,
        )
        return np.concatenate([left, right])

    def _solve_single_arm(
        self,
        joint_qposadr: np.ndarray,
        joint_dofadr: np.ndarray,
        site_id: int,
        target_pos: np.ndarray,
        seed: np.ndarray,
        posture_seed: np.ndarray,
    ) -> np.ndarray:
        q = seed.astype(float).copy()
        jacp = np.zeros((3, self.env.model.nv), dtype=float)
        jacr = np.zeros((3, self.env.model.nv), dtype=float)

        for _ in range(self.config.max_iterations):
            self.env.set_arm_joint_positions(
                np.concatenate(
                    [
                        q if np.array_equal(joint_qposadr, self.env.arm1_qposadr) else self.env.get_arm_joint_positions()[:7],
                        q if np.array_equal(joint_qposadr, self.env.arm2_qposadr) else self.env.get_arm_joint_positions()[7:],
                    ]
                )
            )
            current_pos = self.env.data.site_xpos[site_id].copy()
            current_rot = self.env.data.site_xmat[site_id].reshape(3, 3).copy()
            pos_err = target_pos - current_pos
            rot_err = orientation_error(self.config.target_rot, current_rot)
            if np.linalg.norm(pos_err) < 2e-4 and np.linalg.norm(rot_err) < 2e-3:
                break

            jacp.fill(0.0)
            jacr.fill(0.0)
            mujoco.mj_jacSite(self.env.model, self.env.data, jacp, jacr, site_id)
            J_pos = jacp[:, joint_dofadr]
            J_rot = jacr[:, joint_dofadr]
            J = np.vstack(
                [
                    self.config.position_weight * J_pos,
                    self.config.orientation_weight * J_rot,
                    self.config.posture_weight * np.eye(len(joint_qposadr)),
                ]
            )
            err = np.concatenate(
                [
                    self.config.position_weight * pos_err,
                    self.config.orientation_weight * rot_err,
                    self.config.posture_weight * (posture_seed - q),
                ]
            )
            lhs = J @ J.T + self.config.damping * np.eye(J.shape[0], dtype=float)
            dq = J.T @ np.linalg.solve(lhs, err)
            q = clip_joint_targets(
                q + self.config.step_scale * dq,
                self.env.joint_lower[joint_qposadr],
                self.env.joint_upper[joint_qposadr],
            )

        return q


@dataclass
class LabEnv:
    base_config: BaseEnvConfig = field(default_factory=BaseEnvConfig)
    controller_config: ControllerConfig | None = None
    model: mujoco.MjModel = field(init=False)
    data: mujoco.MjData = field(init=False)
    joint_lower: np.ndarray = field(init=False)
    joint_upper: np.ndarray = field(init=False)
    arm1_qposadr: np.ndarray = field(init=False)
    arm2_qposadr: np.ndarray = field(init=False)
    arm1_dofadr: np.ndarray = field(init=False)
    arm2_dofadr: np.ndarray = field(init=False)
    arm1_site_id: int = field(init=False)
    arm2_site_id: int = field(init=False)
    proxy_left_mocap_id: int = field(init=False)
    proxy_right_mocap_id: int = field(init=False)
    proxy_node_qposadr: np.ndarray = field(init=False)
    proxy_node_qveladr: np.ndarray = field(init=False)
    proxy_node_site_ids: np.ndarray = field(init=False)
    joint_controller: JointPositionController = field(init=False)
    task_controller: SymmetricTaskController = field(init=False)

    def __post_init__(self) -> None:
        if self.controller_config is None:
            self.controller_config = ControllerConfig(task_bounds=self.base_config.task_bounds)
        self.model = load_model(build_lab_scene_xml(self.base_config))
        self.data = make_data(self.model)
        self.arm1_site_id = self.model.site("arm1_attachment_site").id
        self.arm2_site_id = self.model.site("arm2_attachment_site").id
        self.proxy_left_mocap_id = -1
        self.proxy_right_mocap_id = -1
        self.proxy_node_qposadr = np.zeros((0,), dtype=int)
        self.proxy_node_qveladr = np.zeros((0,), dtype=int)
        self.proxy_node_site_ids = np.zeros((0,), dtype=int)
        if self.base_config.enable_proxy_rope:
            self.proxy_left_mocap_id = int(self.model.body_mocapid[self.model.body("proxy_left_anchor").id])
            self.proxy_right_mocap_id = int(self.model.body_mocapid[self.model.body("proxy_right_anchor").id])
            self.proxy_node_qposadr = np.asarray(
                [self.model.joint(f"proxy_node_{index:02d}_joint").qposadr[0] for index in range(proxy_node_count(self.rope_spec))],
                dtype=int,
            )
            self.proxy_node_qveladr = np.asarray(
                [self.model.joint(f"proxy_node_{index:02d}_joint").dofadr[0] for index in range(proxy_node_count(self.rope_spec))],
                dtype=int,
            )
            self.proxy_node_site_ids = np.asarray(
                [self.model.site(f"proxy_node_{index:02d}_site").id for index in range(proxy_node_count(self.rope_spec))],
                dtype=int,
            )
        self.arm1_qposadr, self.arm1_dofadr = self._joint_addresses("arm1_joint")
        self.arm2_qposadr, self.arm2_dofadr = self._joint_addresses("arm2_joint")
        self.joint_lower = np.zeros(self.model.nq, dtype=float)
        self.joint_upper = np.zeros(self.model.nq, dtype=float)
        self.joint_lower[self.arm1_qposadr] = self._joint_limits(self.arm1_qposadr, lower=True)
        self.joint_lower[self.arm2_qposadr] = self._joint_limits(self.arm2_qposadr, lower=True)
        self.joint_upper[self.arm1_qposadr] = self._joint_limits(self.arm1_qposadr, lower=False)
        self.joint_upper[self.arm2_qposadr] = self._joint_limits(self.arm2_qposadr, lower=False)

        actuator_lower = self.model.actuator_ctrlrange[:, 0].copy()
        actuator_upper = self.model.actuator_ctrlrange[:, 1].copy()
        self.joint_controller = JointPositionController(actuator_lower, actuator_upper)
        self.task_controller = SymmetricTaskController(
            self,
            config=self.controller_config,
            desired_state=self.base_config.nominal_task_state,
        )
        self.reset()

    @property
    def task_bounds(self) -> TaskBounds:
        return self.base_config.task_bounds

    @property
    def rope_spec(self) -> RopeSpec:
        return self.base_config.rope_spec

    @property
    def nominal_state(self) -> TaskState:
        return self.base_config.nominal_task_state

    def _joint_addresses(self, prefix: str) -> tuple[np.ndarray, np.ndarray]:
        qposadr: list[int] = []
        dofadr: list[int] = []
        for index in range(1, 8):
            joint = self.model.joint(f"{prefix}{index}")
            qposadr.append(joint.qposadr[0])
            dofadr.append(joint.dofadr[0])
        return np.asarray(qposadr, dtype=int), np.asarray(dofadr, dtype=int)

    def _joint_limits(self, qposadr: np.ndarray, *, lower: bool) -> np.ndarray:
        limits = []
        joint_names = [self.model.joint(i).name for i in range(self.model.njnt)]
        joint_map = {self.model.joint(name).qposadr[0]: self.model.joint(name) for name in joint_names}
        for adr in qposadr:
            joint = joint_map[int(adr)]
            if lower:
                limits.append(joint.range[0])
            else:
                limits.append(joint.range[1])
        return np.asarray(limits, dtype=float)

    def get_arm_joint_positions(self) -> np.ndarray:
        qpos = np.zeros(14, dtype=float)
        qpos[:7] = self.data.qpos[self.arm1_qposadr]
        qpos[7:] = self.data.qpos[self.arm2_qposadr]
        return qpos

    def set_arm_joint_positions(self, arm_qpos: np.ndarray | list[float] | tuple[float, ...]) -> None:
        qpos = np.asarray(arm_qpos, dtype=float)
        if qpos.shape != (14,):
            raise ValueError(f"Expected 14 arm joint positions, got shape {qpos.shape}")
        self.data.qpos[self.arm1_qposadr] = qpos[:7]
        self.data.qpos[self.arm2_qposadr] = qpos[7:]
        self.data.ctrl[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        if self.base_config.enable_proxy_rope:
            self._sync_proxy_anchors()
            self._initialize_proxy_rope()
            mujoco.mj_forward(self.model, self.data)

    def reset(self, task_state: TaskState | None = None) -> mujoco.MjData:
        mujoco.mj_resetData(self.model, self.data)
        desired = self.nominal_state if task_state is None else self.task_bounds.clip(task_state)
        joint_targets = self.task_controller.solve(desired)
        self.joint_controller.set_target(joint_targets)
        self.set_arm_joint_positions(joint_targets)
        self.task_controller.set_target(desired)
        self.data.qvel[:] = 0.0
        if self.base_config.enable_proxy_rope:
            self._sync_proxy_anchors()
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def set_low_level_control(self, joint_targets: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return self.joint_controller.set_target(joint_targets)

    def apply_low_level_delta(self, joint_delta: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
        return self.joint_controller.apply_delta(joint_delta)

    def set_task_target(self, task_state: TaskState | np.ndarray | list[float]) -> TaskState:
        desired = self.task_controller.set_target(task_state)
        joint_targets = self.task_controller.solve(desired)
        self.joint_controller.set_target(joint_targets)
        return desired

    def apply_task_delta(self, delta: np.ndarray | list[float] | tuple[float, float, float]) -> TaskState:
        desired = self.task_controller.apply_delta(delta)
        joint_targets = self.task_controller.solve(desired)
        self.joint_controller.set_target(joint_targets)
        return desired

    def step(self, nstep: int = 1) -> mujoco.MjData:
        self.data.ctrl[:] = self.joint_controller.target
        for _ in range(nstep):
            if self.base_config.enable_proxy_rope:
                self._sync_proxy_anchors()
            mujoco.mj_step(self.model, self.data)
        if self.base_config.enable_proxy_rope:
            self._sync_proxy_anchors()
            self._initialize_proxy_rope()
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def launch_viewer(self) -> None:
        mujoco.viewer.launch(self.model, self.data)

    def _sync_proxy_anchors(self) -> None:
        if not self.base_config.enable_proxy_rope:
            return
        left = self.data.site_xpos[self.arm1_site_id].copy()
        right = self.data.site_xpos[self.arm2_site_id].copy()
        self.data.mocap_pos[self.proxy_left_mocap_id] = left
        self.data.mocap_pos[self.proxy_right_mocap_id] = right
        self.data.mocap_quat[self.proxy_left_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.mocap_quat[self.proxy_right_mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)

    def _initialize_proxy_rope(self) -> None:
        if not self.base_config.enable_proxy_rope:
            return
        left = self.data.site_xpos[self.arm1_site_id].copy()
        right = self.data.site_xpos[self.arm2_site_id].copy()
        node_count = proxy_node_count(self.rope_spec)
        target_length = rope_rest_length(self.task_bounds, self.rope_spec)
        sag_depth = solve_proxy_sag_depth(
            left,
            right,
            node_count=node_count,
            target_length=target_length,
            min_sag_depth=self.rope_spec.sag,
        )
        curve_points = build_proxy_curve_points(left, right, node_count=node_count, sag_depth=sag_depth)
        for index, qposadr in enumerate(self.proxy_node_qposadr):
            position = curve_points[index + 1]
            self.data.qpos[qposadr : qposadr + 3] = position
            self.data.qpos[qposadr + 3 : qposadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        if self.proxy_node_qveladr.size > 0:
            qveladr = self.proxy_node_qveladr[0]
            span = 6 * node_count
            self.data.qvel[qveladr : qveladr + span] = 0.0

    def get_proxy_rope_points(self) -> np.ndarray:
        if not self.base_config.enable_proxy_rope:
            raise RuntimeError("Proxy rope is disabled for this environment.")
        left = self.data.site_xpos[self.arm1_site_id].copy()
        right = self.data.site_xpos[self.arm2_site_id].copy()
        points = [left]
        points.extend(self.data.site_xpos[site_id].copy() for site_id in self.proxy_node_site_ids)
        points.append(right)
        return np.stack(points, axis=0)

    def get_proxy_rope_midpoint(self) -> np.ndarray:
        if not self.base_config.enable_proxy_rope:
            raise RuntimeError("Proxy rope is disabled for this environment.")
        points = self.get_proxy_rope_points()
        midpoint_index = 0.5 * float(points.shape[0] - 1)
        lower_index = int(np.floor(midpoint_index))
        upper_index = int(np.ceil(midpoint_index))
        if lower_index == upper_index:
            return points[lower_index].copy()
        return (0.5 * (points[lower_index] + points[upper_index])).copy()

    def get_proxy_rope_midpoint_height(self) -> float:
        if not self.base_config.enable_proxy_rope:
            raise RuntimeError("Proxy rope is disabled for this environment.")
        return float(self.get_proxy_rope_midpoint()[2])


@dataclass
class RandomWaypointPolicy:
    bounds: TaskBounds
    segment_duration: float = 2.0
    num_waypoints: int = 6
    seed: int | None = None
    _rng: np.random.Generator = field(init=False)
    _waypoints: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        lower = np.array([self.bounds.reach[0], self.bounds.height[0], self.bounds.width[0]], dtype=float)
        upper = np.array([self.bounds.reach[1], self.bounds.height[1], self.bounds.width[1]], dtype=float)
        nominal = TaskState.from_array(NOMINAL_TASK_STATE).as_array()
        points = [nominal]
        for _ in range(self.num_waypoints - 2):
            points.append(self._rng.uniform(lower, upper))
        points.append(nominal)
        self._waypoints = np.asarray(points, dtype=float)

    def sample(self, t: float) -> TaskState:
        total_segments = len(self._waypoints) - 1
        local_t = np.clip(max(t, 0.0) / self.segment_duration, 0.0, float(total_segments))
        segment = min(int(local_t), total_segments - 1)
        alpha = min(local_t - segment, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        state = (1.0 - smooth) * self._waypoints[segment] + smooth * self._waypoints[segment + 1]
        return self.bounds.clip(state)


@dataclass
class RandomSplinePolicy:
    bounds: TaskBounds
    segment_duration: float = 2.0
    midpoint_inflation_scale: float = 0.08
    seed: int | None = None
    _rng: np.random.Generator = field(init=False)
    _control_points: np.ndarray = field(init=False)
    _spline: CubicSpline = field(init=False)

    def __post_init__(self) -> None:
        self._rng = np.random.default_rng(self.seed)
        self._control_points = self._sample_control_points()
        self._spline = CubicSpline(
            np.array([0.0, 0.5, 1.0], dtype=float),
            self._control_points,
            axis=0,
            bc_type=((1, np.zeros(3, dtype=float)), (1, np.zeros(3, dtype=float))),
        )

    def _sample_control_points(self) -> np.ndarray:
        lower = np.array([self.bounds.reach[0], self.bounds.height[0], self.bounds.width[0]], dtype=float)
        upper = np.array([self.bounds.reach[1], self.bounds.height[1], self.bounds.width[1]], dtype=float)
        extent = upper - lower
        inflated_lower = lower - self.midpoint_inflation_scale * extent
        inflated_upper = upper + self.midpoint_inflation_scale * extent
        start = self._rng.uniform(lower, upper)
        midpoint = self._rng.uniform(inflated_lower, inflated_upper)
        end = self._rng.uniform(lower, upper)
        return np.stack([start, midpoint, end], axis=0)

    def sample(self, t: float) -> TaskState:
        alpha = np.clip(max(t, 0.0) / self.segment_duration, 0.0, 1.0)
        state = np.asarray(self._spline(alpha), dtype=float)
        return self.bounds.clip(state)


@dataclass
class WorkspaceEdgeSamplePolicy:
    bounds: TaskBounds
    segment_duration: float = 1.5
    _segments: list[tuple[np.ndarray, np.ndarray]] = field(init=False)

    def __post_init__(self) -> None:
        r0, r1 = self.bounds.reach
        h0, h1 = self.bounds.height
        w0, w1 = self.bounds.width
        corners = np.array(
            [
                [r0, h0, w0],
                [r1, h0, w0],
                [r1, h1, w0],
                [r0, h1, w0],
                [r0, h0, w1],
                [r1, h0, w1],
                [r1, h1, w1],
                [r0, h1, w1],
            ],
            dtype=float,
        )
        edge_indices = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        self._segments = [(corners[start].copy(), corners[end].copy()) for start, end in edge_indices]

    def sample(self, t: float) -> TaskState:
        local_t = np.clip(max(t, 0.0) / self.segment_duration, 0.0, float(len(self._segments)))
        segment = min(int(local_t), len(self._segments) - 1)
        alpha = min(local_t - segment, 1.0)
        smooth = alpha * alpha * (3.0 - 2.0 * alpha)
        start, end = self._segments[segment]
        state = (1.0 - smooth) * start + smooth * end
        return self.bounds.clip(state)


def make_task_policy(mode: str, bounds: TaskBounds) -> RandomWaypointPolicy | RandomSplinePolicy | WorkspaceEdgeSamplePolicy:
    if mode == RANDOM_CUBIC_SPLINE_MODE:
        return RandomSplinePolicy(bounds)
    if mode == RANDOM_WAYPOINT_MODE:
        return RandomWaypointPolicy(bounds)
    if mode == WS_EDGE_SAMPLE_MODE:
        return WorkspaceEdgeSamplePolicy(bounds)
    raise ValueError(f"Unsupported task policy mode: {mode}")


def run_demo(
    output_path: Path,
    *,
    mode: str = RANDOM_CUBIC_SPLINE_MODE,
    width: int = DEFAULT_RENDER_WIDTH,
    height: int = DEFAULT_RENDER_HEIGHT,
    fps: int = DEFAULT_RENDER_FPS,
) -> Path:
    env = LabEnv()
    policy = make_task_policy(mode, env.task_bounds)
    steps_per_frame = max(1, int(round(1.0 / (fps * env.model.opt.timestep))))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    camera_id = env.model.camera("video_cam").id

    frames: list[np.ndarray] = []
    frame_dt = 1.0 / fps
    started = time.time()
    next_frame_time = started

    with mujoco.Renderer(env.model, height=height, width=width) as renderer:
        try:
            with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = DEFAULT_VIEWER_LOOKAT
                viewer.cam.distance = DEFAULT_VIEWER_DISTANCE
                viewer.cam.azimuth = DEFAULT_VIEWER_AZIMUTH
                viewer.cam.elevation = DEFAULT_VIEWER_ELEVATION
                while viewer.is_running():
                    elapsed = time.time() - started
                    desired = policy.sample(elapsed)
                    env.set_task_target(desired)
                    env.step(steps_per_frame)
                    viewer.sync()

                    now = time.time()
                    if now >= next_frame_time:
                        renderer.update_scene(env.data, camera=camera_id)
                        frames.append(renderer.render().copy())
                        next_frame_time += frame_dt
        except KeyboardInterrupt:
            pass

    if frames:
        imageio.mimwrite(output_path, frames, fps=fps, quality=8, macro_block_size=1)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="View and record a rope manipulation video.")
    parser.add_argument(
        "--mode",
        type=str,
        default=MODE,
        choices=[RANDOM_CUBIC_SPLINE_MODE, RANDOM_WAYPOINT_MODE, WS_EDGE_SAMPLE_MODE],
        help="Task generator mode.",
    )
    parser.add_argument("--width", type=int, default=DEFAULT_RENDER_WIDTH, help="Output video width in pixels.")
    parser.add_argument("--height", type=int, default=DEFAULT_RENDER_HEIGHT, help="Output video height in pixels.")
    parser.add_argument("--fps", type=int, default=DEFAULT_RENDER_FPS, help="Output video frames per second.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Output MP4 path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = run_demo(
        args.output,
        mode=args.mode,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    print(
        {
            "output_video": str(output_path.resolve()),
            "mode": args.mode,
            "width": args.width,
            "height": args.height,
            "fps": args.fps,
        }
    )


if __name__ == "__main__":
    main()
