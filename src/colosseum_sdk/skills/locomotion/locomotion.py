import math
from pathlib import Path

import mujoco
import numpy as np

from colosseum_sdk.config.robot import get_robot
from colosseum_sdk.inference.onnx import OnnxInferenceEngine
from colosseum_sdk.skills.base import Skill
from colosseum_sdk.skills.locomotion.gait_phase import GaitPhaseCommand
from colosseum_sdk.skills.locomotion.velocity_command import VelocityCommand
from colosseum_sdk.state import RobotState
from colosseum_sdk.utils import resolve_expr

# Must match arena T1VelocitySymmetric.cpp: policy_dt=0.02s, decimation=4, physics_dt=0.005s
_POLICY_DT: float = 0.02
_ACTION_SCALE: float = 0.25


class LocomotionSkill(Skill):
    """ONNX locomotion policy for T1. Port of arena T1VelocitySymmetric.

    Observation (82 elements):
      [0:3]   gyro — body-frame angular velocity
      [3:6]   projected_gravity — gravity in body frame
      [6:29]  joint_pos - default_joint_pos
      [29:52] joint_vel
      [52:75] last_action (raw network output, before scaling)
      [75:78] velocity command [vx, vy, vyaw]  (rate-limited)
      [78:82] gait phase [cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)]

    Action decoding: target[i] = net_out[i] * 0.25 + default_joint_pos[i]
    """

    decimation: int = 4  # physics steps per policy step (0.005 * 4 = 0.02s)

    def __init__(
        self,
        robot_type: str = "t1",
        model_path: str | Path | None = None,
    ) -> None:
        robot_cfg = get_robot(robot_type)
        onnx_path = Path(model_path) if model_path else robot_cfg.skill_models["locomotion"]
        self._engine = OnnxInferenceEngine(onnx_path)

        self._robot_cfg = robot_cfg
        self._n = len(robot_cfg.joint_names)
        self._default_pos = resolve_expr(robot_cfg.default_joint_pos, robot_cfg.joint_names)

        self._vel_cmd = VelocityCommand()
        self._gait = GaitPhaseCommand()
        self._last_action = np.zeros(self._n, dtype=np.float32)

        # Set by setup() after the MuJoCo model is compiled.
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._qpos_idx: np.ndarray | None = None
        self._dof_idx: np.ndarray | None = None
        self._gyro_adr: int = -1
        self._quat_adr: int = -1
        self._fj_dof_adr: int = -1

    # ------------------------------------------------------------------
    # Public API

    def set_velocity(self, vx: float = 0.0, vy: float = 0.0, vyaw: float = 0.0) -> None:
        """Set desired base velocity. Thread-safe for main-thread → viewer-thread use."""
        self._vel_cmd.set(vx, vy, vyaw)

    def setup(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called by Simulation.compile() once the MjModel is ready."""
        self._model = model
        self._data = data
        names = self._robot_cfg.joint_names
        n = len(names)

        qpos_idx = np.full(n, -1, dtype=np.intp)
        dof_idx = np.full(n, -1, dtype=np.intp)
        for i, name in enumerate(names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qpos_idx[i] = model.jnt_qposadr[jid]
                dof_idx[i] = model.jnt_dofadr[jid]
        self._qpos_idx = qpos_idx
        self._dof_idx = dof_idx

        for sensor_name, attr in [("imu_ang_vel", "_gyro_adr"), ("orientation", "_quat_adr")]:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, sensor_name)
            setattr(self, attr, int(model.sensor_adr[sid]) if sid >= 0 else -1)

        fj_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint")
        self._fj_dof_adr = int(model.jnt_dofadr[fj_id]) if fj_id >= 0 else 0

    def reset(self) -> None:
        self._last_action[:] = 0.0
        self._engine.reset_state()
        self._gait.reset()
        self._vel_cmd.stop()
        self._vel_cmd.vx = 0.0
        self._vel_cmd.vy = 0.0
        self._vel_cmd.vyaw = 0.0

    # ------------------------------------------------------------------
    # Skill interface

    def compute(self) -> dict[str, float]:
        if self._model is None or self._data is None:
            return {}
        state = self._extract_state()
        obs = self._build_observation(state)
        net_out = self._engine.infer(obs)
        self._last_action[:] = net_out
        return self._decode_action(net_out)

    # ------------------------------------------------------------------
    # Internal

    def _extract_state(self) -> RobotState:
        assert self._data is not None
        assert self._qpos_idx is not None
        assert self._dof_idx is not None
        data = self._data
        n = self._n

        joint_pos = np.empty(n)
        joint_vel = np.empty(n)
        valid = self._qpos_idx >= 0
        joint_pos[valid] = data.qpos[self._qpos_idx[valid]]
        joint_vel[valid] = data.qvel[self._dof_idx[valid]]

        if self._gyro_adr >= 0:
            gyro = data.sensordata[self._gyro_adr : self._gyro_adr + 3].copy()
        else:
            gyro = np.zeros(3)

        if self._quat_adr >= 0:
            q = data.sensordata[self._quat_adr : self._quat_adr + 4]  # wxyz
            root_quat = q.copy()
            projected_gravity = _projected_gravity(q)
            base_lin_vel = _rotate_to_body(q, data.qvel[self._fj_dof_adr : self._fj_dof_adr + 3])
        else:
            root_quat = np.array([1.0, 0.0, 0.0, 0.0])
            projected_gravity = np.array([0.0, 0.0, -1.0])
            base_lin_vel = np.zeros(3)

        return RobotState(
            gyro=gyro,
            projected_gravity=projected_gravity,
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            root_quat=root_quat,
            base_lin_vel=base_lin_vel,
        )

    def _build_observation(self, state: RobotState) -> np.ndarray:
        self._vel_cmd.step_filter(_POLICY_DT)
        horiz_speed = math.hypot(self._vel_cmd.vx, self._vel_cmd.vy)
        self._gait.advance(_POLICY_DT, horiz_speed)

        obs = np.empty(82, dtype=np.float32)
        obs[0:3] = state.gyro
        obs[3:6] = state.projected_gravity
        obs[6:29] = state.joint_pos - self._default_pos
        obs[29:52] = state.joint_vel
        obs[52:75] = self._last_action
        obs[75] = self._vel_cmd.vx
        obs[76] = self._vel_cmd.vy
        obs[77] = self._vel_cmd.vyaw
        obs[78], obs[79], obs[80], obs[81] = self._gait.command()
        return obs

    def _decode_action(self, net_out: np.ndarray) -> dict[str, float]:
        targets = net_out * _ACTION_SCALE + self._default_pos
        return {name: float(targets[i]) for i, name in enumerate(self._robot_cfg.joint_names)}


# ------------------------------------------------------------------
# Quaternion helpers (port of MujocoPortal.cpp)

def _projected_gravity(q: np.ndarray) -> np.ndarray:
    """Rotate world gravity [0,0,-1] into body frame. q = (w,x,y,z)."""
    w, x, y, z = q
    return np.array([
         2.0 * (w * y - x * z),
        -2.0 * (w * x + y * z),
        -(1.0 - 2.0 * (x * x + y * y)),
    ])


def _rotate_to_body(q: np.ndarray, v_world: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector into body frame using inverse quaternion. q = (w,x,y,z)."""
    w, x, y, z = q
    vx, vy, vz = v_world
    w2 = w * w
    dot = x * vx + y * vy + z * vz
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    return np.array([
        vx * (2 * w2 - 1) - 2 * w * cx + 2 * x * dot,
        vy * (2 * w2 - 1) - 2 * w * cy + 2 * y * dot,
        vz * (2 * w2 - 1) - 2 * w * cz + 2 * z * dot,
    ])
