import mujoco
import numpy as np

from colosseum_sdk.config.robot import RobotCfg
from colosseum_sdk.utils import resolve_expr


class PDController:
    def __init__(
        self,
        model: mujoco.MjModel,
        robot_cfg: RobotCfg,
        decimation: int = 4,
        physics_dt: float = 0.005,
    ) -> None:
        self._model = model
        self._decimation = decimation

        names = robot_cfg.joint_names
        n = len(names)

        # name → position in the joint_names list, for set_targets().
        self._name_to_idx: dict[str, int] = {name: i for i, name in enumerate(names)}

        # Per-joint MuJoCo address arrays (-1 means joint absent in this model).
        self._qpos_idx = np.full(n, -1, dtype=np.intp)
        self._dof_idx = np.full(n, -1, dtype=np.intp)
        for i, name in enumerate(names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                self._qpos_idx[i] = model.jnt_qposadr[jid]
                self._dof_idx[i] = model.jnt_dofadr[jid]

        self._kp = resolve_expr(robot_cfg.joint_stiffness, names)
        self._kd = resolve_expr(robot_cfg.joint_damping, names)
        self._limit = resolve_expr(robot_cfg.effort_limit, names)

        # Default targets: hold at default_joint_pos.
        self._targets = resolve_expr(robot_cfg.default_joint_pos, names).copy()

    def set_targets(self, targets: dict[str, float]) -> None:
        """Update position targets. Keys are exact joint names."""
        for name, val in targets.items():
            idx = self._name_to_idx.get(name)
            if idx is not None:
                self._targets[idx] = val

    def step(self, data: mujoco.MjData) -> None:
        """Apply PD torques and advance one physics step."""
        valid = self._dof_idx >= 0
        qpos = data.qpos[self._qpos_idx[valid]]
        qvel = data.qvel[self._dof_idx[valid]]
        tau = self._kp[valid] * (self._targets[valid] - qpos) - self._kd[valid] * qvel
        tau = np.clip(tau, -self._limit[valid], self._limit[valid])
        data.qfrc_applied[self._dof_idx[valid]] = tau
        mujoco.mj_step(self._model, data)
