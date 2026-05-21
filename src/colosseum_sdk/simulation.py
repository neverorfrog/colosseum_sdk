import mujoco
import numpy as np
from mjviser import Viewer as ViserViewer

from colosseum_sdk.config.robot import get_robot
from colosseum_sdk.config.scene import SceneCfg
from colosseum_sdk.utils import resolve_expr


class Simulation:
    def __init__(self, cfg: SceneCfg = SceneCfg()):
        self._cfg = cfg
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._viewer: "ViserViewer | None" = None

    def compile(self, open_viewer: bool = False) -> None:
        """Build MjSpec → compile → set init state → open viser viewer if requested."""
        spec = self._build_spec()
        model = spec.compile()
        data = mujoco.MjData(model)
        self._model = model
        self._data = data
        self._apply_init_state()
        if open_viewer:
            self._viewer = ViserViewer(model, data, reset_fn=lambda m, d: self.reset())

    def launch(self) -> None:
        """Compile (if needed), open the viser viewer, and block until Ctrl+C."""
        if self._model is None:
            self.compile(open_viewer=True)
        elif self._viewer is None:
            model, data = self._require_compiled()
            self._viewer = ViserViewer(model, data, reset_fn=lambda m, d: self.reset())
        assert self._viewer is not None
        self._viewer.run()

    def reset(self) -> None:
        """Reset to the configured init state."""
        model, data = self._require_compiled()
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

    @property
    def model(self) -> mujoco.MjModel:
        return self._require_compiled()[0]

    @property
    def data(self) -> mujoco.MjData:
        return self._require_compiled()[1]

    def _require_compiled(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        if self._model is None or self._data is None:
            raise RuntimeError("call compile() before using the simulation")
        return self._model, self._data

    def _build_spec(self) -> mujoco.MjSpec:
        robot_cfg = get_robot(self._cfg.robot_type)
        spec = mujoco.MjSpec.from_file(str(robot_cfg.xml_path))

        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = np.array([0, 0, 0.01])
        floor.rgba = np.array([0.2, 0.3, 0.4, 1.0])

        light = spec.worldbody.add_light()
        light.name = "main"
        light.pos = np.array([0.0, 0.0, 3.0])
        light.dir = np.array([0.0, 0.0, -1.0])
        light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL

        return spec

    def _apply_init_state(self) -> None:
        model, data = self._require_compiled()
        init = self._cfg.robot
        robot_cfg = get_robot(self._cfg.robot_type)

        # Freejoint: 7 dofs = pos(3) + quaternion wxyz(4)
        jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if jnt_id >= 0:
            adr = model.jnt_qposadr[jnt_id]
            model.qpos0[adr : adr + 3] = init.pos
            model.qpos0[adr + 3 : adr + 7] = init.rot

        # Joint positions: use robot defaults unless user supplied explicit pattern dict
        pos_dict = (
            robot_cfg.default_joint_pos if init.joint_pos is None else init.joint_pos
        )
        joint_vals = resolve_expr(pos_dict, robot_cfg.joint_names)
        for i, name in enumerate(robot_cfg.joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                model.qpos0[model.jnt_qposadr[jid]] = joint_vals[i]

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)


if __name__ == "__main__":
    sim = Simulation()
    sim.launch()
