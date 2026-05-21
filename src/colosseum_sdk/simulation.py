import threading
import time

import mujoco
import numpy as np
from mjviser import Viewer as ViserViewer

from colosseum_sdk.config.robot import get_robot
from colosseum_sdk.config.scene import SceneCfg
from colosseum_sdk.control.controller import PDController
from colosseum_sdk.skills.base import Skill
from colosseum_sdk.utils import resolve_expr


class Simulation:
    def __init__(self, cfg: SceneCfg = SceneCfg()):
        self._cfg = cfg
        self._model: mujoco.MjModel | None = None
        self._data: mujoco.MjData | None = None
        self._init_qpos: np.ndarray | None = None
        self._controller: PDController | None = None
        self._step_count: int = 0
        self._skills: list[Skill] = []
        self._viewer: ViserViewer | None = None

    def use_skill(self, skill: Skill) -> None:
        """Register a skill. Later skills override earlier ones per joint."""
        self._skills.append(skill)

    def compile(self) -> None:
        """Build MjSpec → compile → set init state → create PD controller."""
        spec = self._build_spec()
        model = spec.compile()
        data = mujoco.MjData(model)
        # Match arena physics_dt so decimation=4 → policy_dt=0.02s.
        model.opt.timestep = 0.005
        model.opt.iterations = 10
        model.opt.ls_iterations = 20
        self._model = model
        self._data = data
        robot_cfg = get_robot(self._cfg.robot_type)
        self._apply_physics_overrides(model, robot_cfg)
        self._apply_init_state()
        self._controller = PDController(model, robot_cfg)
        self._step_count = 0
        for skill in self._skills:
            skill.setup(model, data)

    def open_viewer(self) -> None:
        """Open the full viser viewer in a background thread (non-blocking).
        Physics and GUI controls (pause/reset/speed) run in that thread.
        The main thread can read sim.data and update skill commands freely.
        """
        model, data = self._require_compiled()
        viewer = ViserViewer(
            model,
            data,
            step_fn=lambda m, d: self._tick(d),
            reset_fn=lambda m, d: self.reset(),
        )
        self._viewer = viewer

        # Viewer.run() installs a SIGINT handler which only works in the main thread.
        # Replicate the loop without it — the daemon thread dies with the process.
        def _loop() -> None:
            viewer._setup_gui()
            mujoco.mj_forward(viewer.model, viewer.data)
            viewer._render()
            now = time.perf_counter()
            viewer._last_tick = now
            viewer._stats_last_time = now
            while True:
                viewer._tick()
                time.sleep(0.001)

        threading.Thread(target=_loop, daemon=True).start()

    def launch(self) -> None:
        """Compile (if needed) + blocking viewer loop. For demos with no manual loop."""
        if self._model is None:
            self.compile()
        model, data = self._require_compiled()
        self._viewer = ViserViewer(
            model,
            data,
            step_fn=lambda m, d: self._tick(d),
            reset_fn=lambda m, d: self.reset(),
        )
        self._viewer.run()

    def reset(self) -> None:
        """Reset to the configured init state."""
        model, data = self._require_compiled()
        mujoco.mj_resetData(model, data)
        if self._init_qpos is not None:
            data.qpos[:] = self._init_qpos
        mujoco.mj_forward(model, data)
        for skill in self._skills:
            skill.reset()

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

    def _tick(self, data: mujoco.MjData) -> None:
        if self._controller is None:
            return
        targets: dict[str, float] = {}
        for skill in self._skills:
            if self._step_count % skill.decimation == 0:
                targets.update(skill.compute())  # later skills win per joint
        self._controller.set_targets(targets)
        self._step_count += 1
        self._controller.step(data)

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

    def _apply_physics_overrides(self, model: mujoco.MjModel, robot_cfg) -> None:
        """Apply per-joint armature and frictionloss — matches arena MujocoPortal.cpp lines 114-117."""
        names = robot_cfg.joint_names
        armature = resolve_expr(robot_cfg.joint_armature, names)
        frictionloss = resolve_expr(robot_cfg.joint_frictionloss, names)
        for i, name in enumerate(names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                dof_adr = model.jnt_dofadr[jid]
                model.dof_armature[dof_adr] = armature[i]
                model.dof_frictionloss[dof_adr] = frictionloss[i]

    def _apply_init_state(self) -> None:
        model, data = self._require_compiled()
        init = self._cfg.robot
        robot_cfg = get_robot(self._cfg.robot_type)

        # model.qpos0 returns a copy on each access — build desired qpos locally.
        qpos = model.qpos0.copy()

        jnt_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_JOINT, "floating_base_joint"
        )
        if jnt_id >= 0:
            adr = model.jnt_qposadr[jnt_id]
            qpos[adr : adr + 3] = init.pos
            qpos[adr + 3 : adr + 7] = init.rot

        pos_dict = (
            robot_cfg.default_joint_pos if init.joint_pos is None else init.joint_pos
        )
        joint_vals = resolve_expr(pos_dict, robot_cfg.joint_names)
        for i, name in enumerate(robot_cfg.joint_names):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid >= 0:
                qpos[model.jnt_qposadr[jid]] = joint_vals[i]

        self._init_qpos = qpos
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)


if __name__ == "__main__":
    sim = Simulation()
    sim.launch()
