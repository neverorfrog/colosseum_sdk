from dataclasses import dataclass, field


@dataclass
class InitStateCfg:
    pos: tuple[float, float, float] = (0.0, 0.0, 0.92)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)  # quaternion wxyz
    # None → use robot_cfg.default_joint_pos; otherwise a regex-pattern dict (first match wins).
    joint_pos: dict[str, float] | None = None


@dataclass
class SceneCfg:
    robot_type: str = "t1"  # key into ROBOT_REGISTRY
    robot: InitStateCfg = field(default_factory=InitStateCfg)
