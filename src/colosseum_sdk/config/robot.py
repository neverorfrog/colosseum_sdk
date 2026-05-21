from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class RobotCfg:
    xml_path: Path
    joint_names: list[str]
    # Per-joint values keyed by exact joint name or regex pattern (first match wins).
    # resolve_expr() in utils.py converts these to ordered np.ndarray at runtime.
    default_joint_pos: dict[str, float]
    joint_stiffness: dict[str, float]
    joint_damping: dict[str, float]
    effort_limit: dict[str, float]
    joint_armature: dict[str, float]
    joint_frictionloss: dict[str, float]
    skill_models: dict[str, Path] = field(default_factory=dict)


ROBOT_REGISTRY: dict[str, "RobotCfg"] = {}


def register_robot(name: str):
    """Decorator that instantiates the class and stores it in ROBOT_REGISTRY.

    Usage:
        @register_robot("t1")
        class T1RobotCfg(RobotCfg):
            def __init__(self):
                super().__init__(xml_path=..., joint_names=[...], ...)
    """

    def decorator(cls: type) -> type:
        ROBOT_REGISTRY[name] = cls()
        return cls

    return decorator


def get_robot(name: str) -> RobotCfg:
    if name not in ROBOT_REGISTRY:
        raise KeyError(f"Robot '{name}' not found. Available: {list(ROBOT_REGISTRY)}")
    return ROBOT_REGISTRY[name]
