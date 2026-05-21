import re

import numpy as np


def resolve_expr(
    pattern_dict: dict[str, float],
    joint_names: list[str],
    default: float = 0.0,
) -> np.ndarray:
    """Return a per-joint array by matching each joint name against pattern_dict keys.

    Keys are tried in insertion order; the first full regex match wins.
    Joints with no match receive `default`.
    """
    values = np.full(len(joint_names), default, dtype=np.float64)
    for i, name in enumerate(joint_names):
        for pattern, val in pattern_dict.items():
            if re.fullmatch(pattern, name):
                values[i] = val
                break
    return values
