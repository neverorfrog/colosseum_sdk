import math

import numpy as np

from colosseum_sdk.skills.base import Skill

_ARM_JOINTS = [
    "Right_Shoulder_Pitch",
    "Right_Shoulder_Roll",
    "Right_Elbow_Pitch",
    "Right_Elbow_Yaw",
]

# Joint ranges (from T1 XML):
#   Right_Shoulder_Pitch: -3.31 to  1.22
#   Right_Shoulder_Roll:  -1.57 to  1.74
#   Right_Elbow_Pitch:    -2.27 to  2.27
#   Right_Elbow_Yaw:       0.00 to  2.44
POSES: dict[str, dict[str, float]] = {
    "home": {
        "Right_Shoulder_Pitch": 0.25,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Pitch": 0.0,
        "Right_Elbow_Yaw": 0.2,
    },
    "raise": {
        "Right_Shoulder_Pitch": -1.4,
        "Right_Shoulder_Roll": 1.4,
        "Right_Elbow_Pitch": 0.0,
        "Right_Elbow_Yaw": 0.2,
    },
}

_DT: float = (
    0.02  # policy step size (matches LocomotionSkill.decimation=4 at 200 Hz physics)
)


class ArmSkill(Skill):
    """Scripted right-arm controller.

    Overrides the 4 right-arm joints produced by LocomotionSkill.
    Register it *after* LocomotionSkill so its targets take priority.

    Usage:
        arm = ArmSkill()
        sim.use_skill(loco)
        sim.use_skill(arm)   # runs on top of loco
        sim.compile()

        arm.go_to("raise")   # smoothly raises arm over 1 s
        arm.go_to("home")    # return to rest
    """

    decimation: int = 4

    def __init__(self) -> None:
        home = POSES["home"]
        self._pos = np.array([home[j] for j in _ARM_JOINTS], dtype=np.float32)
        self._start = self._pos.copy()
        self._target = self._pos.copy()
        self._elapsed = 0.0
        self._duration = 0.0

    # ------------------------------------------------------------------

    def go_to(self, pose_name: str, duration: float = 1.0) -> None:
        """Smoothly interpolate to a named pose over `duration` seconds.

        Safe to call every loop iteration — ignored if already heading to the same pose.
        """
        if pose_name not in POSES:
            raise KeyError(f"Unknown pose '{pose_name}'. Available: {list(POSES)}")
        new_target = np.array(
            [POSES[pose_name][j] for j in _ARM_JOINTS], dtype=np.float32
        )
        self._start = self._pos.copy()
        self._target = new_target
        self._elapsed = 0.0
        self._duration = max(duration, _DT)

    # ------------------------------------------------------------------
    # Skill interface

    def compute(self) -> dict[str, float]:
        if self._elapsed < self._duration:
            alpha = self._elapsed / self._duration
            alpha = alpha * alpha * (3.0 - 2.0 * alpha)  # smoothstep
            self._pos[:] = self._start + alpha * (self._target - self._start)
            self._elapsed += _DT

        return {j: float(self._pos[i]) for i, j in enumerate(_ARM_JOINTS)}

    def reset(self) -> None:
        home = POSES["home"]
        self._pos[:] = [home[j] for j in _ARM_JOINTS]
        self._start[:] = self._pos
        self._target[:] = self._pos
        self._elapsed = 0.0
        self._duration = 0.0
