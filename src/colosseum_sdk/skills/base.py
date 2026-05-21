from abc import ABC, abstractmethod

import mujoco


class Skill(ABC):
    decimation: int = 1  # subclasses override to run at a lower frequency

    @abstractmethod
    def compute(self) -> dict[str, float]:
        """Return {joint_name: position_target} for any subset of joints."""

    def setup(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        """Called once by Simulation.compile() after the model is ready. Override if needed."""

    def reset(self) -> None:
        pass
