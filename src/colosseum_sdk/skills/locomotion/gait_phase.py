import math
from dataclasses import dataclass


@dataclass
class GaitPhaseCommandConfig:
    gait_freq_lo: float = 1.0    # Hz
    gait_freq_hi: float = 1.5    # Hz
    speed_max: float = 1.0       # m/s for freq scaling (unused — freq fixed per episode)
    gate_speed_threshold: float = 0.05  # m/s; below this → standing phase


class GaitPhaseCommand:
    """Two-foot gait phase clock. Port of arena/include/GaitPhaseCommand.h.

    Left foot starts at φ=0, right at φ=π (half-period offset).
    Frequency is fixed at the midpoint of [freq_lo, freq_hi].
    When stopped (|v_xy| < threshold), both phases snap to π → [-1,-1,0,0].
    On stand→walk transition the half-period offset is restored.
    """

    def __init__(self, cfg: GaitPhaseCommandConfig = GaitPhaseCommandConfig()) -> None:
        self._freq_lo = cfg.gait_freq_lo
        self._freq_hi = cfg.gait_freq_hi
        self._threshold = cfg.gate_speed_threshold
        self.freq: float = 0.5 * (cfg.gait_freq_lo + cfg.gait_freq_hi)
        self.phase_left: float = 0.0
        self.phase_right: float = math.pi
        self._was_standing: bool = False

    def reset(self) -> None:
        self.phase_left = 0.0
        self.phase_right = math.pi
        self._was_standing = False

    def advance(self, dt: float, horizontal_speed: float) -> None:
        PI = math.pi
        TWO_PI = 2.0 * PI

        if horizontal_speed <= self._threshold:
            self.phase_left = PI
            self.phase_right = PI
            self._was_standing = True
            return

        if self._was_standing:
            # Restore half-period offset so alternating gait resumes.
            self.phase_right = math.fmod(self.phase_left + PI + PI, TWO_PI) - PI
            self._was_standing = False

        dphi = TWO_PI * dt * self.freq
        self.phase_left = math.fmod(self.phase_left + dphi + PI, TWO_PI) - PI
        self.phase_right = math.fmod(self.phase_right + dphi + PI, TWO_PI) - PI

    def command(self) -> tuple[float, float, float, float]:
        """[cos(φ_L), cos(φ_R), sin(φ_L), sin(φ_R)]"""
        return (
            math.cos(self.phase_left),
            math.cos(self.phase_right),
            math.sin(self.phase_left),
            math.sin(self.phase_right),
        )
