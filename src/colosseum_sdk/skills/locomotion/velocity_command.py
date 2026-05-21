from dataclasses import dataclass


@dataclass
class VelocityCommandConfig:
    vx_max: float = 1.0  # m/s forward/backward
    vy_max: float = 0.5  # m/s lateral
    vyaw_max: float = 1.0  # rad/s yaw
    ramp_vx: float = 3.0  # m/s² acceleration limit
    ramp_vy: float = 3.0
    ramp_vyaw: float = 0.0


class VelocityCommand:
    """Velocity command with rate-limiter. Port of arena/include/VelocityCommand.h.

    Call set() to write an instantaneous target.
    step_filter() ramps the live values toward that target at bounded rate.
    The observation uses the live values (vx, vy, vyaw), not the raw targets.
    """

    def __init__(self, cfg: VelocityCommandConfig = VelocityCommandConfig()) -> None:
        self.vx_max = cfg.vx_max
        self.vy_max = cfg.vy_max
        self.vyaw_max = cfg.vyaw_max
        self._ramp_vx = cfg.ramp_vx
        self._ramp_vy = cfg.ramp_vy
        self._ramp_vyaw = cfg.ramp_vyaw

        self._target_vx: float = 0.0
        self._target_vy: float = 0.0
        self._target_vyaw: float = 0.0

        self.vx: float = 0.0
        self.vy: float = 0.0
        self.vyaw: float = 0.0

    def set(self, vx: float, vy: float, vyaw: float) -> None:
        self._target_vx = max(-self.vx_max, min(self.vx_max, vx))
        self._target_vy = max(-self.vy_max, min(self.vy_max, vy))
        self._target_vyaw = max(-self.vyaw_max, min(self.vyaw_max, vyaw))

    def step_filter(self, dt: float) -> None:
        self.vx = _ramp(self.vx, self._target_vx, self._ramp_vx, dt)
        self.vy = _ramp(self.vy, self._target_vy, self._ramp_vy, dt)
        self.vyaw = _ramp(self.vyaw, self._target_vyaw, self._ramp_vyaw, dt)

    def stop(self) -> None:
        self._target_vx = 0.0
        self._target_vy = 0.0
        self._target_vyaw = 0.0


def _ramp(cur: float, tgt: float, max_rate: float, dt: float) -> float:
    step = max_rate * dt
    err = tgt - cur
    if err > step:
        return cur + step
    if err < -step:
        return cur - step
    return tgt
