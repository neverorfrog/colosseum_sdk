import time

from colosseum_sdk.config.scene import InitStateCfg, SceneCfg
from colosseum_sdk.simulation import Simulation
from colosseum_sdk.skills.arm import ArmSkill
from colosseum_sdk.skills.locomotion import LocomotionSkill

# --- Scene configuration ---
cfg = SceneCfg(
    robot_type="t1",
    robot=InitStateCfg(
        pos=(0.0, 0.0, 0.64),
        rot=(1.0, 0.0, 0.0, 0.0),
    ),
)

# --- Skills ---
loco = LocomotionSkill()
arm = ArmSkill()

# --- Simulation setup ---
sim = Simulation(cfg)
sim.use_skill(loco)  # all 23 joints — lowest priority
sim.use_skill(arm)  # right arm (4 joints) — overrides loco on those joints
sim.compile()
sim.open_viewer()  # opens browser tab; physics + GUI run in background thread

# --- Control loop (50 Hz) ---
t = 0.0
while True:
    if t < 3.0:
        loco.set_velocity(vx=0.5)  # walk forward
    elif t < 5.0:
        loco.set_velocity(vx=0.0)  # stop
        arm.go_to("raise", duration=1.0)  # raise right arm
    elif t > 10.0:
        t = 0.0
        arm.go_to("home", duration=0.5)  # return arm to rest

    t += 0.02
    time.sleep(0.02)
