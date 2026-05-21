from dataclasses import dataclass

import numpy as np


@dataclass
class RobotState:
    gyro: np.ndarray               # (3,) body-frame angular velocity rad/s
    projected_gravity: np.ndarray  # (3,) gravity vector in body frame
    joint_pos: np.ndarray          # (N,) in joint_names order
    joint_vel: np.ndarray          # (N,) in joint_names order
    root_quat: np.ndarray          # (4,) wxyz
    base_lin_vel: np.ndarray       # (3,) body-frame linear velocity
