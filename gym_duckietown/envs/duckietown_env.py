import math
import numpy as np
import gym
from gym import spaces

from ..simulator import Simulator

class DuckietownEnv(Simulator):
    """
    Wrapper to control the simulator using velocity and steering angle
    instead of differential drive motor velocities
    """

    def __init__(
        self,
        gain = 1.0,
        trim = 0.0,
        radius = 0.0318,
        k = 27.0,
        limit = 1.0,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.action_space = spaces.Box(
            low=np.array([-1,-1]),
            high=np.array([1,1]),
            dtype=np.float32
        )

        # should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # directional trim adjustment
        self.trim = trim

        # Minimal turn radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

    def step(self, action):
        vel, angle = action

        # Distance between the wheels
        baseline = self.unwrapped.wheel_dist_left + self.unwrapped.wheel_dist_right

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * baseline) / self.radius
        omega_l = (vel - 0.5 * angle * baseline) / self.radius

        # conversion from motor rotation rate to duty cycle
        # u_r = (gain + trim) (v + 0.5 * omega * b) / (r * k_r)
        u_r = omega_r * k_r_inv
        # u_l = (gain - trim) (v - 0.5 * omega * b) / (r * k_l)
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])

        return super().step(vels)
