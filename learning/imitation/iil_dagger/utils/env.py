import math

from gym import ActionWrapper
import numpy as np

from gym_duckietown.envs.duckietown_env import DuckietownLF
from gym_duckietown.simulator import Simulator


class RescaleActionWrapper(ActionWrapper):
    def __init__(self, env, max_velocity, steering_gain):
        super(ActionWrapper, self).__init__(env)
        self.max_velocity = max_velocity
        self.steering_gain = steering_gain

    def action(self, action):
        action = np.clip(action, -1, 1) # Sometimes, patch+combined is less than one (can actually be -2 to +2, technically)

        action_ = [self.max_velocity*(action[0]--1)/(1--1), action[1]*self.steering_gain]

        print(action_)
        return action_

class SteeringToWheelVelWrapper(ActionWrapper):
    """
    Converts policy that was trained with [velocity|heading] actions to
    [wheelvel_left|wheelvel_right] to comply with AIDO evaluation format
    """

    def __init__(self,
                 env,
                 gain=1.0,
                 trim=0.0,
                 radius=0.0318,
                 k=27.0,
                 limit=1.0,
                 wheel_dist=0.102
                 ):
        ActionWrapper.__init__(self, env)

        # Should be adjusted so that the effective speed of the robot is 0.2 m/s
        self.gain = gain

        # Directional trim adjustment
        self.trim = trim

        # Wheel radius
        self.radius = radius

        # Motor constant
        self.k = k

        # Wheel velocity limit
        self.limit = limit

        self.wheel_dist = wheel_dist

    def action(self, action):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        vels = np.array([u_l_limited, u_r_limited])
        return vels

def launch_env(map_name, randomize_maps_on_reset=False, domain_rand=False):
    environment = DuckietownLF( #todo sim is bad
        domain_rand=domain_rand,
        max_steps=math.inf,
        map_name=map_name,
        randomize_maps_on_reset=randomize_maps_on_reset
    )

    wrapped = RescaleActionWrapper(environment, 0.7, 10)   # TODO pass these through instead
    #wrapped = SteeringToWheelVelWrapper(wrapped)
    wrapped._get_tile = environment._get_tile

    return wrapped