import math
import numpy as np
import gym
from gym import spaces

class HeadingWrapper(gym.Wrapper):
    """
    Duckietown environment with a single continuous value that
    controls the current vehicle heading/direction
    """

    def __init__(self, env):
        super(HeadingWrapper, self).__init__(env)

        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(1,),
            dtype=np.float32
        )

    def step(self, action):
        action = np.tanh(action)

        #print(action)

        # Compute the motor velocities
        lVel = np.array([0.4, 0.5])
        rVel = np.array([0.5, 0.4])

        x = (action + 1) / 2
        #print(x)

        vel = lVel * (1 - x) + x * rVel

        return self.env.step(vel)

    def reset(self, **kwargs):
        self.heading = 0
        return self.env.reset(**kwargs)

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        super(DiscreteWrapper, self).__init__(env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [0.35, 0.58]
        # Turn right
        elif action == 1:
            vels = [0.58, 0.35]
        # Go forward
        elif action == 2:
            vels = [0.58, 0.58]
        else:
            assert False, "unknown action"
        return np.array(vels)
