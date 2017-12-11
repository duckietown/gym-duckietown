import math
import time
import numpy

import pyglet
from pyglet.image import ImageData
from pyglet.gl import *

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# For Python 3 compatibility
import sys
if sys.version_info > (3,):
    buffer = memoryview

# Rendering window size
WINDOW_SIZE = 512

class SimpleSimDscEnv(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        super(SimpleSimDscEnv, self).__init__(env)

        self.action_space = spaces.Discrete(3)

    def _action(self, action):
        if action == 0:
            return [-1, 1]
        elif action == 1:
            return [1, -1]
        elif action == 2:
            return [1, 1]
        else:
            assert False, "unknown action"

class SimpleSimEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    # Camera image size
    CAMERA_WIDTH = 64
    CAMERA_HEIGHT = 64

    # Camera image shape
    IMG_SHAPE = (3, CAMERA_WIDTH, CAMERA_HEIGHT)

    def __init__(self):

        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,)
        )

        # We observe an RGB image with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=SimpleSimEnv.IMG_SHAPE
        )

        self.reward_range = (-1, 1000)

        # Environment configuration
        self.maxSteps = 120

        # For rendering
        self.window = None

        # For displaying text
        self.textLabel = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_SIZE - 19
        )

        # Starting position
        self.startPos = (-0.5, 0.2, 0)

        # Initialize the state
        self.reset()
        self.seed()

    def _close(self):
        pass

    def _reset(self):
        # Step count since episode start
        self.stepCount = 0

        self.curPos = self.startPos

        obs = self._renderObs()

        # Return first observation
        return obs

    def _seed(self, seed=None):
        """
        The seed function sets the random elements of the environment.
        """

        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def _step(self, action):
        self.stepCount += 1

        x, y, z = self.curPos

        # End of lane, to the right
        targetPos = (0.0, 0.2, -2.0)

        dx = x - targetPos[0]
        dz = z - targetPos[2]

        dist = abs(dx) + abs(dz)
        reward = -dist

        done = False

        # If the objective is reached
        if dist <= 0.05:
            reward = 1000
            done = True

        obs = self._renderObs()

        # If the maximum time step count is reached
        if self.stepCount >= self.maxSteps:
            done = True

        return obs, reward, done, {}

    def _renderObs(self):
        # TODO: produce a numpy array









        # FIXME
        return None

    def _render(self, mode='human', close=False):
        if mode == 'rgb_array':
            return self.img

        if close:
            if self.window:
                self.window.close()
            return

        if self.window is None:
            self.window = pyglet.window.Window(width=WINDOW_SIZE, height=WINDOW_SIZE)

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        img = self._renderObs()

        """
        # Draw the image to the rendering window
        width = self.img.shape[0]
        height = self.img.shape[1]
        imgData = ImageData(
            width,
            height,
            'RGB',
            self.img.tobytes(),
            pitch = width * 3,
        )
        glPushMatrix()
        glTranslatef(0, WINDOW_SIZE, 0)
        glScalef(1, -1, 1)
        imgData.blit(0, 0, 0, WINDOW_SIZE, WINDOW_SIZE)
        glPopMatrix()
        """

        # Display position/state information
        pos = self.curPos
        self.textLabel.text = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2])
        self.textLabel.draw()

        self.window.flip()
