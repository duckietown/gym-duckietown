#!/usr/bin/env python

import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import time
import numpy
import zmq
from PIL import Image
import pyglet
from pyglet.image import ImageData
from pyglet.gl import *
import numpy as np

# For Python 3 compatibility
import sys
if sys.version_info > (3,):
    buffer = memoryview

# Rendering window size
WINDOW_SIZE = 512

# Camera image size
CAMERA_WIDTH = 64
CAMERA_HEIGHT = 64

# Camera image shape
IMG_SHAPE = (3, CAMERA_WIDTH, CAMERA_HEIGHT)

# Port to connect to on the server
SERVER_PORT = 7777

def recvArray(socket):
    """Receive a numpy array over zmq"""
    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    A = A.reshape(md['shape'])
    return A


class DuckiebotEnv(gym.Env):
    """An environment that is the actual real robot """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'app'],
        'video.frames_per_second' : 30
    }

    def __init__(self,
                 serverAddr="couguar.local",
                 serverPort=SERVER_PORT):
        print("entering init!!!")
        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,)
        )

        # We observe an RGB image with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=IMG_SHAPE
        )

        self.reward_range = (-1, 1000)

        # Environment configuration
        self.maxSteps = 50

        # Array to render the image into
        self.imgArray = np.zeros(shape=IMG_SHAPE, dtype=np.float32)

        # For rendering
        self.window = None

        # We continually stream in images and then just take the latest one.
        self.latest_img = None

        # For displaying text
        self.textLabel = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_SIZE - 19
        )

        # Connect to the Gym bridge ROS node
        print("connecting...")
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://%s:%s" % (serverAddr, serverPort))
        print("connected! :)")

        # Initialize the state
        self.seed()
        self.reset()


    def _close(self):
        pass

    def _reset(self):
        # Step count since episode start
        self.stepCount = 0

        self.socket.send_json({
            "command":"reset"
        })

        # Receive a camera image from the server
        print("grabbing image..")
        self.img = recvArray(self.socket)
        self.img = numpy.flip(self.img, axis=0)

        print("got image")

        # Return first observation
        return self.img.transpose()

    def _seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def _step(self, action):

        # we don't care about this reward since we're not training..
        reward = 0
        # don't worry about episodes blah blah blah we will just shut down the robot when we're done
        done = False

        # Send the action to the server
        self.socket.send_json({
            "command":"action",
            "values": [ float(action[0]), float(action[1]) ]
        })

        # Receive a camera image from the server
        self.img = recvArray(self.socket)
        self.img = numpy.flip(self.img, axis=0)

        return self.img.transpose(), reward, done, {}

    def _render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'rgb_array':
            return self.img

        if self.window is None:
            context = pyglet.gl.get_current_context()
            self.window = pyglet.window.Window(
                width=WINDOW_SIZE,
                height=WINDOW_SIZE
            )

        self.window.switch_to()
        self.window.dispatch_events()

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, WINDOW_SIZE, WINDOW_SIZE)

        self.window.clear()

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, WINDOW_SIZE, 0, WINDOW_SIZE, 0, 10)

        # Draw the image to the rendering window
        width = self.img.shape[0]
        height = self.img.shape[1]
        imgData = pyglet.image.ImageData(
            width,
            height,
            'RGB',
            self.img.tobytes(),
            pitch = width * 3,
        )
        imgData.blit(0, 0, 0, WINDOW_SIZE, WINDOW_SIZE)

        if mode == 'human':
            self.window.flip()
