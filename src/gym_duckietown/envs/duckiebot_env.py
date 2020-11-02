#!/usr/bin/env python
# coding=utf-8
import math

# For Python 3 compatibility
import sys

import cv2
import gym
import numpy
import numpy as np
import pyglet
import zmq
from gym import spaces
from gym.utils import seeding
from pyglet import gl

if sys.version_info > (3,):
    buffer = memoryview

# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120

# Camera image shape
IMG_SHAPE = (3, CAMERA_HEIGHT, CAMERA_WIDTH)

# Port to connect to on the server
SERVER_PORT = 7777


def recvArray(socket):
    """Receive a numpy array over zmq"""
    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md["dtype"])
    A = A.reshape(md["shape"])
    return A


class DuckiebotEnv(gym.Env):
    """An environment that is the actual real robot """

    metadata = {"render.modes": ["human", "rgb_array", "app"], "video.frames_per_second": 30}

    def __init__(self, serverAddr="akira.local", serverPort=SERVER_PORT):
        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)

        # We observe an RGB image with pixels in [0, 255]
        self.observation_space = spaces.Box(low=0, high=255, shape=IMG_SHAPE, dtype=np.uint8)

        self.reward_range = (-10, 1000)

        # Environment configuration
        self.max_steps = math.inf

        # For rendering
        self.window = None

        # We continually stream in images and then just take the latest one.
        self.latest_img = None

        # For displaying text

        self.textLabel = pyglet.text.Label(font_name="Arial", font_size=14, x=5, y=WINDOW_HEIGHT - 19)

        # Connect to the Gym bridge ROS node
        addr_str = "tcp://%s:%s" % (serverAddr, serverPort)
        print("connecting to %s ..." % addr_str)
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect(addr_str)
        print("connected! :)")

        # Initialize the state
        self.seed()
        self.reset()

    def close(self):
        # Stop the motors
        self.step(numpy.array([0, 0]))

    def _recvFrame(self):
        # Receive a camera image from the server
        self.img = recvArray(self.socket)

        # h, w, _ = self.img.shape
        # if w > h:
        #    d = (w - h) // 2
        #    self.img = self.img[:, d:(w-d), :]

        # Resize the image
        self.img = cv2.resize(self.img, (CAMERA_WIDTH, CAMERA_HEIGHT), interpolation=cv2.INTER_AREA)

        # print(self.img.shape)

        # BGR to RGB
        self.img = self.img[:, :, ::-1]

        # Flip vertically
        self.img = numpy.flip(self.img, axis=0)

    def reset(self):
        # Step count since episode start
        self.step_count = 0

        self.socket.send_json({"command": "reset"})

        # Receive a camera image from the server
        self._recvFrame()

        return self.img

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def step(self, action):
        self.step_count += 1

        # we don't care about rewards or episodes since we're not training
        reward = 0
        done = False

        # Send the action to the server
        self.socket.send_json({"command": "action", "values": [float(action[0]), float(action[1])]})

        # Receive a camera image from the server
        self._recvFrame()

        return self.img, reward, done, {}

    def render(self, mode="human", close=False):
        if close:
            if self.window:
                self.window.close()
            return

        if mode == "rgb_array":
            return self.img

        if self.window is None:
            context = pyglet.gl.get_current_context()
            self.window = pyglet.window.Window(width=WINDOW_WIDTH, height=WINDOW_HEIGHT)

        self.window.switch_to()
        self.window.dispatch_events()

        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        gl.glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        self.window.clear()

        # Setup orghogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

        # Draw the image to the rendering window
        width = self.img.shape[1]
        height = self.img.shape[0]
        imgData = pyglet.image.ImageData(width, height, "RGB", self.img.tobytes(), pitch=width * 3,)
        imgData.blit(0, 0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        if mode == "human":
            self.window.flip()
