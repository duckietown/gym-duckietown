import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy
import zmq

import pyglet
from pyglet.image import ImageData
from pyglet.gl import glPushMatrix, glPopMatrix, glScalef, glTranslatef

# For Python 3 compatibility
import sys
if sys.version_info > (3,):
    buffer = memoryview

# Rendering window size
WINDOW_SIZE = 512

# Port to connect to on the server
SERVER_PORT = 7777

def recvArray(socket):
    """Receive a numpy array over zmq"""
    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

class DuckietownEnv(gym.Env):
    """
    OpenAI gym environment wrapper for the Duckietown simulation.
    Connects to ROS/Gazebo through ZeroMQ
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    # Camera image size
    CAMERA_WIDTH = 64
    CAMERA_HEIGHT = 64

    # Camera image shape
    IMG_SHAPE = (3, CAMERA_WIDTH, CAMERA_HEIGHT)

    def __init__(self, serverAddr="localhost", serverPort=SERVER_PORT):

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
            shape=DuckietownEnv.IMG_SHAPE
        )

        self.reward_range = (-1, 1000)

        # Environment configuration
        self.maxSteps = 250

        # For rendering
        self.window = None

        # For displaying text
        self.textLabel = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_SIZE - 19
        )

        # Last received state data
        self.stateData = None

        # Last received image
        self.img = None

        # Connect to the Gym bridge ROS node
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://%s:%s" % (serverAddr, serverPort))

        # Initialize the state
        self.reset()
        self.seed()

    def _reset(self):
        # Step count since episode start
        self.stepCount = 0

        # Tell the server to reset the simulation
        self.socket.send_json({ "command":"reset" })

        # Receive state data (position, etc)
        self.stateData = self.socket.recv_json()

        # Receive a camera image from the server
        self.img = recvArray(self.socket)

        print(self.img.transpose().shape)

        # Return first observation
        return self.img.transpose()

    def _seed(self, seed=None):
        """
        The seed function sets the random elements of the environment.
        """

        self.np_random, _ = seeding.np_random(seed)

        # TODO: can we ask the server to generate a new random map here?
        # TODO: does the server decide our starting position on the map?

        return [seed]

    def _step(self, action):
        assert self.observation_space.shape

        self.stepCount += 1

        # Send the action to the server
        self.socket.send_json({
            "command":"action",
            "values": [ float(action[0]), float(action[1]) ]
        })

        # State at the previous step
        self.prevState = self.stateData

        # Receive state data (position, etc)
        self.stateData = self.socket.recv_json()

        # Receive a camera image from the server
        self.img = recvArray(self.socket)

        # Currently, robot starts at (1, 1)
        # And is facing the negative x direction
        # Moving forward decreases x
        # y should stay as close to 1 as possible
        x0, y0, z0 = self.prevState['position']
        x1, y1, z1 = self.stateData['position']
        dx = x1 - x0
        dy = abs(y1 - 1) - abs(y0 - 1)
        reward = -dx - dy

        # If past the maximum step count, stop the episode
        done = self.stepCount >= self.maxSteps

        return self.img.transpose(), reward, done, self.stateData

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

        if self.stateData is None:
            return

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

        # Display position/state information
        pos = self.stateData['position']
        self.textLabel.text = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2])
        self.textLabel.draw()

        self.window.flip()
