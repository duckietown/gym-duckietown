import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import numpy
import zmq

import pyglet
from pyglet.image import ImageData
from pyglet.gl import glPushMatrix, glPopMatrix, glScalef, glTranslatef

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
    CAMERA_WIDTH = 100
    CAMERA_HEIGHT = 100

    # Camera image shape
    IMG_SHAPE = (CAMERA_WIDTH, CAMERA_HEIGHT, 3)

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
        self.maxSteps = 1000

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

        # Return first observation
        return self.img

    def _seed(self, seed=None):
        """
        The seed function sets the random elements of the environment.
        """

        self.np_random, _ = seeding.np_random(seed)

        # TODO: can we ask the server to generate a new random map here?
        # TODO: does the server decide our starting position on the map?

        return [seed]

    def _step(self, action):
        assert isinstance(action, tuple) and len(action) == 2

        self.stepCount += 1

        # Send the action to the server
        self.socket.send_json({
            "command":"action",
            "values": action
        })

        # Receive state data (position, etc)
        self.stateData = self.socket.recv_json()

        # Receive a camera image from the server
        self.img = recvArray(self.socket)

        # TODO: figure out what the reward should be based on world data
        reward = 0
        done = False
        if self.stateData['inside_lane'] == False:
            reward = -1
        if self.stateData['colliding'] == True:
            reward = -2

        if self.stepCount >= self.maxSteps:
            done = True

        return self.img, reward, done, self.stateData

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
            self.img.data.__str__()
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
