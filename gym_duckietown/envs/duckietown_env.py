import gym
from gym import error, spaces, utils
from gym.utils import seeding
import math
import time
from subprocess import call, check_call
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

def recvArray(socket):
    """Receive a numpy array over zmq"""
    md = socket.recv_json()
    msg = socket.recv(copy=True, track=False)
    buf = buffer(msg)
    A = numpy.frombuffer(buf, dtype=md['dtype'])
    return A.reshape(md['shape'])

class DiscreteEnv(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        super(DiscreteEnv, self).__init__(env)

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

class DuckietownEnv(gym.Env):
    """
    OpenAI gym environment wrapper for the Duckietown simulation.
    Connects to ROS/Gazebo through ZeroMQ
    """

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    # Port to connect to on the server
    SERVER_PORT = 7777

    # Camera image size
    CAMERA_WIDTH = 64
    CAMERA_HEIGHT = 64

    # Camera image shape
    IMG_SHAPE = (3, CAMERA_WIDTH, CAMERA_HEIGHT)

    def __init__(
        self,
        serverAddr="localhost",
        serverPort=SERVER_PORT,
        startDocker=False):

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

        # Last received state data
        self.stateData = None

        # Last received image
        self.img = None

        # If a docker image should be started
        if startDocker:
            self.docker_name = 'duckietown_%s' % serverPort

            # Kill old containers, if running
            call([
                'docker', 'rm', '-f', self.docker_name
            ])

            print('starting docker container %s' % self.docker_name)
            check_call([
                'docker', 'run', '-d',
                '-p', '%s:7777' % serverPort,
                '--name', self.docker_name,
                '-it', 'yanjundream/duckietown_simulator'
            ])

            print('%s starting gazebo...' % self.docker_name)
            check_call([
                'docker', 'exec', '-d', self.docker_name,
                'bash', '-c',
                'cd / && source ./start.sh && ./run_gazebo.sh'
            ])

            # H4xxx, need a way to wait until we know this is started
            time.sleep(15)

            print('%s starting gym server node...' % self.docker_name)
            check_call([
                'docker', 'exec', '-d', self.docker_name,
                'bash', '-c',
                'cd / && source ./start.sh && python2 ./gym-gazebo-server.py'
            ])

            # H4xxx, need a way to wait until we know this is started
            time.sleep(15)

        # Connect to the Gym bridge ROS node
        context = zmq.Context()
        self.socket = context.socket(zmq.PAIR)
        self.socket.connect("tcp://%s:%s" % (serverAddr, serverPort))

        # Initialize the state
        self.reset()
        self.seed()

    def _close(self):
        if hasattr(self, 'docker_name'):
            print('killing docker container %s' % self.docker_name)
            call(['docker', 'rm', '-f', self.docker_name])

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

        x, y, z = self.stateData['position']

        # End of lane, to the right
        #targetPos = (0.0, 1.12)

        # End of lane, centered on yellow markers
        targetPos = (0.0, 1.00)

        dx = x - targetPos[0]
        dy = y - targetPos[1]

        dist = abs(dx) + abs(dy)
        reward = -dist

        #print('x=%.2f, y=%.2f' % (x, y))
        #print('  d=%.2f' % dist)
        #print('  a=%s' % str(action))

        done = False

        # If the objective is reached
        if dist <= 0.05:
            reward = 1000
            done = True

        # If the maximum time step count is reached
        if self.stepCount >= self.maxSteps:
            done = True

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
