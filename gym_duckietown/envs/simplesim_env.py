import os
import math
import time
import numpy as np

import pyglet
from pyglet.image import ImageData
from pyglet.gl import *
from ctypes import byref, POINTER

import gym
from gym import error, spaces, utils
from gym.utils import seeding

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

def loadTexture(texName):
    # Assemble the absolute path to the texture
    absPathModule = os.path.realpath(__file__)
    moduleDir, _ = os.path.split(absPathModule)
    texPath = os.path.join(moduleDir, texName)

    img = pyglet.image.load(texPath)
    tex = img.get_texture()
    glEnable(tex.target)
    glBindTexture(tex.target, tex.id)
    glTexImage2D(
        GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0,
        GL_RGBA, GL_UNSIGNED_BYTE,
        img.get_image_data().get_data('RGBA', img.width * 4)
    )

    return tex

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
            shape=IMG_SHAPE
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

        # Load the road texture
        self.roadTex = loadTexture('road.png')

        # Create the framebuffer (rendering target)
        self.fb = GLuint(0)
        glGenFramebuffers(1, byref(self.fb))
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb)

        # Create the texture to render into
        self.fbTex = GLuint(0)
        glGenTextures(1, byref(self.fbTex))
        glBindTexture(GL_TEXTURE_2D, self.fbTex)
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            0,
            GL_RGBA,
            GL_FLOAT,
            None
        )
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);

        # Attach the texture to the framebuffer
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, self.fbTex, 0)
        res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        assert res == GL_FRAMEBUFFER_COMPLETE

        # Generate a depth  buffer and bind it to the frame buffer
        depthBuffer = GLuint(0);
        glGenRenderbuffers( 1, byref(depthBuffer))
        glBindRenderbuffer(GL_RENDERBUFFER, depthBuffer)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, CAMERA_WIDTH, CAMERA_HEIGHT)
        glBindRenderbuffer(GL_RENDERBUFFER, 0);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depthBuffer);

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0)

        # Starting position
        self.startPos = (-0.25, 0.2, 0.5)

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

        # Left
        if action[0] < 0:
            self.curPos = (x - 0.06, y, z)
        # Right
        elif action[1] < 0:
            self.curPos = (x + 0.06, y, z)
        # Forward
        else:
            self.curPos = (x, y, z - 0.06)

        # End of lane, to the right
        targetPos = (0.25, 0.2, -2.0)

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
        #fbBinding = GLint(0)
        #glGetIntegerv(GL_FRAMEBUFFER_BINDING, byref(fbBinding))
        #print('current fb binding: %s' % fbBinding)

        isFb = glIsFramebuffer(self.fb)
        assert isFb == True

        # Bind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fb);
        glViewport(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)

        glClearColor(0.4, 0.4, 0.4, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Set modelview matrix
        glMatrixMode(gl.GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            *self.curPos, # eye position
            # Target
            self.curPos[0], self.curPos[1], self.curPos[2] - 1,
            # Up vector
            0, 1.0, 0.0
        )

        # Set the projection matrix
        glMatrixMode(gl.GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, CAMERA_WIDTH / float(CAMERA_HEIGHT), 0.05, 100.0)

        verts = [
            -0.5, 0.0,  0,
            -0.5, 0.0, -1,
             0.5, 0.0, -1,
             0.5, 0.0,  0
        ]
        texCoords = [
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0,
            1.0, 0.0
        ]
        vlist = pyglet.graphics.vertex_list(4, ('v3f', verts), ('t2f', texCoords))

        glEnable(GL_TEXTURE_2D)
        glBindTexture(self.roadTex.target, self.roadTex.id)

        for i in range(3):
            vlist.draw(GL_QUADS)
            glTranslatef(0, 0, -1)

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        data = np.empty((CAMERA_HEIGHT, CAMERA_WIDTH, 3), dtype=np.float32)
        glReadPixels(
            0,
            0,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            GL_RGB,
            GL_FLOAT,
            data.ctypes.data_as(POINTER(GLfloat))
        )
        data = np.uint8(data * 255)
        data = np.flip(data, axis=0)

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        return data

    def _render(self, mode='human', close=False):
        # Render the observation
        img = self._renderObs()

        if mode == 'rgb_array':
            return img

        if close:
            if self.window:
                self.window.close()
            return

        if self.window is None:
            self.window = pyglet.window.Window(width=WINDOW_SIZE, height=WINDOW_SIZE)

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
        width = img.shape[0]
        height = img.shape[1]
        imgData = ImageData(
            width,
            height,
            'RGB',
            img.tobytes(),
            pitch = width * 3,
        )
        glPushMatrix()
        glTranslatef(0, WINDOW_SIZE, 0)
        glScalef(1, -1, 1)
        imgData.blit(0, 0, 0, WINDOW_SIZE, WINDOW_SIZE)
        glPopMatrix()

        # Display position/state information
        pos = self.curPos
        self.textLabel.text = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2])
        self.textLabel.draw()
