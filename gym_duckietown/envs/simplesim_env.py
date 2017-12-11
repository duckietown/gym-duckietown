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
IMG_SHAPE = (CAMERA_WIDTH, CAMERA_HEIGHT, 3)

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

def createFBO():
    """Create a frame buffer object"""

    # Create the framebuffer (rendering target)
    fbId = GLuint(0)
    glGenFramebuffers(1, byref(fbId))
    glBindFramebuffer(GL_FRAMEBUFFER, fbId)

    # Create the texture to render into
    fbTex = GLuint(0)
    glGenTextures(1, byref(fbTex))
    glBindTexture(GL_TEXTURE_2D, fbTex)
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
    glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, fbTex, 0)
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

    return fbId, fbTex

class SimpleSimEnv(gym.Env):
    """Simplistic road simulator to test RL training"""

    metadata = {
        'render.modes': ['human', 'rgb_array', 'app'],
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
        self.maxSteps = 80

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

        # Create a frame buffer object
        self.fbId, self.fbTex = createFBO()

        # Starting position
        self.startPos = (-0.25, 0.2, 0.5)

        # Initialize the state
        self.seed()
        self.reset()

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
            x -= 0.06
        # Right
        elif action[1] < 0:
            x += 0.06
        # Forward
        else:
            z -= 0.06

        # Add a small amount of noise to the position
        # This will randomize the movement dynamics
        posNoise = self.np_random.uniform(low=-0.01, high=0.01, size=(3,))
        x += posNoise[0]
        #y += posNoise[1]
        z += posNoise[2]
        self.curPos = (x, y, z)

        # End of lane, to the right
        targetPos = (0.25, 0.2, -2.0)

        dx = x - targetPos[0]
        dz = z - targetPos[2]

        dist = abs(dx) + abs(dz)
        reward = max(0, 3 - dist)

        done = False

        # If the objective is reached
        if dist <= 0.10:
            reward = 1000 - self.stepCount
            done = True

        # If the agent goes too far left or right,
        # end the episode early
        if dx < -1.00 or dx > 0.50:
            reward = 0
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

        isFb = glIsFramebuffer(self.fbId)
        assert isFb == True

        # Bind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbId);
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
        data = np.empty(shape=IMG_SHAPE, dtype=np.float32)
        glReadPixels(
            0,
            0,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            GL_RGB,
            GL_FLOAT,
            data.ctypes.data_as(POINTER(GLfloat))
        )

        # Add noise to the image
        # TODO: adjustable noise coefficient
        noise = self.np_random.normal(size=IMG_SHAPE, loc=0, scale=0.05)
        data = np.clip(data + noise, a_min=0, a_max=1)

        # Convert the image to RGB888
        data = np.uint8(data * 255)

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
        imgData.blit(0, 0, 0, WINDOW_SIZE, WINDOW_SIZE)

        # Display position/state information
        pos = self.curPos
        self.textLabel.text = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2])
        self.textLabel.draw()

        if mode == 'human':
            self.window.flip()
