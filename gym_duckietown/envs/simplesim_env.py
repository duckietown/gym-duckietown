import os
import math
import time
import numpy as np

import pyglet
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

# Horizon/wall color
HORIZON_COLOR = np.array([0.75, 0.70, 0.70])

# Ground/floor color
GROUND_COLOR = np.array([0.2, 0.2, 0.2])

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108

# Distance betwen robot wheels (10.2cm)
WHEEL_DIST = 0.102

# Road tile dimensions (2ft x 2ft, 61cm wide)
ROAD_TILE_SIZE = 0.61

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

def rotatePoint(px, py, cx, cy, theta):
    dx = px - cx
    dy = py - cy

    dx = dx * math.cos(theta) - dy * math.sin(theta)
    dy = dy * math.cos(theta) + dx * math.sin(theta)

    return cx + dx, cy + dy

class SimpleSimEnv(gym.Env):
    """Simplistic road simulator to test RL training"""

    metadata = {
        'render.modes': ['human', 'rgb_array', 'app'],
        'video.frames_per_second' : 30
    }

    def __init__(self,
        maxSteps=200,
        imgNoiseScale=0
    ):
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

        # Maximum number of steps per episode
        self.maxSteps = maxSteps

        # Amount of image noise to produce (standard deviation)
        self.imgNoiseScale = imgNoiseScale

        # Array to render the image into
        self.imgArray = np.zeros(shape=IMG_SHAPE, dtype=np.float32)

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # For displaying text
        self.textLabel = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_SIZE - 19
        )

        # Load the road texture
        self.roadTex = loadTexture('road.png')

        # Load the road turn texture
        self.roadDiagTex = loadTexture('road_diag.png')

        # Create a frame buffer object
        self.fbId, self.fbTex = createFBO()

        # Create the vertex list for our road quad
        halfSize = ROAD_TILE_SIZE / 2
        verts = [
            -halfSize, 0.0, -halfSize,
             halfSize, 0.0, -halfSize,
             halfSize, 0.0,  halfSize,
            -halfSize, 0.0,  halfSize
        ]
        texCoords = [
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0
        ]
        self.roadVList = pyglet.graphics.vertex_list(4, ('v3f', verts), ('t2f', texCoords))

        # Create the vertex list for the ground quad
        verts = [
            -1, -0.05,  1,
            -1, -0.05, -1,
             1, -0.05, -1,
             1, -0.05,  1
        ]
        self.groundVList = pyglet.graphics.vertex_list(4, ('v3f', verts))

        # Tile grid size
        self.gridWidth = 6
        self.gridHeight = 6
        self.grid = [None] * self.gridWidth * self.gridHeight

        # Assemble the initial grid
        # Left turn
        self._setGrid(0, 0, ('diag_left', 3))
        # First straight
        self._setGrid(0, 1, ('linear', 0))
        self._setGrid(0, 2, ('linear', 0))
        self._setGrid(0, 3, ('linear', 0))
        self._setGrid(0, 4, ('linear', 0))
        # Left turn
        self._setGrid(0, 5, ('diag_left', 0))
        # Second straight
        self._setGrid(1, 5, ('linear', 3))
        self._setGrid(2, 5, ('linear', 3))
        self._setGrid(3, 5, ('linear', 3))
        self._setGrid(4, 5, ('linear', 3))
        # Third turn
        self._setGrid(5, 5, ('diag_left', 1))
        # Third straight
        self._setGrid(5, 4, ('linear', 2))
        self._setGrid(5, 3, ('linear', 2))
        self._setGrid(5, 2, ('linear', 2))
        self._setGrid(5, 1, ('linear', 2))
        # Fourth turn
        self._setGrid(5, 0, ('diag_left', 2))
        # Last straight
        self._setGrid(1, 0, ('linear', 1))
        self._setGrid(2, 0, ('linear', 1))
        self._setGrid(3, 0, ('linear', 1))
        self._setGrid(4, 0, ('linear', 1))

        # Initialize the state
        self.seed()
        self.reset()

    def _close(self):
        pass

    def _setGrid(self, i, j, tile):
        assert i >= 0 and i < self.gridWidth
        assert j >= 0 and j < self.gridHeight
        self.grid[j * self.gridWidth + i] = tile

    def _getGrid(self, i, j):
        assert i >= 0 and i < self.gridWidth
        assert j >= 0 and j < self.gridHeight
        return self.grid[j * self.gridWidth + i]

    def _perturb(self, val, scale=0.1):
        """Add noise to a value"""
        assert scale >= 0
        assert scale < 1

        if isinstance(val, np.ndarray):
            noise = self.np_random.uniform(low=1-scale, high=1+scale, size=val.shape)
        else:
            noise = self.np_random.uniform(low=1-scale, high=1+scale)

        return val * noise

    def _reset(self):
        # Step count since episode start
        self.stepCount = 0

        # Horizon color
        self.horizonColor = self._perturb(HORIZON_COLOR)

        # Ground color
        self.groundColor = self._perturb(GROUND_COLOR, 0.1)

        # Distance between the robot's wheels
        self.wheelDist = self._perturb(WHEEL_DIST)

        # Distance bewteen camera and ground
        self.camHeight = self._perturb(CAMERA_FLOOR_DIST, 0.05)

        # Randomize the starting position
        self.curPos = np.array([
            self.np_random.uniform(-0.25, 0.25),
            self.camHeight,
            0.40
        ])

        # Starting direction angle
        self.curAngle = self._perturb(math.pi/2, 0.2)

        # Get the first camera image
        obs = self._renderObs()

        # Return first observation
        return obs

    def _seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)

        return [seed]

    def getDirVec(self):
        x = math.cos(self.curAngle)
        z = math.sin(self.curAngle)

        return (x, 0, z)

    def getLeftVec(self):
        x = math.sin(self.curAngle)
        z = -math.cos(self.curAngle)

        return (x, 0, z)

    def _updatePos(self, wheelVels, deltaTime):
        """
        Update the position of the robot, simulating differential drive
        """

        Vl, Vr = wheelVels
        l = self.wheelDist

        # If the wheel velocities are the same, then there is no rotation
        if Vl == Vr:
            dx, dy, dz = self.getDirVec()
            px, py, pz = self.curPos
            self.curPos = np.array([
                px + dx * Vl * deltaTime,
                py + dy * Vl * deltaTime,
                pz + dz * Vl * deltaTime
            ])
            return

        # Compute the angular rotation velocity about the ICC (center of curvature)
        w = (Vr - Vl) / l

        # Compute the distance to the center of curvature
        r = (l * (Vl + Vr)) / (2 * (Vl - Vr))

        # Compute the rotatio angle for this time step
        rotAngle = w * deltaTime

        # Rotate the robot's position
        leftVec = self.getLeftVec()
        px, py, pz = self.curPos
        cx = px + leftVec[0] * -r
        cz = pz + leftVec[2] * -r
        npx, npz = rotatePoint(px, pz, cx, cz, -rotAngle)
        self.curPos = np.array([npx, py, npz])

        # Update the robot's angle
        self.curAngle -= rotAngle

    def _step(self, action):
        self.stepCount += 1

        # Update the robot's position
        self._updatePos(action, 0.1)

        # Add a small amount of noise to the position
        # This will randomize the movement dynamics
        self.curPos = self._perturb(self.curPos, 0.01)
        self.curPos[1] = self.camHeight

        # Get the current position
        x, y, z = self.curPos

        # Compute the grid position of the agent
        xR = x / ROAD_TILE_SIZE
        i = int(xR + (0.5 if xR > 0 else -0.5))
        zR = z / ROAD_TILE_SIZE
        j = int(zR + (0.5 if zR > 0 else -0.5))

        # Generate the current camera image
        obs = self._renderObs()

        # If the agent is outside of the grid
        if i < 0 or i >= self.gridWidth or j < 0 or j >= self.gridHeight:
            reward = -10
            done = True
            return obs, reward, done, {}

        cell = self._getGrid(i, j)

        # If there is nothing at this grid cell
        if cell == None:
            reward = -10
            done = True
            return obs, reward, done, {}

        # If the maximum time step count is reached
        if self.stepCount >= self.maxSteps:
            done = True
            reward = 0
            return obs, reward, done, {}

        kind, angle = cell

        # Compute the fractional grid position
        # This is the position of the agent within the cell
        xM = xR + 0.5 - i
        zM = zR + 0.5 - j

        xC, zC = rotatePoint(xM, zM, 0.5, 0.5, -math.pi / 2 * angle)

        #print('i=%s, j=%s' % (i, j))
        #print('xM=%s, zM=%s' % (xM, zM))
        #print('xC=%s, zC=%s' % (xC, zC))

        reward = 0
        done = False

        if kind == 'linear':
            if xC < 0.5:
                #print('in right lane')
                reward = 1
            else:
                #print('in left lane')
                reward = -1

        elif kind == 'diag_left':
            if xC - zC < 0.5:
                #print('in right lane')
                reward = 1
            else:
                reward = -1

        # Bonus for moving forward
        if action[0] > 0 and action[1] > 0:
            reward += 1

        return obs, reward, done, {}

    def _renderObs(self):
        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        #pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        isFb = glIsFramebuffer(self.fbId)
        assert isFb == True

        # Bind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, self.fbId);
        glViewport(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)

        glClearColor(*self.horizonColor, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, CAMERA_WIDTH / float(CAMERA_HEIGHT), 0.05, 100.0)

        # Set modelview matrix
        x, y, z = self.curPos
        dx, dy, dz = self.getDirVec()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(
            # Eye position
            x,
            y + self.np_random.uniform(low=-0.006, high=0.006),
            z,
            # Target
            x + dx,
            y + dy,
            z + dz,
            # Up vector
            0, 1.0, 0.0
        )

        # Draw the ground quad
        glDisable(GL_TEXTURE_2D)
        glColor3f(*self.groundColor)
        glPushMatrix()
        glScalef(50, 1, 50)
        self.groundVList.draw(GL_QUADS)
        glPopMatrix()

        # Draw the road quads
        glEnable(GL_TEXTURE_2D)
        glColor3f(1, 1, 1)

        # For each grid tile
        for j in range(self.gridHeight):
            for i in range(self.gridWidth):
                # Get the tile type and angle
                tile = self._getGrid(i, j)
                if tile == None:
                    continue
                kind, angle = tile
                glPushMatrix()
                glTranslatef(i * ROAD_TILE_SIZE, 0, j * ROAD_TILE_SIZE)
                glRotatef(angle * 90, 0, 1, 0)

                # Bind the appropriate texture
                if kind == 'linear':
                    glBindTexture(self.roadTex.target, self.roadTex.id)
                elif kind == 'diag_left':
                    glBindTexture(self.roadDiagTex.target, self.roadDiagTex.id)
                else:
                    assert False

                self.roadVList.draw(GL_QUADS)
                glPopMatrix()

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        glReadPixels(
            0,
            0,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            GL_RGB,
            GL_FLOAT,
            self.imgArray.ctypes.data_as(POINTER(GLfloat))
        )

        # Add noise to the image
        if self.imgNoiseScale > 0:
            noise = self.np_random.normal(
                size=IMG_SHAPE,
                loc=0,
                scale=self.imgNoiseScale
            )
            np.clip(self.imgArray + noise, a_min=0, a_max=1, out=self.imgArray)

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        return self.imgArray

    def _render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        # Render the observation
        img = self._renderObs()

        if mode == 'rgb_array':
            return img

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
        width = img.shape[0]
        height = img.shape[1]
        img = np.uint8(img * 255)
        imgData = pyglet.image.ImageData(
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
