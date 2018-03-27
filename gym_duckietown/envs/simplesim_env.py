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
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120

# Camera image shape
IMG_SHAPE = (CAMERA_HEIGHT, CAMERA_WIDTH, 3)

# Horizon/wall color
HORIZON_COLOR = np.array([0.64, 0.71, 0.28])

# Road color multiplier
ROAD_COLOR = np.array([0.79, 0.88, 0.53])

# Ground/floor color
GROUND_COLOR = np.array([0.15, 0.15, 0.15])

# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 15

# Camera field of view angle in the Y direction
CAMERA_FOV_Y = 42

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108

# Forward distance between camera and center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066

# Distance betwen robot wheels (10.2cm)
WHEEL_DIST = 0.102

# Road tile dimensions (2ft x 2ft, 61cm wide)
ROAD_TILE_SIZE = 0.61

# Maximum forward robot speed in meters/second
ROBOT_SPEED = 0.45

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

def createFrameBuffers():
    """Create the frame buffer objects"""

    # Create the multisampled frame buffer (rendering target)
    multiFBO = GLuint(0)
    glGenFramebuffers(1, byref(multiFBO))
    glBindFramebuffer(GL_FRAMEBUFFER, multiFBO)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
    # support multisampling on frame buffer objects
    try:
        # Create a multisampled texture to render into
        numSamples = 32
        fbTex = GLuint(0)
        glGenTextures( 1, byref(fbTex));
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, fbTex);
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            numSamples,
            GL_RGBA32F,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            True
        );
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D_MULTISAMPLE,
            fbTex,
            0
        );
        res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        assert res == GL_FRAMEBUFFER_COMPLETE
    except:
        print('Falling back to non-multisampled frame buffer')

        # Create a plain texture texture to render into
        fbTex = GLuint(0)
        glGenTextures( 1, byref(fbTex));
        glBindTexture(GL_TEXTURE_2D, fbTex);
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
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            fbTex,
            0
        );
        res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
        assert res == GL_FRAMEBUFFER_COMPLETE

    # Create the frame buffer used to resolve the final render
    finalFBO = GLuint(0)
    glGenFramebuffers(1, byref(finalFBO))
    glBindFramebuffer(GL_FRAMEBUFFER, finalFBO)

    # Create the texture used to resolve the final render
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
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        fbTex,
        0
    )
    res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
    assert res == GL_FRAMEBUFFER_COMPLETE

    # Unbind the frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return multiFBO, finalFBO

def rotatePoint(px, py, cx, cy, theta):
    dx = px - cx
    dy = py - cy

    dx = dx * math.cos(theta) - dy * math.sin(theta)
    dy = dy * math.cos(theta) + dx * math.sin(theta)

    return cx + dx, cy + dy

def rotMatrix(axis, angle):
    """
    Rotation matrix for a counterclockwise rotation around the given axis
    """

    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

def bezierPoint(cps, t):
    """
    Cubic Bezier curve interpolation
    B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    """

    p  = ((1-t)**3) * cps[0,:]
    p += 3 * t * ((1-t)**2) * cps[1,:]
    p += 3 * (t**2) * (1-t) * cps[2,:]
    p += (t**3) * cps[3,:]

    return p

def bezierTangent(cps, t):
    """
    Tangent of a cubic Bezier curve (first order derivative)
    B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
    """

    p  = 3 * ((1-t)**2) * (cps[1,:] - cps[0,:])
    p += 6 * (1-t) * t * (cps[2,:] - cps[1,:])
    p += 3 * (t ** 2) * (cps[3,:] - cps[2,:])

    norm = np.linalg.norm(p)
    p /= norm

    return p

def bezierClosest(cps, p, t_bot=0, t_top=1, n=8):
    mid = (t_bot + t_top) * 0.5

    if n == 0:
        return mid

    p_bot = bezierPoint(cps, t_bot)
    p_top = bezierPoint(cps, t_top)

    d_bot = np.linalg.norm(p_bot - p)
    d_top = np.linalg.norm(p_top - p)

    if d_bot < d_top:
        return bezierClosest(cps, p, t_bot, mid, n-1)

    return bezierClosest(cps, p, mid, t_top, n-1)

def drawBezier(cps, n = 20):
    pts = [bezierPoint(cps, i/(n-1)) for i in range(0,n)]
    glColor3f(1,0,0)
    glBegin(GL_LINE_STRIP)
    for p in pts:
        glVertex3f(*p)
    glEnd()
    glColor3f(1,1,1)

class SimpleSimEnv(gym.Env):
    """
    Simple road simulator to test RL training.
    Draws a road with turns using OpenGL, and simulates
    basic differential-drive dynamics.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'app'],
        'video.frames_per_second' : 30
    }

    def __init__(self,
        maxSteps=600,
        imgNoiseScale=0
    ):
        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

        # We observe an RGB image with pixels in [0, 255]
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=IMG_SHAPE,
            dtype=np.float32
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
            y = WINDOW_HEIGHT - 19
        )

        # Load the road textures
        self.roadTex = loadTexture('road_plain.png')
        self.roadStopTex = loadTexture('road_stop.png')
        self.roadLeftTex = loadTexture('road_left.png')
        self.roadRightTex = loadTexture('road_right.png')

        # Create a frame buffer object
        self.multiFBO, self.finalFBO = createFrameBuffers()

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
        # Left
        self._setGrid(0, 3, ('diag_left', 0))
        # Straight, towards the left
        self._setGrid(1, 3, ('linear', 1))
        # Right
        self._setGrid(2, 3, ('diag_right', 1))
        # Forward towads the back
        self._setGrid(2, 4, ('linear', 0))
        # Left turn
        self._setGrid(2, 5, ('diag_left', 0))

        # Second straight, towards the left
        self._setGrid(3, 5, ('linear', 1))
        self._setGrid(4, 5, ('linear', 1))
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
        self._setGrid(1, 0, ('linear', 3))
        self._setGrid(2, 0, ('linear', 3))
        self._setGrid(3, 0, ('linear', 3))
        self._setGrid(4, 0, ('linear', 3))

        # Initialize the state
        self.seed()
        self.reset()

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _setGrid(self, i, j, tile):
        assert i >= 0 and i < self.gridWidth
        assert j >= 0 and j < self.gridHeight
        self.grid[j * self.gridWidth + i] = tile

    def _getGrid(self, i, j):
        if i < 0 or i >= self.gridWidth:
            return None
        if j < 0 or j >= self.gridWidth:
            return None
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

    def _getGridPos(self, x, z):
        """
        Compute the tile indices (i,j) for a given (x,z) world position
        """

        # Compute the grid position of the agent
        xR = x / ROAD_TILE_SIZE
        i = int(xR + (0.5 if xR > 0 else -0.5))
        zR = z / ROAD_TILE_SIZE
        j = int(zR + (0.5 if zR > 0 else -0.5))

        return i, j

    def _getCurve(self, i, j):
        """
        Get the Bezier curve control points for a given tile
        """

        tile = self._getGrid(i, j)
        assert tile is not None

        kind, angle = tile

        if kind.startswith('linear'):
            pts = np.array([
                [-0.20, 0,-0.50],
                [-0.20, 0,-0.25],
                [-0.20, 0, 0.25],
                [-0.20, 0, 0.50],
            ]) * ROAD_TILE_SIZE
        elif kind == 'diag_left':
            pts = np.array([
                [-0.20, 0,-0.50],
                [-0.20, 0, 0.00],
                [ 0.00, 0, 0.20],
                [ 0.50, 0, 0.20],
            ]) * ROAD_TILE_SIZE
        elif kind == 'diag_right':
            pts = np.array([
                [-0.20, 0,-0.50],
                [-0.20, 0,-0.20],
                [-0.30, 0,-0.20],
                [-0.50, 0,-0.20],
            ]) * ROAD_TILE_SIZE
        else:
            assert False, kind

        mat = rotMatrix(np.array([0, 1, 0]), angle * math.pi / 2)

        pts = np.matmul(pts, mat)
        pts += np.array([i * ROAD_TILE_SIZE, 0, j * ROAD_TILE_SIZE])

        return pts

    def getDirVec(self):
        x = math.cos(self.curAngle)
        z = math.sin(self.curAngle)
        return np.array([x, 0, z])

    def getLeftVec(self):
        x = math.sin(self.curAngle)
        z = -math.cos(self.curAngle)
        return np.array([x, 0, z])

    def getLanePos(self):
        """
        Get the position of the agent relative to the center of the right lane
        """

        x, _, z = self.curPos
        i, j = self._getGridPos(x, z)

        # Get the closest point along the right lane's Bezier curve
        cps = self._getCurve(i, j)
        t = bezierClosest(cps, self.curPos)
        point = bezierPoint(cps, t)

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = self.getDirVec()
        tangent = bezierTangent(cps, t)
        dotDir = np.dot(dirVec, tangent)

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        posVec = self.curPos - point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(tangent, upVec)
        signedDist = np.dot(posVec, rightVec)

        # Compute the signed angle between the direction and curve tangent
        # Right of the tangent is negative, left is positive
        angle = math.acos(dotDir)
        angle *= 180 / math.pi
        if np.dot(dirVec, rightVec) < 0:
            angle *= -1

        return signedDist, dotDir, angle

    def reset(self):
        # Step count since episode start
        self.stepCount = 0

        # Horizon color
        self.horizonColor = self._perturb(HORIZON_COLOR)

        # Ground color
        self.groundColor = self.np_random.uniform(low=0.05, high=0.6, size=(3,))

        # Road color multiplier
        self.roadColor = self._perturb(ROAD_COLOR, 0.2)

        # Distance between the robot's wheels
        self.wheelDist = self._perturb(WHEEL_DIST)

        # Distance bewteen camera and ground
        self.camHeight = self._perturb(CAMERA_FLOOR_DIST, 0.08)

        # Angle at which the camera is pitched downwards
        self.camAngle = self._perturb(CAMERA_ANGLE, 0.2)

        # Field of view angle of the camera
        self.camFovY = self._perturb(CAMERA_FOV_Y, 0.2)

        # Randomize the starting position and angle
        # Pick a random starting tile and angle, do rejection sampling
        while True:
            self.curPos = np.array([
                self.np_random.uniform(-0.5, self.gridWidth - 0.5) * ROAD_TILE_SIZE,
                0,
                self.np_random.uniform(-0.5, self.gridHeight - 0.5) * ROAD_TILE_SIZE,
            ])

            i, j = self._getGridPos(self.curPos[0], self.curPos[2])
            tile = self._getGrid(i, j)

            if tile is None:
                continue

            kind, angle = tile

            # Choose a random direction
            self.curAngle = self.np_random.uniform(0, 2 * math.pi)

            dist, dotDir, angle = self.getLanePos()
            if dist < -0.20 or dist > 0.12:
                continue
            if angle < -30 or angle > 30:
                continue

            break

        """
        self.curPos = np.array([
            self.np_random.uniform(-0.20, 0.20),
            0.0,
            self.np_random.uniform(0.25, 0.75),
        ])
        self.curAngle = (math.pi/2) + (self.np_random.uniform(-20, 20) * math.pi/180)
        """

        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = 12
        verts = []
        colors = []
        for i in range(0, 3 * numTris):
            p = self.np_random.uniform(low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,))
            c = self.np_random.uniform(low=0, high=0.9)
            c = self._perturb(np.array([c, c, c]), 0.1)
            verts += [p[0], p[1], p[2]]
            colors += [c[0], c[1], c[2]]
        self.triVList = pyglet.graphics.vertex_list(3 * numTris, ('v3f', verts), ('c3f', colors) )

        # Get the first camera image
        obs = self._renderObs()

        # Return first observation
        return obs

    def _updatePos(self, wheelVels, deltaTime):
        """
        Update the position of the robot, simulating differential drive
        """

        Vl, Vr = wheelVels
        l = self.wheelDist

        # If the wheel velocities are the same, then there is no rotation
        if Vl == Vr:
            self.curPos += deltaTime * Vl * self.getDirVec()
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

    def step(self, action):
        self.stepCount += 1

        # Update the robot's position
        self._updatePos(action * ROBOT_SPEED * 1, 0.1)

        # Add a small amount of noise to the position
        # This will randomize the movement dynamics
        posNoise = self.np_random.uniform(low=-0.005, high=0.005, size=(3,))
        self.curPos += posNoise
        self.curPos[1] = 0

        # Get the current position
        x, y, z = self.curPos

        # Generate the current camera image
        obs = self._renderObs()

        # Compute the grid position of the agent
        i, j = self._getGridPos(x, z)
        tile = self._getGrid(i, j)

        # If there is nothing at this grid cell
        if tile == None:
            reward = -10
            done = True
            return obs, reward, done, {}

        # If the maximum time step count is reached
        if self.stepCount >= self.maxSteps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        # Get the position relative to the right lane tangent
        dist, dotDir, angle = self.getLanePos()
        reward = 1.0 * dotDir - 10.00 * abs(dist)

        return obs, reward, done, {}

    def _renderObs(self):
        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        #pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        # Bind the multisampled frame buffer
        glEnable(GL_MULTISAMPLE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.multiFBO);
        glViewport(0, 0, CAMERA_WIDTH, CAMERA_HEIGHT)

        glClearColor(*self.horizonColor, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.camFovY,
            CAMERA_WIDTH / float(CAMERA_HEIGHT),
            0.05,
            100.0
        )

        # Set modelview matrix
        x, _, z = self.curPos
        y = CAMERA_FLOOR_DIST + self.np_random.uniform(low=-0.006, high=0.006)
        dx, dy, dz = self.getDirVec()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(self.camAngle, 1, 0, 0)
        glTranslatef(0, 0, self._perturb(CAMERA_FORWARD_DIST))
        gluLookAt(
            # Eye position
            x,
            y,
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

        # Draw the ground/noise triangles
        self.triVList.draw(GL_TRIANGLES)

        # Draw the road quads
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        # For each grid tile
        for j in range(self.gridHeight):
            for i in range(self.gridWidth):
                # Get the tile type and angle
                tile = self._getGrid(i, j)

                if tile == None:
                    continue

                kind, angle = tile

                glColor3f(*self.roadColor)

                glPushMatrix()
                glTranslatef(i * ROAD_TILE_SIZE, 0, j * ROAD_TILE_SIZE)
                glRotatef(angle * 90, 0, 1, 0)

                # Bind the appropriate texture
                if kind == 'linear':
                    glBindTexture(self.roadTex.target, self.roadTex.id)
                elif kind == 'linear_stop':
                    glBindTexture(self.roadStopTex.target, self.roadStopTex.id)
                elif kind == 'diag_left':
                    glBindTexture(self.roadLeftTex.target, self.roadLeftTex.id)
                elif kind == 'diag_right':
                    glBindTexture(self.roadRightTex.target, self.roadRightTex.id)
                else:
                    assert False, kind

                self.roadVList.draw(GL_QUADS)
                glPopMatrix()

                #pts = self._getCurve(i, j)
                #drawBezier(pts, n = 20)

        # Resolve the multisampled frame buffer into the final frame buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.multiFBO);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.finalFBO);
        glBlitFramebuffer(
            0, 0,
            CAMERA_WIDTH, CAMERA_HEIGHT,
            0, 0,
            CAMERA_WIDTH, CAMERA_HEIGHT,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR
        );

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        glBindFramebuffer(GL_FRAMEBUFFER, self.finalFBO);
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

    def render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        # Render the observation
        img = self._renderObs()

        if mode == 'rgb_array':
            return img

        if self.window is None:
            config = pyglet.gl.Config(double_buffer=False)
            self.window = pyglet.window.Window(
                width=WINDOW_WIDTH,
                height=WINDOW_HEIGHT,
                resizable=False,
                config=config
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        glBindFramebuffer(GL_FRAMEBUFFER, 0);
        glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT)

        # Setup orghogonal projection
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

        # Draw the image to the rendering window
        width = img.shape[1]
        height = img.shape[0]
        img = np.uint8(img * 255)
        imgData = pyglet.image.ImageData(
            width,
            height,
            'RGB',
            img.ctypes.data_as(POINTER(GLubyte)),
            pitch=width * 3,
        )
        imgData.blit(
            0,
            0,
            0,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT
        )

        # Display position/state information
        pos = self.curPos
        self.textLabel.text = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2])
        self.textLabel.draw()

        # Force execution of queued commands
        glFlush()
