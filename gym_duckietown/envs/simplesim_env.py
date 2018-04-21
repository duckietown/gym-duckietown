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

def load_texture(texName):
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

def create_frame_buffers():
    """Create the frame buffer objects"""

    # Create the multisampled frame buffer (rendering target)
    multi_fbo = GLuint(0)
    glGenFramebuffers(1, byref(multi_fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, multi_fbo)

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
    final_fbo = GLuint(0)
    glGenFramebuffers(1, byref(final_fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, final_fbo)

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

    return multi_fbo, final_fbo

def rotate_point(px, py, cx, cy, theta):
    dx = px - cx
    dy = py - cy

    dx = dx * math.cos(theta) - dy * math.sin(theta)
    dy = dy * math.cos(theta) + dx * math.sin(theta)

    return cx + dx, cy + dy

def gen_rot_matrix(axis, angle):
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

def bezier_point(cps, t):
    """
    Cubic Bezier curve interpolation
    B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    """

    p  = ((1-t)**3) * cps[0,:]
    p += 3 * t * ((1-t)**2) * cps[1,:]
    p += 3 * (t**2) * (1-t) * cps[2,:]
    p += (t**3) * cps[3,:]

    return p

def bezier_tangent(cps, t):
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

def bezier_closest(cps, p, t_bot=0, t_top=1, n=8):
    mid = (t_bot + t_top) * 0.5

    if n == 0:
        return mid

    p_bot = bezier_point(cps, t_bot)
    p_top = bezier_point(cps, t_top)

    d_bot = np.linalg.norm(p_bot - p)
    d_top = np.linalg.norm(p_top - p)

    if d_bot < d_top:
        return bezier_closest(cps, p, t_bot, mid, n-1)

    return bezier_closest(cps, p, mid, t_top, n-1)

def bezier_draw(cps, n = 20):
    pts = [bezier_point(cps, i/(n-1)) for i in range(0,n)]
    glBegin(GL_LINE_STRIP)
    glColor3f(1, 0, 0)
    for i, p in enumerate(pts):
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

    def __init__(
        self,
        max_steps=600,
        img_noise_scale=0,
        draw_curve=False
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
        self.max_steps = max_steps

        # Amount of image noise to produce (standard deviation)
        self.img_noise_scale = img_noise_scale

        # Flag to draw the road curve
        self.draw_curve = draw_curve

        # Array to render the image into
        self.img_array = np.zeros(shape=IMG_SHAPE, dtype=np.float32)

        # Window for displaying the environment to humans
        self.window = None

        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1, height=1, visible=False)

        # For displaying text
        self.text_label = pyglet.text.Label(
            font_name="Arial",
            font_size=14,
            x = 5,
            y = WINDOW_HEIGHT - 19
        )

        # Load the road textures
        self.road_tex = load_texture('road_plain.png')
        self.road_stop_tex = load_texture('road_stop.png')
        self.road_stop_left_tex = load_texture('road_stop_left.png')
        self.road_stop_both_tex = load_texture('road_stop_both.png')
        self.road_left_tex = load_texture('road_left.png')
        self.road_right_tex = load_texture('road_right.png')
        self.road_3way_left_tex = load_texture('road_3way_left.png')
        self.black_tile_tex = load_texture('black_tile.png')

        # Create a frame buffer object
        self.multi_fbo, self.final_fbo = create_frame_buffers()

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
        self.road_vlist = pyglet.graphics.vertex_list(4, ('v3f', verts), ('t2f', texCoords))

        # Create the vertex list for the ground quad
        verts = [
            -1, -0.05,  1,
            -1, -0.05, -1,
             1, -0.05, -1,
             1, -0.05,  1
        ]
        self.ground_vlist = pyglet.graphics.vertex_list(4, ('v3f', verts))

        # Load the map
        self._load_map()

        # Initialize the state
        self.seed()
        self.reset()

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _load_map(self):
        """
        Load the map layout from a CSV file
        """

        import csv
        csvfile = open('gym_duckietown/envs/map.csv', 'r')
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        rows = list(reader)

        assert len(rows) > 0
        assert len(rows[0]) > 0

        # Create the grid
        self.grid_height = len(rows)
        self.grid_width = len(rows[0])
        self.grid = [None] * self.grid_width * self.grid_height

        for j, row in enumerate(reversed(rows)):

            assert len(row) == self.grid_width

            for i, cell in enumerate(reversed(row)):
                cell = cell.strip()

                if cell == 'empty':
                    continue

                if ':' in cell:
                    kind, angle = cell.split(':')
                    angle = int(angle)
                else:
                    kind = cell
                    angle = 0

                self._set_grid(i, j, (kind, angle))

    def _set_grid(self, i, j, tile):
        assert i >= 0 and i < self.grid_width
        assert j >= 0 and j < self.grid_height
        self.grid[j * self.grid_width + i] = tile

    def _get_grid(self, i, j):
        if i < 0 or i >= self.grid_width:
            return None
        if j < 0 or j >= self.grid_height:
            return None
        return self.grid[j * self.grid_width + i]

    def _perturb(self, val, scale=0.1):
        """Add noise to a value"""
        assert scale >= 0
        assert scale < 1

        if isinstance(val, np.ndarray):
            noise = self.np_random.uniform(low=1-scale, high=1+scale, size=val.shape)
        else:
            noise = self.np_random.uniform(low=1-scale, high=1+scale)

        return val * noise

    def _get_grid_pos(self, x, z):
        """
        Compute the tile indices (i,j) for a given (x,z) world position
        """

        # Compute the grid position of the agent
        xR = x / ROAD_TILE_SIZE
        i = int(xR + (0.5 if xR > 0 else -0.5))
        zR = z / ROAD_TILE_SIZE
        j = int(zR + (0.5 if zR > 0 else -0.5))

        return i, j

    def _get_curve(self, i, j):
        """
        Get the Bezier curve control points for a given tile
        """

        tile = self._get_grid(i, j)
        assert tile is not None

        kind, angle = tile

        if kind.startswith('linear') or kind.startswith('3way'):
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

        mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)

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
        i, j = self._get_grid_pos(x, z)

        # Get the closest point along the right lane's Bezier curve
        cps = self._get_curve(i, j)
        t = bezier_closest(cps, self.curPos)
        point = bezier_point(cps, t)

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = self.getDirVec()
        tangent = bezier_tangent(cps, t)
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
        self.step_count = 0

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
                self.np_random.uniform(-0.5, self.grid_width - 0.5) * ROAD_TILE_SIZE,
                0,
                self.np_random.uniform(-0.5, self.grid_height - 0.5) * ROAD_TILE_SIZE,
            ])

            i, j = self._get_grid_pos(self.curPos[0], self.curPos[2])
            tile = self._get_grid(i, j)

            if tile is None:
                continue

            kind, angle = tile

            if kind == 'black':
                continue

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
        obs = self._render_obs()

        # Return first observation
        return obs

    def _update_pos(self, wheelVels, deltaTime):
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
        npx, npz = rotate_point(px, pz, cx, cz, -rotAngle)
        self.curPos = np.array([npx, py, npz])

        # Update the robot's angle
        self.curAngle -= rotAngle

    def step(self, action):
        self.step_count += 1

        # Update the robot's position
        self._update_pos(action * ROBOT_SPEED * 1, 0.1)

        # Add a small amount of noise to the position
        # This will randomize the movement dynamics
        posNoise = self.np_random.uniform(low=-0.005, high=0.005, size=(3,))
        self.curPos += posNoise
        self.curPos[1] = 0

        # Get the current position
        x, y, z = self.curPos

        # Generate the current camera image
        obs = self._render_obs()

        # Compute the grid position of the agent
        i, j = self._get_grid_pos(x, z)
        tile = self._get_grid(i, j)

        # If there is no road at this grid cell
        if tile == None or tile[0] == 'black':
            reward = -10
            done = True
            return obs, reward, done, {}

        # If the maximum time step count is reached
        if self.step_count >= self.max_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        reward = 0
        done = False

        # Get the position relative to the right lane tangent
        dist, dotDir, angle = self.getLanePos()
        reward = 1.0 * dotDir - 10.00 * abs(dist)

        return obs, reward, done, {}

    def _render_obs(self):
        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        #pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        # Bind the multisampled frame buffer
        glEnable(GL_MULTISAMPLE)
        glBindFramebuffer(GL_FRAMEBUFFER, self.multi_fbo);
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
        self.ground_vlist.draw(GL_QUADS)
        glPopMatrix()

        # Draw the ground/noise triangles
        self.triVList.draw(GL_TRIANGLES)

        # Draw the road quads
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        # For each grid tile
        for j in range(self.grid_height):
            for i in range(self.grid_width):
                # Get the tile type and angle
                tile = self._get_grid(i, j)

                if tile == None:
                    continue

                kind, angle = tile

                glColor3f(*self.roadColor)

                glPushMatrix()
                glTranslatef(i * ROAD_TILE_SIZE, 0, j * ROAD_TILE_SIZE)
                glRotatef(angle * 90, 0, 1, 0)

                # Bind the appropriate texture
                if kind == 'linear':
                    glBindTexture(self.road_tex.target, self.road_tex.id)
                elif kind == 'linear_stop':
                    glBindTexture(self.road_stop_tex.target, self.road_stop_tex.id)
                elif kind == 'linear_stop_left':
                    glBindTexture(self.road_stop_left_tex.target, self.road_stop_left_tex.id)
                elif kind == 'linear_stop_both':
                    glBindTexture(self.road_stop_both_tex.target, self.road_stop_both_tex.id)
                elif kind == '3way_left':
                    glBindTexture(self.road_3way_left_tex.target, self.road_3way_left_tex.id)
                elif kind == 'diag_left':
                    glBindTexture(self.road_left_tex.target, self.road_left_tex.id)
                elif kind == 'diag_right':
                    glBindTexture(self.road_right_tex.target, self.road_right_tex.id)
                elif kind == 'black':
                    glBindTexture(self.black_tile_tex.target, self.black_tile_tex.id)
                else:
                    assert False, kind

                self.road_vlist.draw(GL_QUADS)
                glPopMatrix()

                if self.draw_curve and kind != "black":
                    pts = self._get_curve(i, j)
                    bezier_draw(pts, n = 20)

        # Resolve the multisampled frame buffer into the final frame buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, self.multi_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, self.final_fbo);
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
        glBindFramebuffer(GL_FRAMEBUFFER, self.final_fbo);
        glReadPixels(
            0,
            0,
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            GL_RGB,
            GL_FLOAT,
            self.img_array.ctypes.data_as(POINTER(GLfloat))
        )

        # Add noise to the image
        if self.img_noise_scale > 0:
            noise = self.np_random.normal(
                size=IMG_SHAPE,
                loc=0,
                scale=self.img_noise_scale
            )
            np.clip(self.img_array + noise, a_min=0, a_max=1, out=self.img_array)

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        return self.img_array

    def render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        # Render the observation
        img = self._render_obs()

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
        img_data = pyglet.image.ImageData(
            width,
            height,
            'RGB',
            img.ctypes.data_as(POINTER(GLubyte)),
            pitch=width * 3,
        )
        img_data.blit(
            0,
            0,
            0,
            width=WINDOW_WIDTH,
            height=WINDOW_HEIGHT
        )

        # Display position/state information
        pos = self.curPos
        self.text_label.text = "(%.2f, %.2f, %.2f)" % (pos[0], pos[1], pos[2])
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()
