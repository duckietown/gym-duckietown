import math
import time
import numpy as np
import yaml

import pyglet
from pyglet.gl import *
from ctypes import byref, POINTER

import gym
from gym import error, spaces, utils
from gym.utils import seeding

# Graphics utility code
from ..utils import *
from ..graphics import *
from ..objmesh import *
from ..collision import *

# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
CAMERA_WIDTH = 160
CAMERA_HEIGHT = 120

# Blue sky horizon color
BLUE_SKY_COLOR = np.array([0.45, 0.82, 1])

# Color meant to approximate interior walls
WALL_COLOR = np.array([0.64, 0.71, 0.28])

# Road color multiplier
ROAD_COLOR = np.array([0.79, 0.88, 0.53])

# Ground/floor color
GROUND_COLOR = np.array([0.15, 0.15, 0.15])

# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 20

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 42

# Distance from camera to floor (10.8cm)
CAMERA_FLOOR_DIST = 0.108

# Forward distance between the camera (at the front)
# and the center of rotation (6.6cm)
CAMERA_FORWARD_DIST = 0.066

# Distance (diameter) between the center of the robot wheels (10.2cm)
WHEEL_DIST = 0.102

# Total robot width at wheel base, used for collision detection
# Note: the actual robot width is 13cm, but we add a litte bit of buffer
#       to faciliate sim-to-real transfer.
ROBOT_WIDTH = 0.13 + 0.02

# Total robot length
# Note: the center of rotation (between the wheels) is not at the
#       geometric center see CAMERA_FORWARD_DIST
ROBOT_LENGTH = 0.18

# Height of the robot, used for scaling
ROBOT_HEIGHT = 0.12

# Road tile dimensions (2ft x 2ft, 61cm wide)
ROAD_TILE_SIZE = 0.61

# Maximum forward robot speed in meters/second
ROBOT_SPEED = 0.45

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25

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
        map_name='udem1',
        max_steps=600,
        draw_curve=False,
        draw_bbox=False,
        domain_rand=True
    ):
        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Flag to draw the road curve
        self.draw_curve = draw_curve

        # Flag to draw bounding boxes
        self.draw_bbox = draw_bbox

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand

        # Produce graphical output
        self.graphics = True

        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )

        # We observe an RGB image with pixels in [0, 255]
        # Note: the pixels are in uint8 format because this is more compact
        # than float32 if sent over the network or stored in a dataset
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(CAMERA_HEIGHT, CAMERA_WIDTH, 3),
            dtype=np.uint8
        )

        self.reward_range = (-1000, 1000)

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

        # Create a frame buffer object for the observation
        self.multi_fbo, self.final_fbo = create_frame_buffers(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            32
        )

        # Array to render the image into (for observation rendering)
        self.img_array = np.zeros(shape=self.observation_space.shape, dtype=np.uint8)

        # Create a frame buffer object for human rendering
        self.multi_fbo_human, self.final_fbo_human = create_frame_buffers(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            4
        )

        # Array to render the image into (for human rendering)
        self.img_array_human = np.zeros(shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3), dtype=np.uint8)

        # Create the vertex list for our road quad
        # Note: the vertices are centered around the origin so we can easily
        # rotate the tiles about their center
        half_size = ROAD_TILE_SIZE / 2
        verts = [
            -half_size, 0.0, -half_size,
             half_size, 0.0, -half_size,
             half_size, 0.0,  half_size,
            -half_size, 0.0,  half_size
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
            -1, -0.8,  1,
            -1, -0.8, -1,
             1, -0.8, -1,
             1, -0.8,  1
        ]
        self.ground_vlist = pyglet.graphics.vertex_list(4, ('v3f', verts))

        # Load the map
        self._load_map(map_name)

        # Initialize the state
        self.seed()
        self.reset()

    def reset(self):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0

        # Horizon color
        # Note: we explicitly sample white and grey/black because
        # these colors are easily confused for road and lane markings
        if self.domain_rand:
            horz_mode = self.np_random.randint(0, 4)
            if horz_mode == 0:
                self.horizon_color = self._perturb(BLUE_SKY_COLOR)
            elif horz_mode == 1:
                self.horizon_color = self._perturb(WALL_COLOR)
            elif horz_mode == 2:
                self.horizon_color = self._perturb([0.15, 0.15, 0.15], 0.4)
            elif horz_mode == 3:
                self.horizon_color = self._perturb([0.9, 0.9, 0.9], 0.4)
        else:
            self.horizon_color = BLUE_SKY_COLOR

        # Setup some basic lighting with a far away sun
        if self.domain_rand:
            lightPos = [
                self.np_random.uniform(-150, 150),
                self.np_random.uniform( 170, 220),
                self.np_random.uniform(-150, 150),
            ]
        else:
            lightPos = [-40, 200, 100]
        ambient = self._perturb([0.65, 0.65, 0.65])
        diffuse = self._perturb([0.50, 0.50, 0.50])
        gl.glLightfv(GL_LIGHT0, GL_POSITION, (GLfloat*4)(*lightPos))
        gl.glLightfv(GL_LIGHT0, GL_AMBIENT, (GLfloat*4)(*ambient))
        gl.glLightfv(GL_LIGHT0, GL_DIFFUSE, (GLfloat*4)(0.5, 0.5, 0.5, 1.0))
        gl.glEnable(GL_LIGHT0)
        gl.glEnable(GL_LIGHTING)
        gl.glEnable(GL_COLOR_MATERIAL)

        # Ground color
        self.ground_color = self._perturb(GROUND_COLOR, 0.3)

        # Distance between the robot's wheels
        self.wheel_dist = self._perturb(WHEEL_DIST)

        # Distance bewteen camera and ground
        self.cam_height = self._perturb(CAMERA_FLOOR_DIST, 0.08)

        # Angle at which the camera is pitched downwards
        self.cam_angle = self._perturb(CAMERA_ANGLE, 0.2)

        # Field of view angle of the camera
        self.cam_fov_y = self._perturb(CAMERA_FOV_Y, 0.2)

        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = 12
        verts = []
        colors = []
        for i in range(0, 3 * numTris):
            p = self.np_random.uniform(low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,))
            c = self.np_random.uniform(low=0, high=0.9)
            c = self._perturb([c, c, c], 0.1)
            verts += [p[0], p[1], p[2]]
            colors += [c[0], c[1], c[2]]
        self.tri_vlist = pyglet.graphics.vertex_list(3 * numTris, ('v3f', verts), ('c3f', colors) )

        # Randomize tile parameters
        for tile in self.grid:
            # Randomize the tile texture
            tile['texture'] = Texture.get(
                tile['kind'],
                rng = self.np_random if self.domain_rand else None
            )

            # Random tile color multiplier
            tile['color'] = self._perturb(ROAD_COLOR, 0.2)

        # Randomize object parameters
        for obj in self.objects:
            # Randomize the object color
            obj['color'] = self._perturb([1, 1, 1], 0.3)

            # Randomize whether the object is visible or not
            if obj['optional'] and self.domain_rand:
                obj['visible'] = self.np_random.randint(0, 2) == 0
            else:
                obj['visible'] = True

        # If the map specifies a starting tile
        if self.start_tile is not None:
            tile = self.start_tile
        else:
            # Select a random drivable tile to start on
            tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
            tile = self.drivable_tiles[tile_idx]

        # Keep trying to find a valid spawn position on this tile
        while True:
            i, j = tile['coords']

            # Choose a random position on this tile
            x = self.np_random.uniform(i, i + 1) * ROAD_TILE_SIZE
            z = self.np_random.uniform(j, j + 1) * ROAD_TILE_SIZE
            self.cur_pos = np.array([x, 0, z])

            # Choose a random direction
            self.cur_angle = self.np_random.uniform(0, 2 * math.pi)

            # If this is too close to an object or not a valid pose, retry
            if self._inconvenient_spawn() or not self._valid_pose():
                continue

            # If the angle is too far away from the driving direction, retry
            dist, dot_dir, angle = self.get_lane_pos()
            if angle < -70 or angle > 70:
                continue

            # Found a valid initial pose
            break

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs

    def _load_map(self, map_name):
        """
        Load the map layout from a CSV file
        """

        file_path = get_file_path('maps', map_name, 'yaml')

        print('loading map file "%s"' % file_path)

        with open(file_path, 'r') as f:
            map_data = yaml.load(f)

        tiles = map_data['tiles']
        assert len(tiles) > 0
        assert len(tiles[0]) > 0

        # Create the grid
        self.grid_height = len(tiles)
        self.grid_width = len(tiles[0])
        self.grid = [None] * self.grid_width * self.grid_height

        # We keep a separate list of drivable tiles
        self.drivable_tiles = []

        # For each row in the grid
        for j, row in enumerate(tiles):
            assert len(row) == self.grid_width, "each row of tiles must have the same length"

            # For each tile in this row
            for i, tile in enumerate(row):
                tile = tile.strip()

                if tile == 'empty':
                    continue

                if '/' in tile:
                    kind, orient = tile.split('/')
                    kind = kind.strip(' ')
                    orient = orient.strip(' ')
                    angle = ['S', 'E', 'N', 'W'].index(orient)
                    drivable = True
                else:
                    kind = tile
                    angle = 0
                    drivable = False

                tile = {
                    'coords': (i, j),
                    'kind': kind,
                    'angle': angle,
                    'drivable': drivable
                }

                if drivable:
                    self.drivable_tiles.append(tile)

                self._set_tile(i, j, tile)

        # Create the objects array
        self.objects = []

        # Arrays for checking collisions with N static objects

        # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
        self.static_corners = []

        # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
        self.static_norms = []

        # For each object
        for desc in map_data.get('objects', []):
            kind = desc['kind']
            x, z = desc['pos']
            rotate = desc['rotate']
            optional = desc.get('optional', False)

            pos = ROAD_TILE_SIZE * np.array((x, 0, z))

            # Load the mesh
            mesh = ObjMesh.get(kind)

            if 'height' in desc:
                scale = desc['height'] / mesh.max_coords[1]
            else:
                scale = desc['scale']
            assert not ('height' in desc and 'scale' in desc), "cannot specify both height and scale"

            obj = {
                'kind': kind,
                'mesh': mesh,
                'pos': pos,
                'scale': scale,
                'y_rot': rotate,
                'optional': optional,
                'min_coords': mesh.min_coords,
                'max_coords': mesh.max_coords,
                'static': True # TODO: load this from yml
            }

            self.objects.append(obj)

            # Compute collision detection information
            if obj['static']:
                angle = rotate * (math.pi / 180)
                corners = generate_corners(pos, mesh.min_coords, mesh.max_coords, angle, scale)
                self.static_corners.append(corners.T)
                self.static_norms.append(generate_norm(corners))

        # If there are static objects
        if len(self.static_corners) > 0:
            self.static_corners = np.stack(self.static_corners, axis=0)
            self.static_norms = np.stack(self.static_norms, axis=0)

        # Get the starting tile from the map, if specified
        self.start_tile = None
        if 'start_tile' in map_data:
            coords = map_data['start_tile']
            self.start_tile = self._get_tile(*coords)

    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, _ = seeding.np_random(seed)
        return [seed]

    def _set_tile(self, i, j, tile):
        assert i >= 0 and i < self.grid_width
        assert j >= 0 and j < self.grid_height
        self.grid[j * self.grid_width + i] = tile

    def _get_tile(self, i, j):
        if i < 0 or i >= self.grid_width:
            return None
        if j < 0 or j >= self.grid_height:
            return None
        return self.grid[j * self.grid_width + i]

    def _perturb(self, val, scale=0.1):
        """
        Add noise to a value. This is used for domain randomization.
        """
        assert scale >= 0
        assert scale < 1

        if isinstance(val, list):
            val = np.array(val)

        if not self.domain_rand:
            return val

        if isinstance(val, np.ndarray):
            noise = self.np_random.uniform(low=1-scale, high=1+scale, size=val.shape)
        else:
            noise = self.np_random.uniform(low=1-scale, high=1+scale)

        return val * noise

    def _get_grid_coords(self, abs_pos):
        """
        Compute the tile indices (i,j) for a given (x,_,z) world position

        x-axis maps to increasing i indices
        z-axis maps to increasing j indices

        Note: may return coordinates outside of the grid if the
        position entered is outside of the grid.
        """

        x, _, z = abs_pos
        i = math.floor(x / ROAD_TILE_SIZE)
        j = math.floor(z / ROAD_TILE_SIZE)

        return i, j

    def _get_curve(self, i, j):
        """
        Get the Bezier curve control points for a given tile
        """

        tile = self._get_tile(i, j)
        assert tile is not None

        kind = tile['kind']
        angle = tile['angle']

        if kind.startswith('straight') or kind.startswith('3way') or kind.startswith('4way'):
            pts = np.array([
                [-0.20, 0,-0.50],
                [-0.20, 0,-0.25],
                [-0.20, 0, 0.25],
                [-0.20, 0, 0.50],
            ]) * ROAD_TILE_SIZE
        elif kind == 'curve_left':
            pts = np.array([
                [-0.20, 0,-0.50],
                [-0.20, 0, 0.00],
                [ 0.00, 0, 0.20],
                [ 0.50, 0, 0.20],
            ]) * ROAD_TILE_SIZE
        elif kind == 'curve_right':
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
        pts += np.array([(i+0.5) * ROAD_TILE_SIZE, 0, (j+0.5) * ROAD_TILE_SIZE])

        return pts

    def get_dir_vec(self):
        """
        Vector pointing in the direction the agent is looking
        """

        x = math.cos(self.cur_angle)
        z = -math.sin(self.cur_angle)
        return np.array([x, 0, z])

    def get_right_vec(self):
        """
        Vector pointing to the right of the agent
        """

        x = math.sin(self.cur_angle)
        z = math.cos(self.cur_angle)
        return np.array([x, 0, z])

    def get_lane_pos(self):
        """
        Get the position of the agent relative to the center of the right lane
        """

        i, j = self._get_grid_coords(self.cur_pos)

        # Get the closest point along the right lane's Bezier curve
        cps = self._get_curve(i, j)
        t = bezier_closest(cps, self.cur_pos)
        point = bezier_point(cps, t)

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = self.get_dir_vec()
        tangent = bezier_tangent(cps, t)
        dotDir = np.dot(dirVec, tangent)
        dotDir = max(-1, min(1, dotDir))

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        posVec = self.cur_pos - point
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

    def _update_pos(self, wheelVels, deltaTime):
        """
        Update the position of the robot, simulating differential drive
        """

        Vl, Vr = wheelVels
        l = self.wheel_dist

        # If the wheel velocities are the same, then there is no rotation
        if Vl == Vr:
            self.cur_pos += deltaTime * Vl * self.get_dir_vec()
            return

        # Compute the angular rotation velocity about the ICC (center of curvature)
        w = (Vr - Vl) / l

        # Compute the distance to the center of curvature
        r = (l * (Vl + Vr)) / (2 * (Vl - Vr))

        # Compute the rotation angle for this time step
        rotAngle = w * deltaTime

        # Rotate the robot's position around the center of rotation
        r_vec = self.get_right_vec()
        px, py, pz = self.cur_pos
        cx = px + r * r_vec[0]
        cz = pz + r * r_vec[2]
        npx, npz = rotate_point(px, pz, cx, cz, rotAngle)
        self.cur_pos = np.array([npx, py, npz])

        # Update the robot's direction angle
        self.cur_angle += rotAngle

    def _drivable_pos(self, pos):
        """
        Check that the current (x,y,z) position is on a drivable tile
        """

        coords = self._get_grid_coords(pos)
        tile = self._get_tile(*coords)
        return tile != None and tile['drivable']

    def _actual_center(self, pos):
        """
        Calculate the position of the geometric center of the duckiebot
        The value of self.cur_pos is the center of rotation.
        """

        dir_vec = self.get_dir_vec()
        return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH/2)) * dir_vec

    def _inconvenient_spawn(self):
        """
        Check that duckie spawn is not too close to any object
        """

        results = [np.linalg.norm(x['pos'] - self.cur_pos) <
            max(x['max_coords']) * 0.5 * x['scale'] + MIN_SPAWN_OBJ_DIST
            for x in self.objects if x['visible']
        ]
        return np.any(results)

    def _collision(self):
        """
        Tensor-based OBB Collision detection
        """

        # If there are no objects to collide against, stop
        if len(self.static_corners) == 0:
            return False

        # Recompute the bounding boxes (BB) for the duckiebot
        self.duckie_corners = duckie_boundbox(
            self._actual_center(self.cur_pos),
            ROBOT_WIDTH,
            ROBOT_LENGTH,
            self.get_dir_vec(),
            self.get_right_vec()
        )

        # Generate the norms corresponding to each face of BB
        self.duckie_norm = generate_norm(self.duckie_corners)

        return intersects(
            self.duckie_corners,
            self.static_corners,
            self.duckie_norm,
            self.static_norms
        )

    def _valid_pose(self):
        """
        Check that the agent is in a valid pose
        """

        # Compute the coordinates of the base of both wheels
        f_vec = self.get_dir_vec()
        r_vec = self.get_right_vec()
        l_pos = self.cur_pos - 0.5 * ROBOT_WIDTH * r_vec
        r_pos = self.cur_pos + 0.5 * ROBOT_WIDTH * r_vec
        f_pos = self.cur_pos + 0.5 * ROBOT_WIDTH * f_vec

        collision = self._collision()

        # Check that the center position and
        # both wheels are on drivable tiles and no collisions
        return (
            self._drivable_pos(self.cur_pos) and
            self._drivable_pos(l_pos) and
            self._drivable_pos(r_pos) and
            self._drivable_pos(f_pos) and
            not collision
        )

    def step(self, action):
        self.step_count += 1

        # Update the robot's position
        self._update_pos(action * ROBOT_SPEED * 1, 0.1)

        # Generate the current camera image
        obs = self.render_obs()

        # If the agent is not in a valid pose (on drivable tiles)
        if not self._valid_pose():
            reward = -1000
            done = True
            return obs, reward, done, {}

        # If the maximum time step count is reached
        if self.step_count >= self.max_steps:
            done = True
            reward = 0
            return obs, reward, done, {}

        # Get the position relative to the right lane tangent
        dist, dotDir, angle = self.get_lane_pos()
        reward = 1.0 * dotDir - 10.00 * abs(dist)
        done = False

        return obs, reward, done, {}

    def render_obs(self):
        """
        Render an observation from the point of view of the agent
        """

        return self._render_img(
            CAMERA_WIDTH,
            CAMERA_HEIGHT,
            self.multi_fbo,
            self.final_fbo,
            self.img_array
        )

    def _render_img(self, width, height, multi_fbo, final_fbo, img_array):
        if self.graphics == False:
            return

        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        #pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        # Bind the multisampled frame buffer
        glEnable(GL_MULTISAMPLE)
        glBindFramebuffer(GL_FRAMEBUFFER, multi_fbo);
        glViewport(0, 0, width, height)

        # Clear the color and depth buffers
        glClearColor(*self.horizon_color, 1.0)
        glClearDepth(1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        # Set the projection matrix
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(
            self.cam_fov_y,
            width / float(height),
            0.04,
            100.0
        )

        # Set modelview matrix
        # Note: we add a bit of noise to the camera position for data augmentation
        pos = self.cur_pos
        if self.domain_rand:
            pos = pos + self.np_random.uniform(low=-0.005, high=0.005, size=(3,))
        x, y, z = pos
        dx, dy, dz = self.get_dir_vec()
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        if self.draw_bbox:
            y += 0.8
            glRotatef(90, 1, 0, 0)
        else:
            y += CAMERA_FLOOR_DIST
            glRotatef(self.cam_angle, 1, 0, 0)
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
        glColor3f(*self.ground_color)
        glPushMatrix()
        glScalef(50, 1, 50)
        self.ground_vlist.draw(GL_QUADS)
        glPopMatrix()

        # Draw the ground/noise triangles
        self.tri_vlist.draw(GL_TRIANGLES)

        # Draw the road quads
        glEnable(GL_TEXTURE_2D)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        # For each grid tile
        for j in range(self.grid_height):
            for i in range(self.grid_width):
                # Get the tile type and angle
                tile = self._get_tile(i, j)

                if tile == None:
                    continue

                kind = tile['kind']
                angle = tile['angle']
                color = tile['color']
                texture = tile['texture']

                glColor3f(*color)

                glPushMatrix()
                glTranslatef((i+0.5) * ROAD_TILE_SIZE, 0, (j+0.5) * ROAD_TILE_SIZE)
                glRotatef(angle * 90, 0, 1, 0)

                # Bind the appropriate texture
                texture.bind()

                self.road_vlist.draw(GL_QUADS)
                glPopMatrix()

                if self.draw_curve and tile['drivable']:
                    pts = self._get_curve(i, j)
                    bezier_draw(pts, n = 20)

        # For each object
        for idx, obj in enumerate(self.objects):
            if not obj['visible']:
                continue

            # Draw the bounding box
            if self.draw_bbox:
                corners = self.static_corners[idx]
                glColor3f(1, 0, 0)
                glBegin(GL_LINE_LOOP)
                glVertex3f(corners[0, 0], 0.01, corners[1, 0])
                glVertex3f(corners[0, 1], 0.01, corners[1, 1])
                glVertex3f(corners[0, 2], 0.01, corners[1, 2])
                glVertex3f(corners[0, 3], 0.01, corners[1, 3])
                glEnd()

            scale = obj['scale']
            y_rot = obj['y_rot']
            mesh = obj['mesh']
            glPushMatrix()
            glTranslatef(*obj['pos'])
            glScalef(scale, scale, scale)
            glRotatef(y_rot, 0, 1, 0)
            glColor3f(*obj['color'])
            mesh.render()
            glPopMatrix()

        # Draw the agent's own bounding box
        if self.draw_bbox:
            corners = self.duckie_corners
            glColor3f(1, 0, 0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(corners[0, 0], 0.01, corners[0, 1])
            glVertex3f(corners[1, 0], 0.01, corners[1, 1])
            glVertex3f(corners[2, 0], 0.01, corners[2, 1])
            glVertex3f(corners[3, 0], 0.01, corners[3, 1])
            glEnd()

        # Resolve the multisampled frame buffer into the final frame buffer
        glBindFramebuffer(GL_READ_FRAMEBUFFER, multi_fbo);
        glBindFramebuffer(GL_DRAW_FRAMEBUFFER, final_fbo);
        glBlitFramebuffer(
            0, 0,
            width, height,
            0, 0,
            width, height,
            GL_COLOR_BUFFER_BIT,
            GL_LINEAR
        );

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        glBindFramebuffer(GL_FRAMEBUFFER, final_fbo);
        glReadPixels(
            0,
            0,
            width,
            height,
            GL_RGB,
            GL_UNSIGNED_BYTE,
            img_array.ctypes.data_as(POINTER(GLubyte))
        )

        # Unbind the frame buffer
        glBindFramebuffer(GL_FRAMEBUFFER, 0);

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

        return img_array

    def render(self, mode='human', close=False):
        if close:
            if self.window:
                self.window.close()
            return

        # Render the image
        img = self._render_img(
            WINDOW_WIDTH,
            WINDOW_HEIGHT,
            self.multi_fbo_human,
            self.final_fbo_human,
            self.img_array_human
        )

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

        # Bind the default frame buffer
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
        img = np.ascontiguousarray(np.flip(img, axis=0))
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
        x, y, z = self.cur_pos
        self.text_label.text = "pos: (%.2f, %.2f, %.2f), angle: %d, steps: %d" % (
            x, y, z,
            int(self.cur_angle * 180 / math.pi),
            self.step_count
        )
        self.text_label.draw()

        # Force execution of queued commands
        glFlush()
