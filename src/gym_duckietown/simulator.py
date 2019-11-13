# coding=utf-8
from __future__ import division

from collections import namedtuple
from ctypes import POINTER
from dataclasses import dataclass
from typing import Tuple
import geometry

from duckietown_world.world_duckietown.pwm_dynamics import get_DB18_nominal, get_DB18_uncalibrated

@dataclass
class DoneRewardInfo:
    done: bool
    done_why: str
    done_code: str
    reward: float


@dataclass
class DynamicsInfo:
    motor_left: float
    motor_right: float

import gym
import yaml
from gym import spaces
from gym.utils import seeding

from .collision import *
# Objects utility code
from .objects import WorldObj, DuckieObj, TrafficLightObj, DuckiebotObj
# Graphics utility code
from .objmesh import *
# Randomization code
from .randomization import Randomizer

# Rendering window size
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600

# Camera image size
DEFAULT_CAMERA_WIDTH = 640
DEFAULT_CAMERA_HEIGHT = 480

# Blue sky horizon color
BLUE_SKY_COLOR = np.array([0.45, 0.82, 1])

# Color meant to approximate interior walls
WALL_COLOR = np.array([0.64, 0.71, 0.28])

# Ground/floor color
GROUND_COLOR = np.array([0.15, 0.15, 0.15])

# Angle at which the camera is pitched downwards
CAMERA_ANGLE = 19.15

# Camera field of view angle in the Y direction
# Note: robot uses Raspberri Pi camera module V1.3
# https://www.raspberrypi.org/documentation/hardware/camera/README.md
CAMERA_FOV_Y = 75

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

# Safety radius multiplier
SAFETY_RAD_MULT = 1.8

# Robot safety circle radius
AGENT_SAFETY_RAD = (max(ROBOT_LENGTH, ROBOT_WIDTH) / 2) * SAFETY_RAD_MULT

# Minimum distance spawn position needs to be from all objects
MIN_SPAWN_OBJ_DIST = 0.25

# Road tile dimensions (2ft x 2ft, 61cm wide)
# self.road_tile_size = 0.61

# Maximum forward robot speed in meters/second
DEFAULT_ROBOT_SPEED = 1.20
# approx 2 tiles/second

DEFAULT_FRAMERATE = 30

DEFAULT_MAX_STEPS = 1500

DEFAULT_MAP_NAME = 'udem1'

DEFAULT_FRAME_SKIP = 1

DEFAULT_ACCEPT_START_ANGLE_DEG = 60

REWARD_INVALID_POSE = -1000

MAX_SPAWN_ATTEMPTS = 5000

LanePosition0 = namedtuple('LanePosition', 'dist dot_dir angle_deg angle_rad')


class LanePosition(LanePosition0):

    def as_json_dict(self):
        """ Serialization-friendly format. """
        return dict(dist=self.dist,
                    dot_dir=self.dot_dir,
                    angle_deg=self.angle_deg,
                    angle_rad=self.angle_rad)


class NotInLane(Exception):
    ''' Raised when the Duckiebot is not in a lane. '''
    pass


class Simulator(gym.Env):
    """
    Simple road simulator to test RL training.
    Draws a road with turns using OpenGL, and simulates
    basic differential-drive dynamics.
    """

    metadata = {
        'render.modes': ['human', 'rgb_array', 'app'],
        'video.frames_per_second': 30
    }

    cur_pos: np.ndarray
    cam_offset: np.ndarray
    road_tile_size: float

    def __init__(
            self,
            map_name=DEFAULT_MAP_NAME,
            max_steps=DEFAULT_MAX_STEPS,
            draw_curve=False,
            draw_bbox=False,
            domain_rand=True,
            frame_rate=DEFAULT_FRAMERATE,
            frame_skip=DEFAULT_FRAME_SKIP,
            camera_width=DEFAULT_CAMERA_WIDTH,
            camera_height=DEFAULT_CAMERA_HEIGHT,
            robot_speed=DEFAULT_ROBOT_SPEED,
            accept_start_angle_deg=DEFAULT_ACCEPT_START_ANGLE_DEG,
            full_transparency=False,
            user_tile_start=None,
            seed=None,
            distortion=False,
            dynamics_rand=False,
            camera_rand=False,
            randomize_maps_on_reset=False,
    ):
        """

        :param map_name:
        :param max_steps:
        :param draw_curve:
        :param draw_bbox:
        :param domain_rand: If true, applies domain randomization
        :param frame_rate:
        :param frame_skip:
        :param camera_width:
        :param camera_height:
        :param robot_speed:
        :param accept_start_angle_deg:
        :param full_transparency:
        :param user_tile_start: If None, sample randomly. Otherwise (i,j). Overrides map start tile
        :param seed:
        :param distortion: If true, distorts the image with fish-eye approximation
        :param dynamics_rand: If true, perturbs the trim of the Duckiebot
        :param camera_rand: If true randomizes over camera miscalibration
        :param randomize_maps_on_reset: If true, randomizes the map on reset (Slows down training)
        """
        # first initialize the RNG
        self.seed_value = seed
        self.seed(seed=self.seed_value)

        # If true, then we publish all transparency information
        self.full_transparency = full_transparency

        # Map name, set in _load_map()
        self.map_name = None

        # Full map file path, set in _load_map()
        self.map_file_path = None

        # The parsed content of the map_file
        self.map_data = None

        # Maximum number of steps per episode
        self.max_steps = max_steps

        # Flag to draw the road curve
        self.draw_curve = draw_curve

        # Flag to draw bounding boxes
        self.draw_bbox = draw_bbox

        # Flag to enable/disable domain randomization
        self.domain_rand = domain_rand
        if self.domain_rand:
            self.randomizer = Randomizer()

        # Frame rate to run at
        self.frame_rate = frame_rate
        self.delta_time = 1.0 / self.frame_rate

        # Number of frames to skip per action
        self.frame_skip = frame_skip

        # Produce graphical output
        self.graphics = True

        # Two-tuple of wheel torques, each in the range [-1, 1]
        self.action_space = spaces.Box(
                low=-1,
                high=1,
                shape=(2,),
                dtype=np.float32
        )

        self.camera_width = camera_width
        self.camera_height = camera_height

        self.robot_speed = robot_speed
        # We observe an RGB image with pixels in [0, 255]
        # Note: the pixels are in uint8 format because this is more compact
        # than float32 if sent over the network or stored in a dataset
        self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.camera_height, self.camera_width, 3),
                dtype=np.uint8
        )

        self.reward_range = (-1000, 1000)

        # Window for displaying the environment to humans
        self.window = None

        import pyglet
        # Invisible window to render into (shadow OpenGL context)
        self.shadow_window = pyglet.window.Window(width=1,
                                                  height=1,
                                                  visible=False)

        # For displaying text
        self.text_label = pyglet.text.Label(
                font_name="Arial",
                font_size=14,
                x=5,
                y=WINDOW_HEIGHT - 19
        )

        # Create a frame buffer object for the observation
        self.multi_fbo, self.final_fbo = create_frame_buffers(
                self.camera_width,
                self.camera_height,
                4
        )

        # Array to render the image into (for observation rendering)
        self.img_array = np.zeros(shape=self.observation_space.shape,
                                  dtype=np.uint8)

        # Create a frame buffer object for human rendering
        self.multi_fbo_human, self.final_fbo_human = create_frame_buffers(
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                4
        )

        # Array to render the image into (for human rendering)
        self.img_array_human = np.zeros(shape=(WINDOW_HEIGHT, WINDOW_WIDTH, 3),
                                        dtype=np.uint8)

        # allowed angle in lane for starting position
        self.accept_start_angle_deg = accept_start_angle_deg

        # Load the map
        self._load_map(map_name)

        # Distortion params, if so, load the library, only if not bbox mode
        self.distortion = distortion and not draw_bbox
        self.camera_rand = False
        if not draw_bbox and distortion:
            if distortion:
                self.camera_rand = camera_rand
                from .distortion import Distortion
                self.camera_model = Distortion(camera_rand=self.camera_rand)

        # Used by the UndistortWrapper, always initialized to False
        self.undistort = False

        # Dynamics randomization
        self.dynamics_rand = dynamics_rand

        # Start tile
        self.user_tile_start = user_tile_start

        self.randomize_maps_on_reset = randomize_maps_on_reset

        if self.randomize_maps_on_reset:
            import os
            self.map_names = os.listdir('maps')
            self.map_names = [mapfile.replace('.yaml', '') for mapfile in self.map_names]

        # Initialize the state
        self.reset()

        self.last_action = np.array([0, 0])
        self.wheelVels = np.array([0, 0])

    def _init_vlists(self):
        import pyglet
        # Create the vertex list for our road quad
        # Note: the vertices are centered around the origin so we can easily
        # rotate the tiles about their center
        half_size = self.road_tile_size / 2
        verts = [
            -half_size, 0.0, -half_size,
            half_size, 0.0, -half_size,
            half_size, 0.0, half_size,
            -half_size, 0.0, half_size
        ]
        texCoords = [
            1.0, 0.0,
            0.0, 0.0,
            0.0, 1.0,
            1.0, 1.0
        ]
        self.road_vlist = pyglet.graphics.vertex_list(4, ('v3f', verts),
                                                      ('t2f', texCoords))

        # Create the vertex list for the ground quad
        verts = [
            -1, -0.8, 1,
            -1, -0.8, -1,
            1, -0.8, -1,
            1, -0.8, 1
        ]
        self.ground_vlist = pyglet.graphics.vertex_list(4, ('v3f', verts))

    def reset(self):
        """
        Reset the simulation at the start of a new episode
        This also randomizes many environment parameters (domain randomization)
        """

        # Step count since episode start
        self.step_count = 0
        self.timestamp = 0.0

        # Robot's current speed
        self.speed = 0

        if self.randomize_maps_on_reset:
            map_name = np.random.choice(self.map_names)
            self._load_map(map_name)

        if self.domain_rand:
            self.randomization_settings = self.randomizer.randomize()

        # Horizon color
        # Note: we explicitly sample white and grey/black because
        # these colors are easily confused for road and lane markings
        if self.domain_rand:
            horz_mode = self.randomization_settings['horz_mode']
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
            light_pos = self.randomization_settings['light_pos']
        else:
            light_pos = [-40, 200, 100]

        ambient = self._perturb([0.50, 0.50, 0.50], 0.3)
        # XXX: diffuse is not used?
        diffuse = self._perturb([0.70, 0.70, 0.70], 0.3)
        from pyglet import gl
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_POSITION, (gl.GLfloat * 4)(*light_pos))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_AMBIENT, (gl.GLfloat * 4)(*ambient))
        gl.glLightfv(gl.GL_LIGHT0, gl.GL_DIFFUSE, (gl.GLfloat * 4)(0.5, 0.5, 0.5, 1.0))
        gl.glEnable(gl.GL_LIGHT0)
        gl.glEnable(gl.GL_LIGHTING)
        gl.glEnable(gl.GL_COLOR_MATERIAL)

        # Ground color
        self.ground_color = self._perturb(GROUND_COLOR, 0.3)

        # Distance between the robot's wheels
        self.wheel_dist = self._perturb(WHEEL_DIST)

        # Set default values

        # Distance bewteen camera and ground
        self.cam_height = CAMERA_FLOOR_DIST

        # Angle at which the camera is rotated
        self.cam_angle = [CAMERA_ANGLE, 0, 0]

        # Field of view angle of the camera
        self.cam_fov_y = CAMERA_FOV_Y

        # Perturb using randomization API (either if domain rand or only camera rand
        if self.domain_rand or self.camera_rand:
            self.cam_height *= self.randomization_settings['camera_height']
            self.cam_angle = [CAMERA_ANGLE * self.randomization_settings['camera_angle'], 0, 0]
            self.cam_fov_y *= self.randomization_settings['camera_fov_y']

        # Camera offset for use in free camera mode
        self.cam_offset = np.array([0, 0, 0])

        # Create the vertex list for the ground/noise triangles
        # These are distractors, junk on the floor
        numTris = 12
        verts = []
        colors = []
        for _ in range(0, 3 * numTris):
            p = self.np_random.uniform(low=[-20, -0.6, -20], high=[20, -0.3, 20], size=(3,))
            c = self.np_random.uniform(low=0, high=0.9)
            c = self._perturb([c, c, c], 0.1)
            verts += [p[0], p[1], p[2]]
            colors += [c[0], c[1], c[2]]
        import pyglet
        self.tri_vlist = pyglet.graphics.vertex_list(3 * numTris, ('v3f', verts), ('c3f', colors))

        # Randomize tile parameters
        for tile in self.grid:
            rng = self.np_random if self.domain_rand else None
            # Randomize the tile texture
            tile['texture'] = Texture.get(tile['kind'], rng=rng)

            # Random tile color multiplier
            tile['color'] = self._perturb([1, 1, 1], 0.2)

        # Randomize object parameters
        for obj in self.objects:
            # Randomize the object color
            obj.color = self._perturb([1, 1, 1], 0.3)

            # Randomize whether the object is visible or not
            if obj.optional and self.domain_rand:
                obj.visible = self.np_random.randint(0, 2) == 0
            else:
                obj.visible = True

        # If the map specifies a starting tile
        if self.user_tile_start:
            logger.info('using user tile start: %s' % self.user_tile_start)
            i, j = self.user_tile_start
            tile = self._get_tile(i, j)
            if tile is None:
                msg = 'The tile specified does not exist.'
                raise Exception(msg)
            logger.debug('tile: %s' % tile)
        else:
            if self.start_tile is not None:
                tile = self.start_tile
            else:
                # Select a random drivable tile to start on
                tile_idx = self.np_random.randint(0, len(self.drivable_tiles))
                tile = self.drivable_tiles[tile_idx]

        # If the map specifies a starting pose
        if self.start_pose is not None:
            logger.info('using map pose start: %s' % self.start_pose)
            propose_pos, propose_angle = self.start_pose
        else:
            # Keep trying to find a valid spawn position on this tile
            for _ in range(MAX_SPAWN_ATTEMPTS):
                i, j = tile['coords']

                # Choose a random position on this tile
                x = self.np_random.uniform(i, i + 1) * self.road_tile_size
                z = self.np_random.uniform(j, j + 1) * self.road_tile_size
                propose_pos = np.array([x, 0, z])

                # Choose a random direction
                propose_angle = self.np_random.uniform(0, 2 * math.pi)

                # logger.debug('Sampled %s %s angle %s' % (propose_pos[0],
                #                                          propose_pos[1],
                #                                          np.rad2deg(propose_angle)))

                # If this is too close to an object or not a valid pose, retry
                inconvenient = self._inconvenient_spawn(propose_pos)

                if inconvenient:
                    # msg = 'The spawn was inconvenient.'
                    # logger.warning(msg)
                    continue

                invalid = not self._valid_pose(propose_pos, propose_angle, safety_factor=1.3)
                if invalid:
                    # msg = 'The spawn was invalid.'
                    # logger.warning(msg)
                    continue

                # If the angle is too far away from the driving direction, retry
                try:
                    lp = self.get_lane_pos2(propose_pos, propose_angle)
                except NotInLane:
                    continue
                M = self.accept_start_angle_deg
                ok = -M < lp.angle_deg < +M
                if not ok:
                    continue
                # Found a valid initial pose
                break
            else:
                msg = 'Could not find a valid starting pose after %s attempts' % MAX_SPAWN_ATTEMPTS
                raise Exception(msg)

        self.cur_pos = propose_pos
        self.cur_angle = propose_angle

        init_vel = np.array([0, 0])

        # Initialize Dynamics model
        if self.dynamics_rand:
            trim = 0 + self.randomization_settings['trim']
            p = get_DB18_uncalibrated(delay=0.15, trim=trim)
        else:
            p = get_DB18_nominal(delay=0.15)

        q = self.cartesian_from_weird(self.cur_pos, self.cur_angle)
        v0 = geometry.se2_from_linear_angular(init_vel, 0)
        c0 = q, v0
        self.state = p.initialize(c0=c0, t0=0)

        logger.info('Starting at %s %s' % (self.cur_pos, self.cur_angle))

        # Generate the first camera image
        obs = self.render_obs()

        # Return first observation
        return obs

    def _load_map(self, map_name):
        """
        Load the map layout from a YAML file
        """

        # Store the map name
        self.map_name = map_name

        # Get the full map file path
        self.map_file_path = get_file_path('maps', map_name, 'yaml')

        logger.debug('loading map file "%s"' % self.map_file_path)

        with open(self.map_file_path, 'r') as f:
            self.map_data = yaml.load(f, Loader=yaml.Loader)

        self._interpret_map(self.map_data)

    def _interpret_map(self, map_data: dict):
        if not 'tile_size' in map_data:
            msg = 'Must now include explicit tile_size in the map data.'
            raise ValueError(msg)
        self.road_tile_size = map_data['tile_size']
        self._init_vlists()

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
            msg = "each row of tiles must have the same length"
            if len(row) != self.grid_width:
                raise Exception(msg)

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
                elif '4' in tile:
                    kind = '4way'
                    angle = 2
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

                self._set_tile(i, j, tile)

                if drivable:
                    tile['curves'] = self._get_curve(i, j)
                    self.drivable_tiles.append(tile)

        self.mesh = ObjMesh.get('duckiebot')
        self._load_objects(map_data)

        # Get the starting tile from the map, if specified
        self.start_tile = None
        if 'start_tile' in map_data:
            coords = map_data['start_tile']
            self.start_tile = self._get_tile(*coords)

        # Get the starting pose from the map, if specified
        self.start_pose = None
        if 'start_pose' in map_data:
            self.start_pose = map_data['start_pose']

    def _load_objects(self, map_data):
        # Create the objects array
        self.objects = []

        # The corners for every object, regardless if collidable or not
        self.object_corners = []

        # Arrays for checking collisions with N static objects
        # (Dynamic objects done separately)
        # (N x 2): Object position used in calculating reward
        self.collidable_centers = []

        # (N x 2 x 4): 4 corners - (x, z) - for object's boundbox
        self.collidable_corners = []

        # (N x 2 x 2): two 2D norms for each object (1 per axis of boundbox)
        self.collidable_norms = []

        # (N): Safety radius for object used in calculating reward
        self.collidable_safety_radii = []

        # For each object
        for obj_idx, desc in enumerate(map_data.get('objects', [])):
            kind = desc['kind']

            pos = desc['pos']
            x, z = pos[0:2]
            y = pos[2] if len(pos) == 3 else 0.0

            rotate = desc['rotate']
            optional = desc.get('optional', False)

            pos = self.road_tile_size * np.array((x, y, z))

            # Load the mesh
            mesh = ObjMesh.get(kind)

            if 'height' in desc:
                scale = desc['height'] / mesh.max_coords[1]
            else:
                scale = desc['scale']
            assert not ('height' in desc and 'scale' in desc), "cannot specify both height and scale"

            static = desc.get('static', True)

            obj_desc = {
                'kind': kind,
                'mesh': mesh,
                'pos': pos,
                'scale': scale,
                'y_rot': rotate,
                'optional': optional,
                'static': static,
            }

            # obj = None
            if static:
                if kind == "trafficlight":
                    obj = TrafficLightObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
                else:
                    obj = WorldObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT)
            else:
                if kind == "duckiebot":
                    obj = DuckiebotObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, WHEEL_DIST,
                                       ROBOT_WIDTH, ROBOT_LENGTH)
                elif kind == "duckie":
                    obj = DuckieObj(obj_desc, self.domain_rand, SAFETY_RAD_MULT, self.road_tile_size)
                else:
                    msg = 'I do not know what object this is: %s' % kind
                    raise Exception(msg)

            self.objects.append(obj)

            # Compute collision detection information

            # angle = rotate * (math.pi / 180)

            # Find drivable tiles object could intersect with
            possible_tiles = find_candidate_tiles(obj.obj_corners, self.road_tile_size)

            # If the object intersects with a drivable tile
            if static and kind != "trafficlight" and self._collidable_object(
                    obj.obj_corners, obj.obj_norm, possible_tiles
            ):
                self.collidable_centers.append(pos)
                self.collidable_corners.append(obj.obj_corners.T)
                self.collidable_norms.append(obj.obj_norm)
                self.collidable_safety_radii.append(obj.safety_radius)

        # If there are collidable objects
        if len(self.collidable_corners) > 0:
            self.collidable_corners = np.stack(self.collidable_corners, axis=0)
            self.collidable_norms = np.stack(self.collidable_norms, axis=0)

            # Stack doesn't do anything if there's only one object,
            # So we add an extra dimension to avoid shape errors later
            if len(self.collidable_corners.shape) == 2:
                self.collidable_corners = self.collidable_corners[np.newaxis]
                self.collidable_norms = self.collidable_norms[np.newaxis]

        self.collidable_centers = np.array(self.collidable_centers)
        self.collidable_safety_radii = np.array(self.collidable_safety_radii)

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
        """
            Returns None if the duckiebot is not in a tile.
        """
        i = int(i)
        j = int(j)
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
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale, size=val.shape)
        else:
            noise = self.np_random.uniform(low=1 - scale, high=1 + scale)

        return val * noise

    def _collidable_object(self, obj_corners, obj_norm, possible_tiles):
        """
        A function to check if an object intersects with any
        drivable tiles, which would mean our agent could run into them.
        Helps optimize collision checking with agent during runtime
        """

        if possible_tiles.shape == 0:
            return False

        drivable_tiles = []
        for c in possible_tiles:
            tile = self._get_tile(c[0], c[1])
            if tile and tile['drivable']:
                drivable_tiles.append((c[0], c[1]))

        if not drivable_tiles:
            return False

        drivable_tiles = np.array(drivable_tiles)

        # Tiles are axis aligned, so add normal vectors in bulk
        tile_norms = np.array([[1, 0], [0, 1]] * len(drivable_tiles))

        # None of the candidate tiles are drivable, don't add object
        if len(drivable_tiles) == 0:
            return False

        # Find the corners for each candidate tile
        drivable_tiles = np.array([
            tile_corners(
                    self._get_tile(pt[0], pt[1])['coords'],
                    self.road_tile_size
            ).T for pt in drivable_tiles
        ])

        # Stack doesn't do anything if there's only one object,
        # So we add an extra dimension to avoid shape errors later
        if len(tile_norms.shape) == 2:
            tile_norms = tile_norms[np.newaxis]
        else:  # Stack works as expected
            drivable_tiles = np.stack(drivable_tiles, axis=0)
            tile_norms = np.stack(tile_norms, axis=0)

        # Only add it if one of the vertices is on a drivable tile
        return intersects(obj_corners, drivable_tiles, obj_norm, tile_norms)

    def get_grid_coords(self, abs_pos):
        """
        Compute the tile indices (i,j) for a given (x,_,z) world position

        x-axis maps to increasing i indices
        z-axis maps to increasing j indices

        Note: may return coordinates outside of the grid if the
        position entered is outside of the grid.
        """

        x, _, z = abs_pos
        i = math.floor(x / self.road_tile_size)
        j = math.floor(z / self.road_tile_size)

        return int(i), int(j)

    def _get_curve(self, i, j):
        """
            Get the Bezier curve control points for a given tile
        """
        tile = self._get_tile(i, j)
        assert tile is not None

        kind = tile['kind']
        angle = tile['angle']

        # Each tile will have a unique set of control points,
        # Corresponding to each of its possible turns

        if kind.startswith('straight'):
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.25],
                    [-0.20, 0, 0.25],
                    [-0.20, 0, 0.50],
                ],
                [
                    [0.20, 0, 0.50],
                    [0.20, 0, 0.25],
                    [0.20, 0, -0.25],
                    [0.20, 0, -0.50],
                ]
            ]) * self.road_tile_size

        elif kind == 'curve_left':
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, 0.00],
                    [0.00, 0, 0.20],
                    [0.50, 0, 0.20],
                ],
                [
                    [0.50, 0, -0.20],
                    [0.30, 0, -0.20],
                    [0.20, 0, -0.30],
                    [0.20, 0, -0.50],
                ]
            ]) * self.road_tile_size

        elif kind == 'curve_right':
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.20],
                    [-0.30, 0, -0.20],
                    [-0.50, 0, -0.20],
                ],

                [
                    [-0.50, 0, 0.20],
                    [-0.30, 0, 0.20],
                    [0.30, 0, 0.00],
                    [0.20, 0, -0.50],
                ]
            ]) * self.road_tile_size

        # Hardcoded all curves for 3way intersection
        elif kind.startswith('3way'):
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.25],
                    [-0.20, 0, 0.25],
                    [-0.20, 0, 0.50],
                ],
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, 0.00],
                    [0.00, 0, 0.20],
                    [0.50, 0, 0.20],
                ],
                [
                    [0.20, 0, 0.50],
                    [0.20, 0, 0.25],
                    [0.20, 0, -0.25],
                    [0.20, 0, -0.50],
                ],
                [
                    [0.50, 0, -0.20],
                    [0.30, 0, -0.20],
                    [0.20, 0, -0.20],
                    [0.20, 0, -0.50],
                ],
                [
                    [0.20, 0, 0.50],
                    [0.20, 0, 0.20],
                    [0.30, 0, 0.20],
                    [0.50, 0, 0.20],
                ],
                [
                    [0.50, 0, -0.20],
                    [0.30, 0, -0.20],
                    [-0.20, 0, 0.00],
                    [-0.20, 0, 0.50],
                ],
            ]) * self.road_tile_size

        # Template for each side of 4way intersection
        elif kind.startswith('4way'):
            pts = np.array([
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, 0.00],
                    [0.00, 0, 0.20],
                    [0.50, 0, 0.20],
                ],
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.25],
                    [-0.20, 0, 0.25],
                    [-0.20, 0, 0.50],
                ],
                [
                    [-0.20, 0, -0.50],
                    [-0.20, 0, -0.20],
                    [-0.30, 0, -0.20],
                    [-0.50, 0, -0.20],
                ]
            ]) * self.road_tile_size
        else:
            assert False, kind

        # Rotate and align each curve with its place in global frame
        if kind.startswith('4way'):
            fourway_pts = []
            # Generate all four sides' curves,
            # with 3-points template above
            for rot in np.arange(0, 4):
                mat = gen_rot_matrix(np.array([0, 1, 0]), rot * math.pi / 2)
                pts_new = np.matmul(pts, mat)
                pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
                fourway_pts.append(pts_new)

            fourway_pts = np.reshape(np.array(fourway_pts), (12, 4, 3))
            return fourway_pts

        # Hardcoded each curve; just rotate and shift
        elif kind.startswith('3way'):
            threeway_pts = []
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts_new = np.matmul(pts, mat)
            pts_new += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])
            threeway_pts.append(pts_new)

            threeway_pts = np.array(threeway_pts)
            threeway_pts = np.reshape(threeway_pts, (6, 4, 3))
            return threeway_pts

        else:
            mat = gen_rot_matrix(np.array([0, 1, 0]), angle * math.pi / 2)
            pts = np.matmul(pts, mat)
            pts += np.array([(i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size])

        return pts

    def get_dir_vec(self, angle=None):
        """
        Vector pointing in the direction the agent is looking
        """
        if angle == None:
            angle = self.cur_angle

        x = math.cos(angle)
        z = -math.sin(angle)
        return np.array([x, 0, z])

    def get_right_vec(self, angle=None):
        """
        Vector pointing to the right of the agent
        """
        if angle == None:
            angle = self.cur_angle

        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])

    def closest_curve_point(self, pos, angle=None):
        """
            Get the closest point on the curve to a given point
            Also returns the tangent at that point.

            Returns None, None if not in a lane.
        """

        i, j = self.get_grid_coords(pos)
        tile = self._get_tile(i, j)

        if tile is None or not tile['drivable']:
            return None, None

        # Find curve with largest dotproduct with heading
        curves = self._get_tile(i, j)['curves']
        curve_headings = curves[:, -1, :] - curves[:, 0, :]
        curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
        dir_vec = get_dir_vec(angle)

        dot_prods = np.dot(curve_headings, dir_vec)

        # Closest curve = one with largest dotprod
        cps = curves[np.argmax(dot_prods)]

        # Find closest point and tangent to this curve
        t = bezier_closest(cps, pos)
        point = bezier_point(cps, t)
        tangent = bezier_tangent(cps, t)

        return point, tangent

    def get_lane_pos2(self, pos, angle):
        """
        Get the position of the agent relative to the center of the right lane

        Raises NotInLane if the Duckiebot is not in a lane.
        """

        # Get the closest point along the right lane's Bezier curve,
        # and the tangent at that point
        point, tangent = self.closest_curve_point(pos, angle)
        if point is None:
            msg = 'Point not in lane: %s' % pos
            raise NotInLane(msg)

        assert point is not None

        # Compute the alignment of the agent direction with the curve tangent
        dirVec = get_dir_vec(angle)
        dotDir = np.dot(dirVec, tangent)
        dotDir = max(-1, min(1, dotDir))

        # Compute the signed distance to the curve
        # Right of the curve is negative, left is positive
        posVec = pos - point
        upVec = np.array([0, 1, 0])
        rightVec = np.cross(tangent, upVec)
        signedDist = np.dot(posVec, rightVec)

        # Compute the signed angle between the direction and curve tangent
        # Right of the tangent is negative, left is positive
        angle_rad = math.acos(dotDir)

        if np.dot(dirVec, rightVec) < 0:
            angle_rad *= -1

        angle_deg = np.rad2deg(angle_rad)
        # return signedDist, dotDir, angle_deg

        return LanePosition(dist=signedDist, dot_dir=dotDir, angle_deg=angle_deg,
                            angle_rad=angle_rad)

    def _drivable_pos(self, pos):
        """
        Check that the given (x,y,z) position is on a drivable tile
        """

        coords = self.get_grid_coords(pos)
        tile = self._get_tile(*coords)
        if tile is None:
            msg = f'No tile found at {pos} {coords}'
            logger.debug(msg)
            return False

        if not tile['drivable']:
            msg = f'{pos} corresponds to tile at {coords} which is not drivable: {tile}'
            logger.debug(msg)
            return False

        return True

    def proximity_penalty2(self, pos, angle):
        """
        Calculates a 'safe driving penalty' (used as negative rew.)
        as described in Issue #24

        Describes the amount of overlap between the "safety circles" (circles
        that extend further out than BBoxes, giving an earlier collision 'signal'
        The number is max(0, prox.penalty), where a lower (more negative) penalty
        means that more of the circles are overlapping
        """

        pos = _actual_center(pos, angle)
        if len(self.collidable_centers) == 0:
            static_dist = 0

        # Find safety penalty w.r.t static obstacles
        else:
            d = np.linalg.norm(self.collidable_centers - pos, axis=1)

            if not safety_circle_intersection(d, AGENT_SAFETY_RAD, self.collidable_safety_radii):
                static_dist = 0
            else:
                static_dist = safety_circle_overlap(d, AGENT_SAFETY_RAD, self.collidable_safety_radii)

        total_safety_pen = static_dist
        for obj in self.objects:
            # Find safety penalty w.r.t dynamic obstacles
            total_safety_pen += obj.proximity(pos, AGENT_SAFETY_RAD)

        return total_safety_pen

    def _inconvenient_spawn(self, pos):
        """
        Check that agent spawn is not too close to any object
        """

        results = [np.linalg.norm(x.pos - pos) <
                   max(x.max_coords) * 0.5 * x.scale + MIN_SPAWN_OBJ_DIST
                   for x in self.objects if x.visible
                   ]
        return np.any(results)

    def _collision(self, agent_corners):
        """
        Tensor-based OBB Collision detection
        """

        # If there are no objects to collide against, stop
        if len(self.collidable_corners) == 0:
            return False

        # Generate the norms corresponding to each face of BB
        agent_norm = generate_norm(agent_corners)

        # Check collisions with static objects
        collision = intersects(
                agent_corners,
                self.collidable_corners,
                agent_norm,
                self.collidable_norms
        )

        if collision:
            return True

        # Check collisions with Dynamic Objects
        for obj in self.objects:
            if obj.check_collision(agent_corners, agent_norm):
                return True

        # No collision with any object
        return False

    def _valid_pose(self, pos, angle, safety_factor=1.0):
        """
            Check that the agent is in a valid pose

            safety_factor = minimum distance
        """

        # Compute the coordinates of the base of both wheels
        pos = _actual_center(pos, angle)
        f_vec = get_dir_vec(angle)
        r_vec = get_right_vec(angle)

        l_pos = pos - (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        r_pos = pos + (safety_factor * 0.5 * ROBOT_WIDTH) * r_vec
        f_pos = pos + (safety_factor * 0.5 * ROBOT_LENGTH) * f_vec

        # Check that the center position and
        # both wheels are on drivable tiles and no collisions

        all_drivable = (self._drivable_pos(pos) and
                        self._drivable_pos(l_pos) and
                        self._drivable_pos(r_pos) and
                        self._drivable_pos(f_pos))


        # Recompute the bounding boxes (BB) for the agent
        agent_corners = get_agent_corners(pos, angle)
        no_collision = not self._collision(agent_corners)

        res = (no_collision and all_drivable)

        if not res:
            logger.debug(f'Invalid pose. Collision free: {no_collision} On drivable area: {all_drivable}')
            logger.debug(f'safety_factor: {safety_factor}')
            logger.debug(f'pos: {pos}')
            logger.debug(f'l_pos: {l_pos}')
            logger.debug(f'r_pos: {r_pos}')
            logger.debug(f'f_pos: {f_pos}')

        return res

    def update_physics(self, action, delta_time=None):
        if delta_time is None:
            delta_time = self.delta_time
        self.wheelVels = action * self.robot_speed * 1
        prev_pos = self.cur_pos

        # Update the robot's position
        self.cur_pos, self.cur_angle = _update_pos(self, action)

        self.step_count += 1
        self.timestamp += delta_time

        self.last_action = action

        # Compute the robot's speed
        delta_pos = self.cur_pos - prev_pos
        self.speed = np.linalg.norm(delta_pos) / delta_time

        # Update world objects
        for obj in self.objects:
            if not obj.static and obj.kind == "duckiebot":
                obj_i, obj_j = self.get_grid_coords(obj.pos)
                same_tile_obj = [
                    o for o in self.objects if
                    tuple(self.get_grid_coords(o.pos)) == (obj_i, obj_j) and o != obj
                ]

                obj.step(delta_time, self.closest_curve_point, same_tile_obj)
            else:
                obj.step(delta_time)

    def get_agent_info(self):
        info = {}
        pos = self.cur_pos
        angle = self.cur_angle
        # Get the position relative to the right lane tangent

        info['action'] = list(self.last_action)
        if self.full_transparency:
            #             info['desc'] = """
            #
            # cur_pos, cur_angle ::  simulator frame (non cartesian)
            #
            # egovehicle_pose_cartesian :: cartesian frame
            #
            #     the map goes from (0,0) to (grid_height, grid_width)*self.road_tile_size
            #
            # """
            try:
                lp = self.get_lane_pos2(pos, angle)
                info['lane_position'] = lp.as_json_dict()
            except NotInLane:
                pass

            info['robot_speed'] = self.speed
            info['proximity_penalty'] = self.proximity_penalty2(pos, angle)
            info['cur_pos'] = [float(pos[0]), float(pos[1]), float(pos[2])]
            info['cur_angle'] = float(angle)
            info['wheel_velocities'] = [self.wheelVels[0], self.wheelVels[1]]

            # put in cartesian coordinates
            # (0,0 is bottom left)
            # q = self.cartesian_from_weird(self.cur_pos, self.)
            # info['cur_pos_cartesian'] = [float(p[0]), float(p[1])]
            # info['egovehicle_pose_cartesian'] = {'~SE2Transform': {'p': [float(p[0]), float(p[1])],
            #                                                        'theta': angle}}

            info['timestamp'] = self.timestamp
            info['tile_coords'] = list(self.get_grid_coords(pos))
            # info['map_data'] = self.map_data
        misc = {}
        misc['Simulator'] = info
        return misc

    def cartesian_from_weird(self, pos, angle) -> np.ndarray:
        gx, gy, gz = pos
        grid_height = self.grid_height
        tile_size = self.road_tile_size

        # this was before but obviously doesn't work for grid_height = 1
        # cp = [gx, (grid_height - 1) * tile_size - gz]
        cp = [gx, grid_height * tile_size - gz]

        return geometry.SE2_from_translation_angle(cp, angle)

    def weird_from_cartesian(self, q: np.ndarray) -> Tuple[list, float]:

        cp, angle = geometry.translation_angle_from_SE2(q)

        gx = cp[0]
        gy = 0
        # cp[1] = (grid_height - 1) * tile_size - gz
        grid_height = self.grid_height
        tile_size = self.road_tile_size
        # this was before but obviously doesn't work for grid_height = 1
        # gz = (grid_height - 1) * tile_size - cp[1]
        gz = grid_height * tile_size - cp[1]
        return [gx, gy, gz], angle

    def compute_reward(self, pos, angle, speed):
        # Compute the collision avoidance penalty
        col_penalty = self.proximity_penalty2(pos, angle)

        # Get the position relative to the right lane tangent
        try:
            lp = self.get_lane_pos2(pos, angle)
        except NotInLane:
            reward = 40 * col_penalty
        else:

            # Compute the reward
            reward = (
                    +1.0 * speed * lp.dot_dir +
                    -10 * np.abs(lp.dist) +
                    +40 * col_penalty
            )
        return reward

    def step(self, action: np.ndarray):
        action = np.clip(action, -1, 1)
        # Actions could be a Python list
        action = np.array(action)
        for _ in range(self.frame_skip):
            self.update_physics(action)

        # Generate the current camera image
        obs = self.render_obs()
        misc = self.get_agent_info()

        d = self._compute_done_reward()
        misc['Simulator']['msg'] = d.done_why

        return obs, d.reward, d.done, misc

    def _compute_done_reward(self) -> DoneRewardInfo:
        # If the agent is not in a valid pose (on drivable tiles)
        if not self._valid_pose(self.cur_pos, self.cur_angle):
            msg = 'Stopping the simulator because we are at an invalid pose.'
            logger.info(msg)
            reward = REWARD_INVALID_POSE
            done_code = 'invalid-pose'
            done = True
        # If the maximum time step count is reached
        elif self.step_count >= self.max_steps:
            msg = 'Stopping the simulator because we reached max_steps = %s' % self.max_steps
            logger.info(msg)
            done = True
            reward = 0
            done_code = 'max-steps-reached'
        else:
            done = False
            reward = self.compute_reward(self.cur_pos, self.cur_angle, self.robot_speed)
            msg = ''
            done_code = 'in-progress'
        return DoneRewardInfo(done=done, done_why=msg, reward=reward, done_code=done_code)

    def _render_img(self, width, height, multi_fbo,
                    final_fbo, img_array, top_down=True):
        """
        Render an image of the environment into a frame buffer
        Produce a numpy RGB array image as output
        """

        if not self.graphics:
            return

        # Switch to the default context
        # This is necessary on Linux nvidia drivers
        # pyglet.gl._shadow_window.switch_to()
        self.shadow_window.switch_to()

        from pyglet import gl
        # Bind the multisampled frame buffer
        gl.glEnable(gl.GL_MULTISAMPLE)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)
        gl.glViewport(0, 0, width, height)

        # Clear the color and depth buffers

        c0, c1, c2 = self.horizon_color
        gl.glClearColor(c0, c1, c2, 1.0)
        gl.glClearDepth(1.0)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)

        # Set the projection matrix
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.gluPerspective(
                self.cam_fov_y,
                width / float(height),
                0.04,
                100.0
        )

        # Set modelview matrix
        # Note: we add a bit of noise to the camera position for data augmentation
        pos = self.cur_pos
        angle = self.cur_angle
        # logger.info('Pos: %s angle %s' % (self.cur_pos, self.cur_angle))
        if self.domain_rand:
            pos = pos + self.randomization_settings['camera_noise']

        x, y, z = pos + self.cam_offset
        dx, dy, dz = get_dir_vec(angle)
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()

        if self.draw_bbox:
            y += 0.8
            gl.glRotatef(90, 1, 0, 0)
        elif not top_down:
            y += self.cam_height
            gl.glRotatef(self.cam_angle[0], 1, 0, 0)
            gl.glRotatef(self.cam_angle[1], 0, 1, 0)
            gl.glRotatef(self.cam_angle[2], 0, 0, 1)
            gl.glTranslatef(0, 0, self._perturb(CAMERA_FORWARD_DIST))

        if top_down:
            gl.gluLookAt(
                    # Eye position
                    (self.grid_width * self.road_tile_size) / 2,
                    5,
                    (self.grid_height * self.road_tile_size) / 2,
                    # Target
                    (self.grid_width * self.road_tile_size) / 2,
                    0,
                    (self.grid_height * self.road_tile_size) / 2,
                    # Up vector
                    0, 0, -1.0
            )
        else:
            gl.gluLookAt(
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
        gl.glDisable(gl.GL_TEXTURE_2D)
        gl.glColor3f(*self.ground_color)
        gl.glPushMatrix()
        gl.glScalef(50, 1, 50)
        self.ground_vlist.draw(gl.GL_QUADS)
        gl.glPopMatrix()

        # Draw the ground/noise triangles
        self.tri_vlist.draw(gl.GL_TRIANGLES)

        # Draw the road quads
        gl.glEnable(gl.GL_TEXTURE_2D)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

        # For each grid tile
        for j in range(self.grid_height):
            for i in range(self.grid_width):
                # Get the tile type and angle
                tile = self._get_tile(i, j)

                if tile is None:
                    continue

                # kind = tile['kind']
                angle = tile['angle']
                color = tile['color']
                texture = tile['texture']

                gl.glColor3f(*color)

                gl.glPushMatrix()
                gl.glTranslatef((i + 0.5) * self.road_tile_size, 0, (j + 0.5) * self.road_tile_size)
                gl.glRotatef(angle * 90, 0, 1, 0)

                # Bind the appropriate texture
                texture.bind()

                self.road_vlist.draw(gl.GL_QUADS)
                gl.glPopMatrix()

                if self.draw_curve and tile['drivable']:
                    # Find curve with largest dotproduct with heading
                    curves = self._get_tile(i, j)['curves']
                    curve_headings = curves[:, -1, :] - curves[:, 0, :]
                    curve_headings = curve_headings / np.linalg.norm(curve_headings).reshape(1, -1)
                    dirVec = get_dir_vec(angle)
                    dot_prods = np.dot(curve_headings, dirVec)

                    # Current ("closest") curve drawn in Red
                    pts = curves[np.argmax(dot_prods)]
                    bezier_draw(pts, n=20, red=True)

                    pts = self._get_curve(i, j)
                    for idx, pt in enumerate(pts):
                        # Don't draw current curve in blue
                        if idx == np.argmax(dot_prods):
                            continue
                        bezier_draw(pt, n=20)

        # For each object
        for idx, obj in enumerate(self.objects):
            obj.render(self.draw_bbox)

        # Draw the agent's own bounding box
        if self.draw_bbox:
            corners = get_agent_corners(pos, angle)
            gl.glColor3f(1, 0, 0)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(corners[0, 0], 0.01, corners[0, 1])
            gl.glVertex3f(corners[1, 0], 0.01, corners[1, 1])
            gl.glVertex3f(corners[2, 0], 0.01, corners[2, 1])
            gl.glVertex3f(corners[3, 0], 0.01, corners[3, 1])
            gl.glEnd()

        if top_down:
            gl.glPushMatrix()
            gl.glTranslatef(*self.cur_pos)
            gl.glScalef(1, 1, 1)
            gl.glRotatef(self.cur_angle * 180 / np.pi, 0, 1, 0)
            # glColor3f(*self.color)
            self.mesh.render()
            gl.glPopMatrix()

        # Resolve the multisampled frame buffer into the final frame buffer
        gl.glBindFramebuffer(gl.GL_READ_FRAMEBUFFER, multi_fbo)
        gl.glBindFramebuffer(gl.GL_DRAW_FRAMEBUFFER, final_fbo)
        gl.glBlitFramebuffer(
                0, 0,
                width, height,
                0, 0,
                width, height,
                gl.GL_COLOR_BUFFER_BIT,
                gl.GL_LINEAR
        )

        # Copy the frame buffer contents into a numpy array
        # Note: glReadPixels reads starting from the lower left corner
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)
        gl.glReadPixels(
                0,
                0,
                width,
                height,
                gl.GL_RGB,
                gl.GL_UNSIGNED_BYTE,
                img_array.ctypes.data_as(POINTER(gl.GLubyte))
        )

        # Unbind the frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Flip the image because OpenGL maps (0,0) to the lower-left corner
        # Note: this is necessary for gym.wrappers.Monitor to record videos
        # properly, otherwise they are vertically inverted.
        img_array = np.ascontiguousarray(np.flip(img_array, axis=0))

        return img_array

    def render_obs(self):
        """
        Render an observation from the point of view of the agent
        """

        observation = self._render_img(
                self.camera_width,
                self.camera_height,
                self.multi_fbo,
                self.final_fbo,
                self.img_array,
                top_down=False
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort:
            observation = self.camera_model.distort(observation)

        return observation

    def render(self, mode='human', close=False):
        """
        Render the environment for human viewing
        """

        if close:
            if self.window:
                self.window.close()
            return

        top_down = mode == 'top_down'
        # Render the image
        img = self._render_img(
                WINDOW_WIDTH,
                WINDOW_HEIGHT,
                self.multi_fbo_human,
                self.final_fbo_human,
                self.img_array_human,
                top_down=top_down
        )

        # self.undistort - for UndistortWrapper
        if self.distortion and not self.undistort and mode != "free_cam":
            img = self.camera_model.distort(img)

        if mode == 'rgb_array':
            return img

        from pyglet import gl, window, image

        if self.window is None:
            config = gl.Config(double_buffer=False)
            self.window = window.Window(
                    width=WINDOW_WIDTH,
                    height=WINDOW_HEIGHT,
                    resizable=False,
                    config=config
            )

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()

        # Bind the default frame buffer
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

        # Setup orghogonal projection
        gl.glMatrixMode(gl.GL_PROJECTION)
        gl.glLoadIdentity()
        gl.glMatrixMode(gl.GL_MODELVIEW)
        gl.glLoadIdentity()
        gl.glOrtho(0, WINDOW_WIDTH, 0, WINDOW_HEIGHT, 0, 10)

        # Draw the image to the rendering window
        width = img.shape[1]
        height = img.shape[0]
        img = np.ascontiguousarray(np.flip(img, axis=0))
        img_data = image.ImageData(
                width,
                height,
                'RGB',
                img.ctypes.data_as(POINTER(gl.GLubyte)),
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
        if mode != "free_cam":
            x, y, z = self.cur_pos
            self.text_label.text = "pos: (%.2f, %.2f, %.2f), angle: %d, steps: %d, speed: %.2f m/s" % (
                x, y, z,
                int(self.cur_angle * 180 / math.pi),
                self.step_count,
                self.speed
            )
            self.text_label.draw()

        # Force execution of queued commands
        gl.glFlush()


def get_dir_vec(cur_angle):
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(cur_angle)
    z = -math.sin(cur_angle)
    return np.array([x, 0, z])


def get_right_vec(cur_angle):
    """
    Vector pointing to the right of the agent
    """

    x = math.sin(cur_angle)
    z = math.cos(cur_angle)
    return np.array([x, 0, z])


def _update_pos(self, action):
    """
    Update the position of the robot, simulating differential drive

    returns pos, angle
    """

    action = DynamicsInfo(motor_left=action[0], motor_right=action[1])

    self.state = self.state.integrate(self.delta_time, action)
    q = self.state.TSE2_from_state()[0]
    pos, angle = self.weird_from_cartesian(q)
    pos = np.asarray(pos)
    return pos, angle


def _actual_center(pos, angle):
    """
    Calculate the position of the geometric center of the agent
    The value of self.cur_pos is the center of rotation.
    """

    dir_vec = get_dir_vec(angle)
    return pos + (CAMERA_FORWARD_DIST - (ROBOT_LENGTH / 2)) * dir_vec


def get_agent_corners(pos, angle):
    agent_corners = agent_boundbox(
            _actual_center(pos, angle),
            ROBOT_WIDTH,
            ROBOT_LENGTH,
            get_dir_vec(angle),
            get_right_vec(angle)
    )
    return agent_corners
