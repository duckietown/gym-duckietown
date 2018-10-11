import numpy as np
from .collision import *
from .graphics import load_texture
from .utils import get_file_path

import pyglet
from pyglet.gl import *


class WorldObj():
    def __init__(self, obj, domain_rand, safety_radius_mult):
        """
        Initializes the object and its properties
        """
        self.process_obj_dict(obj, safety_radius_mult)

        self.domain_rand = domain_rand
        self.angle = self.y_rot * (math.pi / 180)

        self.generate_geometry()

    def generate_geometry(self):
        # Find corners and normal vectors assoc w. object
        self.obj_corners = generate_corners(self.pos,
            self.min_coords, self.max_coords, self.angle, self.scale)
        self.obj_norm = generate_norm(self.obj_corners)

    def process_obj_dict(self, obj, safety_radius_mult):
        self.kind = obj['kind']
        self.mesh = obj['mesh']
        self.pos = obj['pos']
        self.scale = obj['scale']
        self.y_rot = obj['y_rot']
        self.optional = obj['optional']
        self.min_coords = obj['mesh'].min_coords
        self.max_coords = obj['mesh'].max_coords
        self.static = obj['static']
        self.safety_radius = safety_radius_mult *\
            calculate_safety_radius(self.mesh, self.scale)
        self.optional = obj['optional']

    def render(self, draw_bbox):
        """
        Renders the object to screen
        """
        if not self.visible:
            return

        # Draw the bounding box
        if draw_bbox:
            glColor3f(1, 0, 0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(self.obj_corners.T[0, 0], 0.01, self.obj_corners.T[1, 0])
            glVertex3f(self.obj_corners.T[0, 1], 0.01, self.obj_corners.T[1, 1])
            glVertex3f(self.obj_corners.T[0, 2], 0.01, self.obj_corners.T[1, 2])
            glVertex3f(self.obj_corners.T[0, 3], 0.01, self.obj_corners.T[1, 3])
            glEnd()

        glPushMatrix()
        glTranslatef(*self.pos)
        glScalef(self.scale, self.scale, self.scale)
        glRotatef(self.y_rot, 0, 1, 0)
        glColor3f(*self.color)
        self.mesh.render()
        glPopMatrix()

    # Below are the functions that need to
    # be reimplemented for any dynamic object
    def check_collision(self, agent_corners, agent_norm):
        """
        See if the agent collided with this object
        For static, return false (static collisions checked w
        numpy in a batch operation)
        """
        if not self.static:
            raise NotImplementedError
        return False

    def proximity(self, agent_pos, agent_safety_rad):
        """
        See if the agent is too close to this object
        For static, return 0 (static safedriving checked w
        numpy in a batch operation)
        """
        if not self.static:
            raise NotImplementedError
        return 0.0

    def step(self, delta_time):
        """
        Use a motion model to move the object in the world
        """
        if not self.static:
            raise NotImplementedError


class DuckieObj(WorldObj):
    def __init__(self, obj, domain_rand, safety_radius_mult, walk_distance):
        WorldObj.__init__(self, obj, domain_rand, safety_radius_mult)

        self.walk_distance = walk_distance + 0.25

        # Dynamic duckie stuff

        # Randomize velocity and wait time
        if self.domain_rand:
            self.pedestrian_wait_time = np.random.randint(3, 20)
            self.vel = np.abs(np.random.normal(0.02, 0.005))
        else:
            self.pedestrian_wait_time = 8
            self.vel = 0.02

        # Movement parameters
        self.heading = heading_vec(self.angle)
        self.start = np.copy(self.pos)
        self.center = self.pos
        self.pedestrian_active = False

        # Walk wiggle parameter
        self.wiggle = np.random.choice([14, 15, 16], 1)
        self.wiggle = np.pi / self.wiggle

        self.time = 0

    def check_collision(self, agent_corners, agent_norm):
        """
        See if the agent collided with this object
        """
        return intersects_single_obj(
            agent_corners,
            self.obj_corners.T,
            agent_norm,
            self.obj_norm
        )

    def proximity(self, agent_pos, agent_safety_rad):
        """
        See if the agent is too close to this object
        based on a heuristic for the "overlap" between
        their safety circles
        """
        d = np.linalg.norm(agent_pos - self.center)
        score = d - agent_safety_rad - self.safety_radius

        return min(0, score)

    def step(self, delta_time):
        """
        Use a motion model to move the object in the world
        """

        self.time += delta_time

        # If not walking, no need to do anything
        if not self.pedestrian_active:
            self.pedestrian_wait_time -= delta_time
            if self.pedestrian_wait_time <= 0:
                self.pedestrian_active = True
            return

        # Update centers and bounding box
        vel_adjust = self.heading * self.vel
        self.center += vel_adjust
        self.obj_corners += vel_adjust[[0, -1]]

        distance = np.linalg.norm(self.center - self.start)

        if distance > self.walk_distance:
            self.finish_walk()

        self.pos = self.center
        angle_delta = self.wiggle * math.sin(48 * self.time)
        self.y_rot = (self.angle + angle_delta) * (180 / np.pi)
        self.obj_norm = generate_norm(self.obj_corners)

    def finish_walk(self):
        """
        After duckie crosses, update relevant attributes
        (vel, rot, wait time until next walk)
        """
        self.start = np.copy(self.center)
        self.angle += np.pi
        self.pedestrian_active = False

        if self.domain_rand:
            # Assign a random velocity (in opp. direction) and a wait time
            self.vel = -1 * np.sign(self.vel) * np.abs(np.random.normal(0.02, 0.005))
            self.pedestrian_wait_time = np.random.randint(3, 20)
        else:
            # Just give it the negative of its current velocity
            self.vel *= -1
            self.pedestrian_wait_time = 8


class TrafficLightObj(WorldObj):
    def __init__(self, obj, domain_rand, safety_radius_mult):
        WorldObj.__init__(self, obj, domain_rand, safety_radius_mult)

        self.texs = [
            load_texture(get_file_path("textures", "trafficlight_card0", "jpg")),
            load_texture(get_file_path("textures", "trafficlight_card1", "jpg"))
        ]
        self.time = 0

        # Frequency and current pattern of the lights
        if self.domain_rand:
            self.freq = np.random.randint(4, 7)
            self.pattern = np.random.randint(0, 2)
        else:
            self.freq = 5
            self.pattern = 0

        # Use the selected pattern
        self.mesh.textures[0] = self.texs[self.pattern]

    def step(self, delta_time):
        """
        Changes the light color periodically
        """

        self.time += delta_time
        if round(self.time, 3) % self.freq == 0:  # Swap patterns
            self.pattern ^= 1
            self.mesh.textures[0] = self.texs[self.pattern]

    def is_green(self, direction='N'):
        if direction == 'N' or direction == 'S':
            if self.y_rot == 45 or self.y_rot == 135:
                return self.pattern == 0
            elif self.y_rot == 225 or self.y_rot == 315:
                return self.pattern == 1
        elif direction == 'E' or direction == 'W':
            if self.y_rot == 45 or self.y_rot == 135:
                return self.pattern == 1
            elif self.y_rot == 225 or self.y_rot == 315:
                return self.pattern == 0
        return False
