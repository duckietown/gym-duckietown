# coding=utf-8
from .collision import *
from .graphics import load_texture
from .utils import get_file_path




class WorldObj(object):
    def __init__(self, obj, domain_rand, safety_radius_mult):
        """
        Initializes the object and its properties
        """
        # XXX this is relied on by things but it is not always set
        # (Static analysis complains)
        self.visible = True
        # same
        self.color = (0, 0, 0)
        # maybe have an abstract method is_visible, get_color()


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


    def render(self, draw_bbox):
        """
        Renders the object to screen
        """
        if not self.visible:
            return

        from pyglet import gl

        # Draw the bounding box
        if draw_bbox:
            gl.glColor3f(1, 0, 0)
            gl.glBegin(gl.GL_LINE_LOOP)
            gl.glVertex3f(self.obj_corners.T[0, 0], 0.01, self.obj_corners.T[1, 0])
            gl.glVertex3f(self.obj_corners.T[0, 1], 0.01, self.obj_corners.T[1, 1])
            gl.glVertex3f(self.obj_corners.T[0, 2], 0.01, self.obj_corners.T[1, 2])
            gl.glVertex3f(self.obj_corners.T[0, 3], 0.01, self.obj_corners.T[1, 3])
            gl.glEnd()

        gl.glPushMatrix()
        gl.glTranslatef(*self.pos)
        gl.glScalef(self.scale, self.scale, self.scale)
        gl.glRotatef(self.y_rot, 0, 1, 0)
        gl.glColor3f(*self.color)
        self.mesh.render()
        gl.glPopMatrix()

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


class DuckiebotObj(WorldObj):
    def __init__(self, obj, domain_rand, safety_radius_mult, wheel_dist, 
            robot_width, robot_length, gain=1.0, trim=0.0, radius=0.0318, 
            k=27.0, limit=1.0):
        WorldObj.__init__(self, obj, domain_rand, safety_radius_mult)

        if self.domain_rand:
            self.follow_dist = np.random.uniform(0.05, 0.15)
            self.velocity = np.random.uniform(0.05, 0.15)
        else:
            self.follow_dist = 0.05
            self.velocity = 0.1

        # TODO: Make these DR as well
        self.gain = gain
        self.trim = trim
        self.radius = radius
        self.k = k
        self.limit = limit

        self.wheel_dist = wheel_dist 

        self.robot_width = robot_width
        self.robot_length = robot_length

    # FIXME: this does not follow the same signature as WorldOb
    def step(self, delta_time, closest_curve_point, objects):
        """
        Take a step, implemented as a PID controller
        """

        # Find the curve point closest to the agent, and the tangent at that point
        closest_point, closest_tangent = closest_curve_point(self.pos, self.angle)

        while True:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * self.follow_dist
            curve_point, _ = closest_curve_point(follow_point, self.angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            self.follow_dist *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(self.get_right_vec(self.angle), point_vec)
        steering = 2 * -dot

        self._update_pos([self.velocity, steering], delta_time)

    def get_dir_vec(self, angle):
        x = math.cos(angle)
        z = -math.sin(angle)
        return np.array([x, 0, z])

    def get_right_vec(self, angle):
        x = math.sin(angle)
        z = math.cos(angle)
        return np.array([x, 0, z])

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
        d = np.linalg.norm(agent_pos - self.pos)
        score = d - agent_safety_rad - self.safety_radius

        return min(0, score)

    def _update_pos(self, action, deltaTime):
        vel, angle = action

        # assuming same motor constants k for both motors
        k_r = self.k
        k_l = self.k

        # adjusting k by gain and trim
        k_r_inv = (self.gain + self.trim) / k_r
        k_l_inv = (self.gain - self.trim) / k_l

        omega_r = (vel + 0.5 * angle * self.wheel_dist) / self.radius
        omega_l = (vel - 0.5 * angle * self.wheel_dist) / self.radius

        # conversion from motor rotation rate to duty cycle
        u_r = omega_r * k_r_inv
        u_l = omega_l * k_l_inv

        # limiting output to limit, which is 1.0 for the duckiebot
        u_r_limited = max(min(u_r, self.limit), -self.limit)
        u_l_limited = max(min(u_l, self.limit), -self.limit)

        # If the wheel velocities are the same, then there is no rotation
        if u_l_limited == u_r_limited:
            self.pos = self.pos + deltaTime * u_l_limited * self.get_dir_vec(self.angle)
            return

        # Compute the angular rotation velocity about the ICC (center of curvature)
        w = (u_r_limited - u_l_limited) / self.wheel_dist

        # Compute the distance to the center of curvature
        r = (self.wheel_dist * (u_l_limited + u_r_limited)) / (2 * (u_l_limited - u_r_limited))

        # Compute the rotation angle for this time step
        rotAngle = w * deltaTime

        # Rotate the robot's position around the center of rotation
        r_vec = self.get_right_vec(self.angle)
        px, py, pz = self.pos
        cx = px + r * r_vec[0]
        cz = pz + r * r_vec[2]
        npx, npz = rotate_point(px, pz, cx, cz, rotAngle)

        # Update position
        self.pos = np.array([npx, py, npz])

        # Update the robot's direction angle
        self.angle += rotAngle
        self.y_rot += rotAngle * 180 / np.pi 

        # Recompute the bounding boxes (BB) for the duckiebot
        self.obj_corners = agent_boundbox(
            self.pos,
            self.robot_width,
            self.robot_length,
            self.get_dir_vec(self.angle),
            self.get_right_vec(self.angle)
        )


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
            # TODO: Fix this: This will go to 0 over time
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
