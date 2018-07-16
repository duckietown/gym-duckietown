import numpy as np
from ..collision import *

from .world_obj import WorldObj

class PedestrianObj(WorldObj):
    def __init__(self, obj, domain_rand, draw_bbox, road_tile_sz, dt=0.05):
        super().__init__(obj, domain_rand, draw_bbox)

        # Dynamic duckie stuff

        # Randomize velocity and wait time
        if self.domain_rand:
            self.pedestrian_wait_time = np.random.randint(1, 30) * 10
            self.vel = np.abs(np.random.normal(0.1, 0.02))
        else:
            self.pedestrian_wait_time = 100
            self.vel = 0.02

        self.heading = heading_vec(self.angle)
        self.start = np.copy(self.obj['pos'])
        self.center = self.obj['pos']
        self.pedestrian_active = False
        self.step_count = 0
        self.road_tile_sz = road_tile_sz
        self.wiggle = np.random.choice([10, 11, 12], 1)
        self.wiggle = np.pi / self.wiggle

    def check_collision(self, agent_corners, agent_norm):
        return intersects_single_obj(
            agent_corners,
            self.obj_corners.T,
            agent_norm,
            self.obj_norm
        )

    def safe_driving(self, agent_pos, agent_safety_rad):
        d = np.linalg.norm(agent_pos - self.center)
        score = d - agent_safety_rad - self.obj['safety_radius']

        return min(0, score)

    def step(self):
        self.step_count += 1
        self.pedestrian_active = np.logical_or(
            self.step_count % self.pedestrian_wait_time == 0,
            self.pedestrian_active
        )

        if not self.pedestrian_active: 
            return

        vel_adjust = self.heading * self.vel
        self.center += vel_adjust
        self.obj_corners += vel_adjust[[0, -1]]

        distance = np.linalg.norm(self.center - self.start)

        self.update_rendering()
        if distance > self.road_tile_sz:
            self.finish_walk()
            

    def update_rendering(self):
        self.obj['pos'] = self.center
        self.obj['y_rot'] = (self.angle + self.wiggle) * (180 / np.pi) 
        self.wiggle *= -1
        self.obj_norm = generate_norm(self.obj_corners)

    def finish_walk(self):
        self.start = np.copy(self.center)
        self.angle += np.pi
        self.pedestrian_active = False

        if self.domain_rand:
            # Assign a random velocity (in opp. direction) and a wait time
            self.vel = -1 * np.sign(self.vel) * np.abs(np.random.normal(0.1, 0.02))
            self.pedestrian_wait_time = np.random.randint(1, 30) * 10
        else:
            # Just give it the negative of its current velocuty
            self.vel *= -1