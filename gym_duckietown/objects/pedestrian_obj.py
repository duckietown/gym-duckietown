import numpy as np
from ..collision import *

from world_obj import WorldObj

class PedestrianObj(WorldObj):
	def __init__(self, obj, domain_rand):
		super().__init__(self, obj, domain_rand)

		# Dynamic duckie stuff

		# Randomize velocity and wait time
        if self.domain_rand:
            self.pedestrian_wait_time = np.random.randint(1, 30) * 10
            self.vel = np.abs(np.random.normal(0.2, 0.02))
        else:
            self.pedestrian_wait_time = 100
            self.vel = 0.02

        angle = self.obj['y_rot'] * (math.pi / 180)
        self.rot = angle

        self.heading = heading_vec(angle)
        self.start = self.obj['pos']
        self.center = self.obj['pos']

    def check_collision(self):
        pass

    def safe_driving(self):
        pass

    def step(self):
        pass

    def update_corners(self):
        pass