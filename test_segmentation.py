import math
import os
import random

import numpy as np

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import AGENT_SAFETY_RAD

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.7
FOLLOWING_DISTANCE = 0.24
AGENT_SAFETY_GAIN = 1.15


class PurePursuitPolicy:
    """
    A Pure Pusuit controller class to act as an expert to the model
    ...
    Methods
    -------
    forward(images)
        makes a model forward pass on input images

    loss(*args)
        takes images and target action to compute the loss function used in optimization

    predict(observation)
        takes an observation image and predicts using env information the action
    """

    def __init__(
        self, env, ref_velocity=REF_VELOCITY, following_distance=FOLLOWING_DISTANCE, max_iterations=1000
    ):
        """
        Parameters
        ----------
        ref_velocity : float
            duckiebot maximum velocity (default 0.7)
        following_distance : float
            distance used to follow the trajectory in pure pursuit (default 0.24)
        """
        self.env = env
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity

    def predict(self, observation):
        """
        Parameters
        ----------
        observation : image
            image of current observation from simulator
        Returns
        -------
        action: list
            action having velocity and omega of current observation
        """
        closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(
            self.env.cur_pos, self.env.cur_angle
        )
        if closest_point is None or closest_tangent is None:
            self.env.reset()
            closest_point, closest_tangent = self.env.unwrapped.closest_curve_point(
                self.env.cur_pos, self.env.cur_angle
            )

        current_world_objects = self.env.objects
        # to slow down if there's a duckiebot in front of you
        # this is used to avoid hitting another moving duckiebot in the map
        # in case of training LFV baseline
        velocity_slow_down = 1
        for obj in current_world_objects:
            if not obj.static and obj.kind == "duckiebot":
                if True:
                    collision_penalty = abs(
                        obj.proximity(self.env.cur_pos, AGENT_SAFETY_RAD * AGENT_SAFETY_GAIN)
                    )
                    if collision_penalty > 0:
                        # this means we are approaching and we need to slow down
                        velocity_slow_down = collision_penalty
                        break

        lookup_distance = self.following_distance
        # projected_angle used to detect corners and to reduce the velocity accordingly
        projected_angle, _, _ = self._get_projected_angle_difference(0.3)
        velocity_scale = 1

        current_tile_pos = self.env.get_grid_coords(self.env.cur_pos)
        current_tile = self.env._get_tile(*current_tile_pos)
        if "curve" in current_tile["kind"] or abs(projected_angle) < 0.92:
            # slowing down by a scale of 0.5
            velocity_scale = 0.5
        _, closest_point, curve_point = self._get_projected_angle_difference(lookup_distance)

        if closest_point is None:  # if cannot find a curve point in max iterations
            return [0, 0]

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)
        right_vec = np.array([math.sin(self.env.cur_angle), 0, math.cos(self.env.cur_angle)])
        dot = np.dot(right_vec, point_vec)
        omega = -1 * dot
        # range of dot is just -pi/2 and pi/2 and will be multiplied later by a gain adjustable if we are
        # testing on a hardware or not
        velocity = self.ref_velocity * velocity_scale
        if velocity_slow_down < 0.2:
            velocity = 0
            omega = 0

        action = [velocity, omega]

        return action

    def _get_projected_angle_difference(self, lookup_distance):
        # Find the projection along the path
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        curve_angle = None

        while iterations < 10:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, curve_angle = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_angle is not None and curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        if curve_angle is None:  # if cannot find a curve point in max iterations
            return None, None, None

        else:
            return np.dot(curve_angle, closest_tangent), closest_point, curve_point


def seed(seed):
    # torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


seed(random.randint(0, 9999999))

from PIL import Image


def to_image(np_array):

    img = Image.fromarray(np_array, "RGB")
    img.show()
    i = 0


os.chdir("./src/gym_duckietown")

environment = DuckietownEnv(
    domain_rand=False, max_steps=math.inf, randomize_maps_on_reset=False, map_name="loop_obstacles"
)

policy = PurePursuitPolicy(environment)

MAX_STEPS = 500

while True:
    obs = environment.reset()
    environment.render(segment=True)
    rewards = []

    nb_of_steps = 0

    while True:
        action = list(policy.predict(np.array(obs)))
        action[1] *= 7

        obs, rew, done, misc = environment.step(np.array(action))
        rewards.append(rew)
        environment.render(segment=int(nb_of_steps / 50) % 2 == 0)

        # to_image(obs)

        nb_of_steps += 1

        if done or nb_of_steps > MAX_STEPS:
            break
    print("mean episode reward:", np.mean(rewards))

environment.close()
