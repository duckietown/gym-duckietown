import numpy as np


# parameters for the pure pursuit controller
from gym_duckietown.simulator import get_right_vec

POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.8
GAIN = 10
FOLLOWING_DISTANCE = 0.3


class PurePursuitExpert:
    def __init__(
        self,
        env,
        ref_velocity=REF_VELOCITY,
        position_threshold=POSITION_THRESHOLD,
        following_distance=FOLLOWING_DISTANCE,
        max_iterations=1000,
    ):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold

    def predict(self, observation):  # we don't really care about the observation for this implementation
        closest_point, closest_tangent = self.env.closest_curve_point(self.env.cur_pos, self.env.cur_angle)

        iterations = 0
        lookup_distance = self.following_distance
        curve_point = None
        while iterations < self.max_iterations:
            # Project a point ahead along the curve tangent,
            # then find the closest point to to that
            follow_point = closest_point + closest_tangent * lookup_distance
            curve_point, _ = self.env.closest_curve_point(follow_point, self.env.cur_angle)

            # If we have a valid point on the curve, stop
            if curve_point is not None:
                break

            iterations += 1
            lookup_distance *= 0.5

        # Compute a normalized vector to the curve point
        point_vec = curve_point - self.env.cur_pos
        point_vec /= np.linalg.norm(point_vec)

        dot = np.dot(get_right_vec(self.env.cur_angle), point_vec)
        steering = GAIN * -dot

        return self.ref_velocity, steering
