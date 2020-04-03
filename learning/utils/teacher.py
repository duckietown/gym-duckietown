import numpy as np
import gym


# parameters for the pure pursuit controller
POSITION_THRESHOLD = 0.04
REF_VELOCITY = 0.8
GAIN = 4
FOLLOWING_DISTANCE = 0.3


class PurePursuitExpert:
    def __init__(self, env, ref_velocity=REF_VELOCITY, position_threshold=POSITION_THRESHOLD,
                 following_distance=FOLLOWING_DISTANCE, gain=GAIN, max_iterations=1000):
        self.env = env.unwrapped
        self.following_distance = following_distance
        self.max_iterations = max_iterations
        self.ref_velocity = ref_velocity
        self.position_threshold = position_threshold
        self.gain = gain
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

        dot = np.dot(self.env.get_right_vec(), point_vec)
        steering = self.gain * -dot

        return self.ref_velocity, steering
    
class CartpoleController:
    """
    From: https://github.com/NaleRaphael/cartpole-control/blob/beb4aebd0d9174c08fe1e7fbb24e3b796bb6f37b/cpc/agent.py
    This agent is a pure PID controller, so its parameters (kp, ki, kd) is not
    going to be tuned automatically.
    However, we can apply a learning model to tune them later.
    """
    def __init__(self, env, kp=1, ki=0, kd=75,set_angle=0.0):
        """
        Parameters
        ----------
        action_space : gym.spaces
            Determine the valid action that agent can generate.
        fs : float
            Samping frequency. (Hz)
        kp : float
            Gain of propotional controller.
        ki : float
            Gain of integral controller.
        kd : float
            Gain of derivative controller.
        """
        self.env = env
        self.action_space = env.action_space
        self.kp = kp
        self.ki = ki
        self.kd = kd

        self.set_angle = set_angle
        self.tau = 1.0/self.env.metadata['video.frames_per_second']

        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

        # cache
        self.output = 0.0
        self.err_prev = 0.0

    def controller(self, v_in, v_fb):
        """
        Parameters
        ----------
        v_in : int or float
            Input command.
        v_fb : int or float
            Feedback from observer.
        Returns
        -------
        output : float
            Output command.
        Note
        ----
        Output of PID controller:
        .. math::
            u(t) = K_p e(t) + K_i \int_{0}^{t} e(t)dt + K_d {de}/{dt}
        """
        err = v_in - v_fb

        self.p_term = err
        self.i_term += err*self.tau
        self.d_term = (err - self.err_prev)*self.tau
        self.output = self.kp*self.p_term + self.ki*self.i_term + self.kd*self.d_term

        # update cache
        self.err_prev = err

        return self.output

    def choose_action(self, val):
        if isinstance(self.action_space, gym.spaces.Discrete):
            action = 0 if val >= 0 else 1
        elif isinstance(self.action_space, gym.spaces.Box):
            action = None   # rewrite this for continous action space
        return action

    def predict(self, state):
        output = self.controller(self.set_angle, state[2])
        temp = self.choose_action(output)
        self.action = temp
        return self.action