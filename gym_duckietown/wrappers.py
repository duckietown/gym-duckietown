import math
import numpy as np
import gym
from gym import spaces
import itertools 

class DiscreteWrapper(gym.ActionWrapper):
    """
    Duckietown environment with discrete actions (left, right, forward)
    instead of continuous control
    """

    def __init__(self, env):
        gym.ActionWrapper.__init__(self, env)
        self.action_space = spaces.Discrete(3)

    def action(self, action):
        # Turn left
        if action == 0:
            vels = [0.6, +1.0]
        # Turn right
        elif action == 1:
            vels = [0.6, -1.0]
        # Go forward
        elif action == 2:
            vels = [0.7, 0.0]
        else:
            assert False, "unknown action"
        return np.array(vels)

class PyTorchObsWrapper(gym.ObservationWrapper):
    """
    Transpose the observation image tensors for PyTorch
    """

    def __init__(self, env=None):
        gym.ObservationWrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0,0,0],
            self.observation_space.high[0,0,0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return observation.transpose(2, 1, 0)


class ResizeWrapper(gym.ObservationWrapper):
    def __init__(self, env=None, resize_w=80, resize_h=80):
        gym.ObservationWrapper.__init__(self, env)
        self.resize_h = resize_h
        self.resize_w = resize_w
        obs_shape = self.observation_space.shape
        self.observation_space = spaces.Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[1, 1, 1],
            [obs_shape[0], resize_h, resize_w],
            dtype=self.observation_space.dtype)

    def observation(self, observation):
        return observation

    def reset(self):
        import cv2
        obs = gym.ObservationWrapper.reset(self)
        return cv2.resize(obs.swapaxes(0,2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC).swapaxes(0,2)

    def step(self, actions):
        import cv2
        obs, reward, done, info = gym.ObservationWrapper.step(self, actions)
        return cv2.resize(obs.swapaxes(0,2), dsize=(self.resize_w, self.resize_h), interpolation=cv2.INTER_CUBIC).swapaxes(0,2), reward, done, info


class UndistortWrapper(gym.ObservationWrapper):
    """ 
    To Undo the Fish eye transformation - undistorts the image with plumbbob distortion
    Using the default configuration parameters on the duckietown/Software repo
    https://github.com/duckietown/Software/blob/master18/catkin_ws/src/
    ...05-teleop/pi_camera/include/pi_camera/camera_info.py
    """

    def __init__(self, env=None):
        import cv2 
        gym.ObservationWrapper.__init__(self, env)

        assert env.unwrapped.distortion, "Distortion is false, no need for this wrapper"

        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix =  [
            305.5718893575089,  0,                  303.0797142544728,
            0,                  308.8338858195428,  231.8845403702499,
            0,                  0,                  1,
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [
            -0.2944667743901807, 0.0701431287084318, 
            0.0005859930422629722, -0.0006697840226199427, 0
        ]
        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # P - Projection Matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        projection_matrix = [
            220.2460277141687,  0,  301.8668918355899, 
            0,                  0,  238.6758484095299, 
            227.0880056118307,  0,  0, 
            0,                  1,  0,
        ]
        self.projection_matrix = np.reshape(projection_matrix, (3, 4))

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

    def observation(self, observation):
        return self._undistort(observation)

    def _undistort(self, observation):
        import cv2
        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.camera_matrix, 
                self.distortion_coefs, self.rectification_matrix, 
                self.projection_matrix, (W, H), cv2.CV_32FC1)

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)

        