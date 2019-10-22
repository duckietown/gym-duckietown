# coding=utf-8
import itertools

import cv2
import numpy as np
import carnivalmirror as cm


class Distortion(object):
    def __init__(self, camera_rand=False):
        # Image size
        self.H = 480
        self.W = 640
        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix = [
            305.5718893575089, 0, 303.0797142544728,
            0, 308.8338858195428, 231.8845403702499,
            0, 0, 1,
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [
            -0.2, 0.0305,
            0.0005859930422629722, -0.0006697840226199427, 0
        ]

        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # P - Projection Matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        projection_matrix = [
            220.2460277141687, 0, 301.8668918355899, 0,
            0, 238.6758484095299, 227.0880056118307, 0,
            0, 0, 1, 0,
        ]
        self.projection_matrix = np.reshape(projection_matrix, (3, 4))

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

        # Used for distortion
        self.rmapx = None
        self.rmapy = None
        if camera_rand:
            self.camera_matrix, self.distortion_coefs = self.randomize_camera()

    def randomize_camera(self):
        # Create a Calibration object for the correct calibration. That will be used
        # as a reference calibration when calculating APPD values
        reference = cm.Calibration(K=self.camera_matrix, D=self.distortion_coefs,
                                   width=self.W, height=self.H)

        # Define ranges for the parameters:
        # TODO move this to config file
        K = self.camera_matrix

        ranges = {'fx': (0.95 * K[0, 0], 1.05 * K[0, 0]),
                  'fy': (0.95 * K[1, 1], 1.05 * K[1, 1]),
                  'cx': (0.95 * K[0, 2], 1.05 * K[0, 2]),
                  'cy': (0.95 * K[1, 2], 1.05 * K[1, 2]),
                  'k1': (-0.03, 0.03),
                  'k2': (-0.003, 0.003),
                  'p1': (-0.00002, 0.00002),
                  'p2': (-0.0002, 0.0002),
                  'k3': (-0.0001, 0.0001)}

        # Create a Sampler object. Two samplers are provided:
        #   ParameterSampler:   samples uniformly in the camera intrinsics and
        #                       distortion parameters on a specified range

        # Create a ParameterSampler:
        sampler = cm.ParameterSampler(ranges=ranges, cal_width=self.W, cal_height=self.H)

        # Create a UniformAPPDSampler
        # sampler = cm.UniformAPPDSampler(ranges=ranges, cal_width=self.W, cal_height=self.H, reference=reference,
        #                                 temperature=5, appd_range_dicovery_samples=1000, appd_range_bins=10,
        #                                 width=int(self.W / 6), height=int(self.H / 6),
        #                                 min_cropped_size=(int(self.W / 6 / 1.5), int(self.H / 6 / 1.5)))
        # sampler = cm.ParallelBufferedSampler(sampler=sampler, buffer_size=16, n_jobs=4)

        calibration = sampler.next()

        # Do that to stop the background sampling
        # sampler.stop()

        return calibration.get_K(), calibration.get_D()

    def distort(self, observation):
        """
        Distort observation using parameters in constructor
        """

        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(self.camera_matrix,
                                                               self.distortion_coefs, self.rectification_matrix,
                                                               self.projection_matrix, (W, H), cv2.CV_32FC1)

            # Invert the transformations for the distortion
            self.rmapx, self.rmapy = self._invert_map(self.mapx, self.mapy)

        return cv2.remap(observation, self.rmapx, self.rmapy, interpolation=cv2.INTER_NEAREST)

    def _undistort(self, observation):
        """
        Undistorts a distorted image using camera parameters
        """

        # If mapx is None, then distort was never called
        assert self.mapx is not None, "You cannot call undistort on a rectified image"

        return cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)

    def _invert_map(self, mapx, mapy):
        """
        Utility function for simulating distortion
        Source: https://github.com/duckietown/Software/blob/master18/catkin_ws
        ... /src/10-lane-control/ground_projection/include/ground_projection/
        ... ground_projection_geometry.py
        """

        H, W = mapx.shape[0:2]
        rmapx = np.empty_like(mapx)
        rmapx.fill(np.nan)
        rmapy = np.empty_like(mapx)
        rmapy.fill(np.nan)

        for y, x in itertools.product(range(H), range(W)):
            tx = mapx[y, x]
            ty = mapy[y, x]

            tx = int(np.round(tx))
            ty = int(np.round(ty))

            if (0 <= tx < W) and (0 <= ty < H):
                rmapx[ty, tx] = x
                rmapy[ty, tx] = y

        self._fill_holes(rmapx, rmapy)
        return rmapx, rmapy

    def _fill_holes(self, rmapx, rmapy):
        """
        Utility function for simulating distortion
        Source: https://github.com/duckietown/Software/blob/master18/catkin_ws
        ... /src/10-lane-control/ground_projection/include/ground_projection/
        ... ground_projection_geometry.py
        """
        H, W = rmapx.shape[0:2]

        R = 2
        F = R * 2 + 1

        def norm(_):
            return np.hypot(_[0], _[1])

        deltas0 = [(i - R - 1, j - R - 1) for i, j in itertools.product(range(F), range(F))]
        deltas0 = [x for x in deltas0 if norm(x) <= R]
        deltas0.sort(key=norm)

        def get_deltas():
            return deltas0

        holes = set()

        for i, j in itertools.product(range(H), range(W)):
            if np.isnan(rmapx[i, j]):
                holes.add((i, j))

        while holes:
            nholes = len(holes)
            nholes_filled = 0

            for i, j in list(holes):
                # there is nan
                nholes += 1
                for di, dj in get_deltas():
                    u = i + di
                    v = j + dj
                    if (0 <= u < H) and (0 <= v < W):
                        if not np.isnan(rmapx[u, v]):
                            rmapx[i, j] = rmapx[u, v]
                            rmapy[i, j] = rmapy[u, v]
                            nholes_filled += 1
                            holes.remove((i, j))
                            break

            if nholes_filled == 0:
                break