# coding=utf-8
import itertools
import time

import carnivalmirror as cm
import cv2
import numpy as np


class Distortion:
    def __init__(self, camera_rand=False):
        # Image size
        self.H = 480
        self.W = 640
        # K - Intrinsic camera matrix for the raw (distorted) images.
        camera_matrix = [
            305.5718893575089,
            0,
            303.0797142544728,
            0,
            308.8338858195428,
            231.8845403702499,
            0,
            0,
            1,
        ]
        self.camera_matrix = np.reshape(camera_matrix, (3, 3))

        # distortion parameters - (k1, k2, t1, t2, k3)
        distortion_coefs = [-0.2, 0.0305, 0.0005859930422629722, -0.0006697840226199427, 0]

        self.distortion_coefs = np.reshape(distortion_coefs, (1, 5))

        # R - Rectification matrix - stereo cameras only, so identity
        self.rectification_matrix = np.eye(3)

        # Initialize mappings

        # Used for rectification
        self.mapx = None
        self.mapy = None

        # Used for distortion
        self.rmapx = None
        self.rmapy = None
        if camera_rand:
            self.camera_matrix, self.distortion_coefs = self.randomize_camera()

        # New camera matrix - specifies the intrinsic (camera) matrix
        #  of the processed (rectified) image
        self.new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
            cameraMatrix=self.camera_matrix,
            distCoeffs=self.distortion_coefs,
            imageSize=(self.W, self.H),
            alpha=0,
        )

    def randomize_camera(self):
        """Randomizes parameters of the camera according to a specified range"""
        K = self.camera_matrix
        D = self.distortion_coefs

        # Define ranges for the parameters:
        # TODO move this to config file
        ranges = {
            "fx": (0.95 * K[0, 0], 1.05 * K[0, 0]),
            "fy": (0.95 * K[1, 1], 1.05 * K[1, 1]),
            "cx": (0.95 * K[0, 2], 1.05 * K[0, 2]),
            "cy": (0.95 * K[1, 2], 1.05 * K[1, 2]),
            "k1": (0.95 * D[0, 0], 1.05 * D[0, 0]),
            "k2": (0.95 * D[0, 1], 1.05 * D[0, 1]),
            "p1": (0.95 * D[0, 2], 1.05 * D[0, 2]),
            "p2": (0.95 * D[0, 3], 1.05 * D[0, 3]),
            "k3": (0.95 * D[0, 4], 1.05 * D[0, 4]),
        }

        # Create a ParameterSampler:
        sampler = cm.ParameterSampler(ranges=ranges, cal_width=self.W, cal_height=self.H)

        # Get a calibration from sampler
        calibration = sampler.next()

        return calibration.get_K(self.H), calibration.get_D()

    def distort(self, observation, interpolation=cv2.INTER_NEAREST):
        """
        Distort observation using parameters in constructor

        cv2.INTER_NEAREST, INTER_LINEAR
        """

        if self.mapx is None:
            # Not initialized - initialize all the transformations we'll need
            self.mapx = np.zeros(observation.shape)
            self.mapy = np.zeros(observation.shape)

            H, W, _ = observation.shape

            # Initialize self.mapx and self.mapy (updated)
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                cameraMatrix=self.camera_matrix,
                distCoeffs=self.distortion_coefs,
                R=self.rectification_matrix,
                newCameraMatrix=self.new_camera_matrix,
                size=(W, H),
                m1type=cv2.CV_32FC1,
            )

            # print(self.mapx.dtype, self.mapy.dtype)
            # Invert the transformations for the distortion
            self.rmapx, self.rmapy = self._invert_map(self.mapx, self.mapy)
            # write_to_file(self.rmapx, 'rmapx.jpg')
            # write_to_file(self.rmapy, 'rmapy.jpg')
            #
            # write_to_file(self.mapx, 'mapx.jpg')
            # write_to_file(self.mapy, 'mapy.jpg')

        res = cv2.remap(
            observation,
            self.rmapx,
            self.rmapy,
            interpolation=interpolation,
            # borderMode=cv2.BORDER_REPLICATE,
        )
        return res

    def _undistort(self, observation: np.array) -> np.array:
        """
        Undistorts a distorted image using camera parameters
        """

        # If mapx is None, then distort was never called
        assert self.mapx is not None, "You cannot call undistort on a rectified image"

        res = cv2.remap(observation, self.mapx, self.mapy, cv2.INTER_NEAREST)
        return res

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
        #
        # closeness = np.zeros((H, W))
        # closeness.fill(100)

        around_rmapx = np.zeros((H, W), "float32")
        around_rmapy = np.zeros((H, W), "float32")
        around = np.zeros((H, W), "float32")

        # M = 1
        t0 = time.time()
        deltas = [
            (-1, -1, 7),
            (-1, 0, 10),
            (-1, +1, 7),
            (0, -1, 10),
            (0, 0, 20),
            (0, +1, 10),
            (+1, -1, 7),
            (+1, 0, 10),
            (+1, +1, 7),
        ]

        mapx_disc = np.clip(mapx.astype("int32"), 2, W - 2)
        mapy_disc = np.clip(mapy.astype("int32"), 2, H - 2)
        # Hs = [_ for _ in range(H) if _ % 2 == 0]
        # Ws = [_ for _ in range(W) if (_ + 1) % 2 == 0]
        #
        xs = np.zeros((H, W), "int32")
        ys = np.zeros((H, W), "int32")
        for j in range(W):
            xs[:, j] = j
        for i in range(H):
            ys[i, :] = i

        for di, dj, w in deltas:
            mapy_disc_d = mapy_disc + di
            mapx_disc_d = mapx_disc + dj

            around[mapy_disc_d, mapx_disc_d] += w
            around_rmapx[mapy_disc_d, mapx_disc_d] += w * xs
            around_rmapy[mapy_disc_d, mapx_disc_d] += w * ys
            #
            if False:
                for y, x in itertools.product(range(H), range(W)):
                    i = mapy_disc_d[y, x]
                    j = mapx_disc_d[y, x]

                    around_rmapx[i, j] += x * w
                    around_rmapy[i, j] += y * w
                    # around_rmapx[i, j] += xs[y, x] * w
                    # around_rmapy[i, j] += ys[y, x] * w
                    # around[i, j] += w

        dt1 = time.time() - t0
        t0 = time.time()

        nonzero = around[i, j] > 0
        rmapx[nonzero] = around_rmapx[nonzero] / around[nonzero]
        rmapy[nonzero] = around_rmapy[nonzero] / around[nonzero]
        for i, j in itertools.product(range(H), range(W)):
            w = around[i, j]
            if w > 0:
                rmapx[i, j] = around_rmapx[i, j] / w
                rmapy[i, j] = around_rmapy[i, j] / w
        dt = time.time() - t0
        # logger.info(f'rmap creation took {dt1:.3f} / {dt:.3f} seconds')
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


def write_to_file(rgb, fname):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    compress = cv2.imencode(".jpg", bgr)[1]
    jpg_data = np.array(compress).tostring()
    with open(fname, "wb") as f:
        f.write(jpg_data)
