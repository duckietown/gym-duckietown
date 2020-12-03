# coding=utf-8
import math
from ctypes import byref
from functools import lru_cache
from typing import Tuple

import cv2
import numpy as np
import pyglet
import pyglet.image
from PIL import Image
from pyglet import gl
from pyglet.gl import GLubyte

from duckietown_world import get_texture_file
from . import logger


def get_texture(tex_name: str, rng=None, segment: bool = False) -> "Texture":
    paths = get_texture_file(tex_name)

    if rng:
        path_idx = rng.randint(0, len(paths))
        path = paths[path_idx]
    else:
        path = paths[0]

    oldpath = path
    if segment:
        path += ".SEGMENTED"

    if path not in Texture.tex_cache:
        Texture.tex_cache[path] = Texture(load_texture(oldpath, segment), tex_name=tex_name, rng=rng)

    return Texture.tex_cache[path]


class Texture:
    """
    Manage the caching of textures, and texture randomization
    """

    # Cache of textures
    tex_cache = {}

    def __init__(self, tex, tex_name, rng):
        assert not isinstance(tex, str)
        self.tex = tex
        self.tex_name = tex_name
        self.rng = rng

    def bind(self, segment=False):
        if segment:
            self = get_texture(self.tex_name, self.rng, True)

        gl.glBindTexture(self.tex.target, self.tex.id)


def should_segment_out(tex_path):
    for yes in ["sign", "trafficlight", "asphalt"]:
        if yes in tex_path:
            return True
    for no in ["left", "right", "way", "curve", "straight"]:
        if no in tex_path:
            return False
    return True


@lru_cache(maxsize=None)
def load_texture(tex_path: str, segment: bool = False, segment_into_color=None):
    """ segment_into_black controls what type of segmentation we apply: for tiles and all ground textures,
    replacing
    unimportant stuff with black is a good idea. For other things, replacing it with transparency is good too
    (for example, we don't want black traffic lights, because they go over the roads, and they'd cut our
    view of
    things).
    """
    if segment_into_color is None:
        segment_into_color = [0, 0, 0]
    logger.debug(f"loading texture: {tex_path}")
    img = pyglet.image.load(tex_path)
    # img_format = 'RGBA'
    # pitch = img.width * len(img_format)
    # pixels = img.get_data(img_format, pitch)
    #
    #
    # for i in range(x, width):
    #     for j in range(y, height):
    #         pixels[i, j] = (0, 0, 0, 0)

    if segment:
        if should_segment_out(tex_path):  # replace all by 'segment_into_color'
            # https://gamedev.stackexchange.com/questions/55945/how-to-draw-image-in-memory-manually-in-pyglet
            to_fill = np.ones((img.height, img.width), dtype=int)
            to_fill = np.kron(to_fill, np.array(segment_into_color, dtype=int))
            to_fill = list(to_fill.flatten())
            rawData = (GLubyte * len(to_fill))(*to_fill)
            img = pyglet.image.ImageData(img.width, img.height, "RGB", rawData)
        else:  # replace asphalt by black
            # https://gist.github.com/nkymut/1cb40ea6ae4de0cf9ded7332f1ca0d55

            im = cv2.imread(tex_path, cv2.IMREAD_UNCHANGED)

            # remove the red text
            im_for_text_filter = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            where_theres_red = im_for_text_filter[:,:,0] >= 1
            where_theres_green = np.logical_and(im_for_text_filter[:,:,1] >= 1, im_for_text_filter[:,:,1] <= 70)
            where_theres_blue = np.logical_and(im_for_text_filter[:, :, 2] >= 1, im_for_text_filter[:, :, 2] <= 70)
            red_text = np.logical_and(np.logical_and(where_theres_blue, where_theres_green), where_theres_red)
            im[red_text] = 0

            """
            If we use plain colored textures, this is not needed. Kept in a comment in case we want to go back
            to non-plain-colored tiles.
            
            This is very slow computationally, so if we can avoid it, we should.
            
            hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

            lower = np.array([0, 0, 0], dtype="uint8")
            upper = np.array([179, 100, 160], dtype="uint8")
            mask = cv2.inRange(hsv, lower, upper)
            mask = cv2.bitwise_not(mask)

            kernel1 = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], np.uint8)
            kernel2 = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], np.uint8)
            hitormiss1 = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel1)
            hitormiss2 = cv2.morphologyEx(hitormiss1, cv2.MORPH_ERODE, kernel2)
            mask = cv2.bitwise_and(hitormiss1, hitormiss2)

            result = cv2.bitwise_and(hsv, hsv, mask=mask)
            im = cv2.cvtColor(result, cv2.COLOR_HSV2BGR)
            """
            im = cv2.cvtColor(im, cv2.COLOR_BGRA2BGR)   # if you uncomment the big comment above, remove this line

            rows, cols, channels = im.shape

            raw_img = Image.fromarray(im).tobytes()

            top_to_bottom_flag = -1
            bytes_per_row = channels * cols
            img = pyglet.image.ImageData(
                width=cols, height=rows, format="BGR", data=raw_img, pitch=top_to_bottom_flag * bytes_per_row
            )

    tex = img.get_texture()
    # if img.width == img.height:
    #     tex = tex.get_mipmapped_texture()
    gl.glEnable(tex.target)
    gl.glBindTexture(tex.target, tex.id)
    rawimage = img.get_image_data()

    if tex_path.endswith("jpg"):
        image_data = rawimage.get_data("RGB", img.width * 3)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            img.width,
            img.height,
            0,
            gl.GL_RGB,
            gl.GL_UNSIGNED_BYTE,
            image_data,
        )

    else:
        image_data = rawimage.get_data("RGBA", img.width * 4)

        gl.glTexImage2D(
            gl.GL_TEXTURE_2D,
            0,
            gl.GL_RGBA,
            img.width,
            img.height,
            0,
            gl.GL_RGBA,
            gl.GL_UNSIGNED_BYTE,
            image_data,
        )

    return tex


def create_frame_buffers(width: int, height: int, num_samples: int) -> Tuple[int, int]:
    """Create the frame buffer objects"""

    # Create a frame buffer (rendering target)
    multi_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(multi_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, multi_fbo)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
    # support multisampling on frame buffer objects
    # noinspection PyBroadException
    try:
        # Create a multisampled texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures(1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D_MULTISAMPLE, fbTex)
        gl.glTexImage2DMultisample(
            gl.GL_TEXTURE_2D_MULTISAMPLE, num_samples, gl.GL_RGBA32F, width, height, True
        )
        gl.glFramebufferTexture2D(
            gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D_MULTISAMPLE, fbTex, 0
        )

        # Attach a multisampled depth buffer to the FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorageMultisample(
            gl.GL_RENDERBUFFER, num_samples, gl.GL_DEPTH_COMPONENT, width, height
        )
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    except BaseException as e:
        # logger.warning(e=traceback.format_exc())
        logger.debug("Falling back to non-multisampled frame buffer")

        # Create a plain texture texture to render into
        fbTex = gl.GLuint(0)
        gl.glGenTextures(1, byref(fbTex))
        gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbTex, 0)

        # Attach depth buffer to FBO
        depth_rb = gl.GLuint(0)
        gl.glGenRenderbuffers(1, byref(depth_rb))
        gl.glBindRenderbuffer(gl.GL_RENDERBUFFER, depth_rb)
        gl.glRenderbufferStorage(gl.GL_RENDERBUFFER, gl.GL_DEPTH_COMPONENT, width, height)
        gl.glFramebufferRenderbuffer(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_RENDERBUFFER, depth_rb)

    # Sanity check

    if pyglet.options["debug_gl"]:
        res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Create the frame buffer used to resolve the final render
    final_fbo = gl.GLuint(0)
    gl.glGenFramebuffers(1, byref(final_fbo))
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, final_fbo)

    # Create the texture used to resolve the final render
    fbTex = gl.GLuint(0)
    gl.glGenTextures(1, byref(fbTex))
    gl.glBindTexture(gl.GL_TEXTURE_2D, fbTex)
    gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA, width, height, 0, gl.GL_RGBA, gl.GL_FLOAT, None)
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, fbTex, 0)

    if pyglet.options["debug_gl"]:
        res = gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER)
        assert res == gl.GL_FRAMEBUFFER_COMPLETE

    # Enable depth testing
    gl.glEnable(gl.GL_DEPTH_TEST)

    # Unbind the frame buffer
    gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    return multi_fbo, final_fbo


def rotate_point(px, py, cx, cy, theta):
    """
    Rotate a 2D point around a center
    """

    dx = px - cx
    dy = py - cy

    new_dx = dx * math.cos(theta) + dy * math.sin(theta)
    new_dy = dy * math.cos(theta) - dx * math.sin(theta)

    return cx + new_dx, cy + new_dy


def gen_rot_matrix(axis0: np.ndarray, angle: float) -> np.ndarray:
    """
    Rotation matrix for a counterclockwise rotation around the given axis
    """

    axis = axis0 / math.sqrt(np.dot(axis0, axis0))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array(
        [
            [a * a + b * b - c * c - d * d, 2 * (b * c - a * d), 2 * (b * d + a * c)],
            [2 * (b * c + a * d), a * a + c * c - b * b - d * d, 2 * (c * d - a * b)],
            [2 * (b * d - a * c), 2 * (c * d + a * b), a * a + d * d - b * b - c * c],
        ]
    )


def bezier_point(cps, t):
    """
    Cubic Bezier curve interpolation
    B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    """

    p = ((1 - t) ** 3) * cps[0, :]
    p += 3 * t * ((1 - t) ** 2) * cps[1, :]
    p += 3 * (t ** 2) * (1 - t) * cps[2, :]
    p += (t ** 3) * cps[3, :]

    return p


def bezier_tangent(cps, t):
    """
    Tangent of a cubic Bezier curve (first order derivative)
    B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
    """

    p = 3 * ((1 - t) ** 2) * (cps[1, :] - cps[0, :])
    p += 6 * (1 - t) * t * (cps[2, :] - cps[1, :])
    p += 3 * (t ** 2) * (cps[3, :] - cps[2, :])

    norm = np.linalg.norm(p)
    p /= norm

    return p


def bezier_closest(cps, p, t_bot=0, t_top=1, n=8):
    mid = (t_bot + t_top) * 0.5

    if n == 0:
        return mid

    p_bot = bezier_point(cps, t_bot)
    p_top = bezier_point(cps, t_top)

    d_bot = np.linalg.norm(p_bot - p)
    d_top = np.linalg.norm(p_top - p)

    if d_bot < d_top:
        # noinspection PyTypeChecker
        return bezier_closest(cps, p, t_bot, mid, n - 1)

    # noinspection PyTypeChecker
    return bezier_closest(cps, p, mid, t_top, n - 1)


def bezier_draw(cps, n=20, red=False):
    pts = [bezier_point(cps, i / (n - 1)) for i in range(0, n)]
    gl.glBegin(gl.GL_LINE_STRIP)

    if red:
        gl.glColor3f(1, 0, 0)
    else:
        gl.glColor3f(0, 0, 1)

    for i, p in enumerate(pts):
        gl.glVertex3f(*p)

    gl.glEnd()
    gl.glColor3f(1, 1, 1)
