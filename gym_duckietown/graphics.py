import math

import os
import numpy as np

import pyglet
from pyglet.gl import *
from ctypes import byref, POINTER

from .utils import *

class Texture:
    """
    Manage the caching of textures, and texture randomization
    """

    # List of textures available for a given path
    tex_paths = {}

    # Cache of textures
    tex_cache = {}

    @classmethod
    def get(self, tex_name, rng=None):
        paths = self.tex_paths.get(tex_name, [])

        # Get an inventory of the existing texture files
        if len(paths) == 0:
            for i in range(1, 10):
                path = get_file_path('textures', '%s_%d' % (tex_name, i), 'png')
                if not os.path.exists(path):
                    break
                paths.append(path)

        assert len(paths) > 0, 'failed to load textures for name "%s"' % tex_name

        if rng:
            path_idx = rng.randint(0, len(paths))
            path = paths[path_idx]
        else:
            path = paths[0]

        if path not in self.tex_cache:
            self.tex_cache[path] = Texture(load_texture(path))

        return self.tex_cache[path]

    def __init__(self, tex):
        assert not isinstance(tex, str)
        self.tex = tex

    def bind(self):
        glBindTexture(self.tex.target, self.tex.id)

def load_texture(tex_path):
    print('loading texture "%s"' % tex_path)

    img = pyglet.image.load(tex_path)
    tex = img.get_texture()
    glEnable(tex.target)
    glBindTexture(tex.target, tex.id)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        img.width,
        img.height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        img.get_image_data().get_data('RGBA', img.width * 4)
    )

    return tex

def create_frame_buffers(width, height, num_samples):
    """Create the frame buffer objects"""

    # Create a frame buffer (rendering target)
    multi_fbo = GLuint(0)
    glGenFramebuffers(1, byref(multi_fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, multi_fbo)

    # The try block here is because some OpenGL drivers
    # (Intel GPU drivers on macbooks in particular) do not
    # support multisampling on frame buffer objects
    try:
        # Create a multisampled texture to render into
        fbTex = GLuint(0)
        glGenTextures( 1, byref(fbTex));
        glBindTexture(GL_TEXTURE_2D_MULTISAMPLE, fbTex);
        glTexImage2DMultisample(
            GL_TEXTURE_2D_MULTISAMPLE,
            num_samples,
            GL_RGBA32F,
            width,
            height,
            True
        );
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D_MULTISAMPLE,
            fbTex,
            0
        );

        # Attach a multisampled depth buffer to the FBO
        depth_rb = GLuint(0)
        glGenRenderbuffers(1, byref(depth_rb))
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
        glRenderbufferStorageMultisample(GL_RENDERBUFFER, num_samples, GL_DEPTH_COMPONENT, width, height);
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);

    except:
        print('Falling back to non-multisampled frame buffer')

        # Create a plain texture texture to render into
        fbTex = GLuint(0)
        glGenTextures( 1, byref(fbTex));
        glBindTexture(GL_TEXTURE_2D, fbTex);
        glTexImage2D(
            GL_TEXTURE_2D,
            0,
            GL_RGBA,
            width,
            height,
            0,
            GL_RGBA,
            GL_FLOAT,
            None
        )
        glFramebufferTexture2D(
            GL_FRAMEBUFFER,
            GL_COLOR_ATTACHMENT0,
            GL_TEXTURE_2D,
            fbTex,
            0
        );

        # Attach depth buffer to FBO
        depth_rb = GLuint(0)
        glGenRenderbuffers(1, byref(depth_rb))
        glBindRenderbuffer(GL_RENDERBUFFER, depth_rb)
        glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT, width, height)
        glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, depth_rb);

    # Sanity check
    if pyglet.options['debug_gl']:
      res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
      assert res == GL_FRAMEBUFFER_COMPLETE

    # Create the frame buffer used to resolve the final render
    final_fbo = GLuint(0)
    glGenFramebuffers(1, byref(final_fbo))
    glBindFramebuffer(GL_FRAMEBUFFER, final_fbo)

    # Create the texture used to resolve the final render
    fbTex = GLuint(0)
    glGenTextures(1, byref(fbTex))
    glBindTexture(GL_TEXTURE_2D, fbTex)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        height,
        0,
        GL_RGBA,
        GL_FLOAT,
        None
    )
    glFramebufferTexture2D(
        GL_FRAMEBUFFER,
        GL_COLOR_ATTACHMENT0,
        GL_TEXTURE_2D,
        fbTex,
        0
    )
    if pyglet.options['debug_gl']:
      res = glCheckFramebufferStatus(GL_FRAMEBUFFER)
      assert res == GL_FRAMEBUFFER_COMPLETE

    # Enable depth testing
    glEnable(GL_DEPTH_TEST)

    # Unbind the frame buffer
    glBindFramebuffer(GL_FRAMEBUFFER, 0)

    return multi_fbo, final_fbo

def rotate_point(px, py, cx, cy, theta):
    """
    Rotate a 2D point around a center
    """

    dx = px - cx
    dy = py - cy

    new_x = dx * math.cos(theta) - dy * math.sin(theta)
    new_x = dy * math.cos(theta) + dx * math.sin(theta)

    return cx + new_x, cy + new_y

def gen_rot_matrix(axis, angle):
    """
    Rotation matrix for a counterclockwise rotation around the given axis
    """

    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(angle / 2.0)
    b, c, d = -axis * math.sin(angle / 2.0)

    return np.array([
        [a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
        [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
        [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]
    ])

def bezier_point(cps, t):
    """
    Cubic Bezier curve interpolation
    B(t) = (1-t)^3 * P0 + 3t(1-t)^2 * P1 + 3t^2(1-t) * P2 + t^3 * P3
    """

    p  = ((1-t)**3) * cps[0,:]
    p += 3 * t * ((1-t)**2) * cps[1,:]
    p += 3 * (t**2) * (1-t) * cps[2,:]
    p += (t**3) * cps[3,:]

    return p

def bezier_tangent(cps, t):
    """
    Tangent of a cubic Bezier curve (first order derivative)
    B'(t) = 3(1-t)^2(P1-P0) + 6(1-t)t(P2-P1) + 3t^2(P3-P2)
    """

    p  = 3 * ((1-t)**2) * (cps[1,:] - cps[0,:])
    p += 6 * (1-t) * t * (cps[2,:] - cps[1,:])
    p += 3 * (t ** 2) * (cps[3,:] - cps[2,:])

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
        return bezier_closest(cps, p, t_bot, mid, n-1)

    return bezier_closest(cps, p, mid, t_top, n-1)

def bezier_draw(cps, n = 20):
    pts = [bezier_point(cps, i/(n-1)) for i in range(0,n)]
    glBegin(GL_LINE_STRIP)
    glColor3f(1, 0, 0)
    for i, p in enumerate(pts):
        glVertex3f(*p)
    glEnd()
    glColor3f(1,1,1)
