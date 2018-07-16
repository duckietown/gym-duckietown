import numpy as np
from ..collision import *

import pyglet
from pyglet.gl import *

class WorldObj:
    def __init__(self, obj, domain_rand, draw_bbox):
        self.obj = obj
        self.domain_rand = domain_rand
        self.draw_bbox = draw_bbox
        self.angle = self.obj['y_rot'] * (math.pi / 180)
        self.generate_geometry()

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, item):
        self.obj[key] = item

    def generate_geometry(self):
        # Find corners and normal vectors assoc w. object
        self.obj_corners = generate_corners(self.obj['pos'], 
            self.obj['min_coords'], self.obj['max_coords'], self.angle, self.obj['scale'])
        self.obj_norm = generate_norm(self.obj_corners)

    def render(self):
        if not self.obj['visible']:
            return

        # Draw the bounding box
        if self.draw_bbox:
            glColor3f(1, 0, 0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(self.obj_corners.T[0, 0], 0.01, self.obj_corners.T[1, 0])
            glVertex3f(self.obj_corners.T[0, 1], 0.01, self.obj_corners.T[1, 1])
            glVertex3f(self.obj_corners.T[0, 2], 0.01, self.obj_corners.T[1, 2])
            glVertex3f(self.obj_corners.T[0, 3], 0.01, self.obj_corners.T[1, 3])
            glEnd()

        scale = self.obj['scale']
        y_rot = self.obj['y_rot']
        mesh = self.obj['mesh']
        glPushMatrix()
        glTranslatef(*self.obj['pos'])
        glScalef(scale, scale, scale)
        glRotatef(y_rot, 0, 1, 0)
        glColor3f(*self.obj['color'])
        mesh.render()
        glPopMatrix()

    # Below are the functions that need to 
    # be reimplemented for any dynamic object    
    def check_collision(self, agent_corners, agent_norm):
        if not self.obj['static']: 
            raise NotImplementedError
        return False

    def safe_driving(self, agent_pos, agent_safety_rad):
        if not self.obj['static']: 
            raise NotImplementedError
        return 0.0

    def step(self):
        if not self.obj['static']: 
            raise NotImplementedError

    def update_rendering(self):
        if not self.obj['static']: 
            raise NotImplementedError