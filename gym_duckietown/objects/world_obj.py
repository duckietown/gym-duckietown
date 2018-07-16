import numpy as np
from ..collision import *

class WorldObj:
    def __init__(self, obj, domain_rand):
        self.obj = obj
        self.domain_rand = domain_rand
        self.generate_geometry()

    def __getitem__(self, key):
        return self.obj[key]

    def __setitem__(self, key, item):
        self.obj[key] = item

    def generate_geometry(self):
        # Find corners and normal vectors assoc w. object
        self.obj_corners = generate_corners(pos, mesh.min_coords, 
            mesh.max_coords, angle, scale)
        self.obj_norm = generate_norm(obj_corners)

    def render(self):
        if not self.obj['visible']:
            return

        # Draw the bounding box
        if self.draw_bbox:
            glColor3f(1, 0, 0)
            glBegin(GL_LINE_LOOP)
            glVertex3f(self.corners[0, 0], 0.01, self.corners[1, 0])
            glVertex3f(self.corners[0, 1], 0.01, self.corners[1, 1])
            glVertex3f(self.corners[0, 2], 0.01, self.corners[1, 2])
            glVertex3f(self.corners[0, 3], 0.01, self.corners[1, 3])
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
    def check_collision(self):
        if self.obj['static']: 
            return
        else:
            raise NotImplementedError

    def safe_driving(self):
        if self.obj['static']: 
            return 0.
        else:
            raise NotImplementedError

    def step(self):
        if self.obj['static']: 
            return
        else:
            raise NotImplementedError

    def update_corners(self):
        if self.obj['static']: 
            return
        else:
            raise NotImplementedError