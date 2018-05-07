import os
import math
import numpy as np
import pyglet
from .graphics import *

class ObjMesh:
    """
    Load and render OBJ model files
    """

    def __init__(self, mesh_name):
        """
        Load an OBJ model file

        Limitations:
        - only one object/group
        - only triangle faces
        """

        # Comments
        # mtllib mtl_name
        # o object_name
        # v x y z
        # vt u v
        # vn x y z
        # f v0/t0/n0 v1/t1/n1 v2/t2/n2

        # Assemble the absolute path to the mesh file
        abs_path_module = os.path.realpath(__file__)
        module_dir, _ = os.path.split(abs_path_module)
        file_path = os.path.join(module_dir, 'meshes', mesh_name)

        print('loading mesh file "%s"' % file_path)
        mesh_file = open(file_path, 'r')

        verts = []
        texs = []
        normals = []
        faces = []

        # For each line of the input file
        for line in mesh_file:
            line = line.rstrip(' \r\n')

            # Skip comments
            if line.startswith('#') or line == '':
                continue

            tokens = line.split(' ')
            tokens = map(lambda t: t.strip(' '), tokens)
            tokens = list(filter(lambda t: t != '', tokens))

            prefix = tokens[0]
            tokens = tokens[1:]

            if prefix == 'v':
                vert = list(map(lambda v: float(v), tokens))
                verts.append(vert)

            if prefix == 'vt':
                tc = list(map(lambda v: float(v), tokens))
                texs.append(tc)

            if prefix == 'vn':
                normal = list(map(lambda v: float(v), tokens))
                normals.append(normal)

            if prefix == 'f':
                assert len(tokens) == 3, "only triangle faces are supported"

                face = []
                for token in tokens:
                    indices = list(map(lambda idx: int(idx), token.split('/')))
                    face.append(indices)

                faces.append(face)

        mesh_file.close()

        self.num_faces = len(faces)

        print('num verts=%d' % len(verts))
        print('num_faces=%d' % self.num_faces)

        # Create numpy arrays to store the vertex data
        list_verts = np.zeros(shape=3 * 3 * self.num_faces, dtype=np.float32)
        list_texcs = np.zeros(shape=3 * 2 * self.num_faces, dtype=np.float32)
        list_norms = np.zeros(shape=3 * 3 * self.num_faces, dtype=np.float32)

        cur_vert_idx = 0

        # For each triangle
        for face in faces:
            # For each triplet of indices
            for triplet in face:
                v_idx, t_idx, n_idx = triplet

                # Note: OBJ uses 1-based indexing
                vert = verts[v_idx-1]
                texc = texs[t_idx-1]
                normal = normals[n_idx-1]

                list_verts[3*cur_vert_idx:3*(cur_vert_idx+1)] = vert
                list_texcs[2*cur_vert_idx:2*(cur_vert_idx+1)] = texc
                list_norms[3*cur_vert_idx:3*cur_vert_idx+3] = normal

                cur_vert_idx += 1

        # Create a vertex list to be used for rendering
        self.vlist = pyglet.graphics.vertex_list(
            3 * self.num_faces,
            ('v3f', list_verts),
            ('t2f', list_texcs),
            ('n3f', list_norms)
        )

        # Load the texture associated with this mesh
        self.texture = load_texture(mesh_name.replace('obj', 'png'))

    def render(self):
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)

        glBindTexture(self.texture.target, self.texture.id)

        self.vlist.draw(GL_TRIANGLES)

        glDisable(GL_TEXTURE_2D)
