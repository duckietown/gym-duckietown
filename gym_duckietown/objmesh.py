import os
import math
import numpy as np
import pyglet
from .graphics import *
from .utils import *

class ObjMesh:
    """
    Load and render OBJ model files
    """

    # Loaded mesh files, indexed by mesh file path
    cache = {}

    @classmethod
    def get(self, mesh_name):
        """
        Load a mesh or used a cached version
        """

        # Assemble the absolute path to the mesh file
        file_path = get_file_path('meshes', mesh_name, 'obj')

        if file_path in self.cache:
            return self.cache[file_path]

        mesh = ObjMesh(file_path)
        self.cache[file_path] = mesh

        return mesh

    def __init__(self, file_path):
        """
        Load an OBJ model file

        Limitations:
        - only one object/group
        - only triangle faces
        """

        # Comments
        # mtllib file_name
        # o object_name
        # v x y z
        # vt u v
        # vn x y z
        # usemtl mtl_name
        # f v0/t0/n0 v1/t1/n1 v2/t2/n2

        print('loading mesh "%s"' % file_path)

        # Attempt to load the materials library
        materials = self._load_mtl(file_path)

        mesh_file = open(file_path, 'r')

        verts = []
        texs = []
        normals = []
        faces = []
        face_mtls = []

        cur_mtl = None

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

            if prefix == 'usemtl':
                mtl_name = tokens[0]
                cur_mtl = materials[mtl_name] if mtl_name in materials else None

            if prefix == 'f':
                assert len(tokens) == 3, "only triangle faces are supported"

                face = []
                for token in tokens:
                    indices = filter(lambda t: t != '', token.split('/'))
                    indices = list(map(lambda idx: int(idx), indices))
                    assert len(indices) == 2 or len(indices) == 3
                    face.append(indices)

                faces.append(face)
                face_mtls.append(cur_mtl)

        mesh_file.close()

        self.num_faces = len(faces)

        print('num verts=%d' % len(verts))
        print('num_faces=%d' % self.num_faces)

        # Create numpy arrays to store the vertex data
        list_verts = np.zeros(shape=(3 * self.num_faces, 3), dtype=np.float32)
        list_norms = np.zeros(shape=(3 * self.num_faces, 3), dtype=np.float32)
        list_texcs = np.zeros(shape=(3 * self.num_faces, 2), dtype=np.float32)
        list_color = np.zeros(shape=(3 * self.num_faces, 3), dtype=np.float32)

        cur_vert_idx = 0

        # For each triangle
        for f_idx, face in enumerate(faces):
            # Get the color for this face
            f_mtl = face_mtls[f_idx]
            f_color = f_mtl['Kd'] if f_mtl else np.array((1,1,1))

            # For each tuple of indices
            for indices in face:
                # Note: OBJ uses 1-based indexing
                # and texture coordinates are optional
                if len(indices) == 3:
                    v_idx, t_idx, n_idx = indices
                    vert = verts[v_idx-1]
                    texc = texs[t_idx-1]
                    normal = normals[n_idx-1]
                else:
                    v_idx, n_idx = indices
                    vert = verts[v_idx-1]
                    normal = normals[n_idx-1]
                    texc = [0, 0]

                list_verts[cur_vert_idx, :] = vert
                list_texcs[cur_vert_idx, :] = texc
                list_norms[cur_vert_idx, :] = normal
                list_color[cur_vert_idx, :] = f_color

                # Move to the next vertex
                cur_vert_idx += 1

        # Re-center the object so that y=0 is at the base,
        # and the object is centered in x and z
        x_coords = list_verts[:, 0]
        z_coords = list_verts[:, 2]
        min_y = list_verts[:, 1].min()
        mean_x = (x_coords.min() + x_coords.max()) / 2
        mean_z = (z_coords.min() + z_coords.max()) / 2
        list_verts[:, 1] -= min_y
        list_verts[:, 0] -= mean_x
        list_verts[:, 2] -= mean_z

        # Compute the object extents after centering
        x_coords = list_verts[:, 0]
        y_coords = list_verts[:, 1]
        z_coords = list_verts[:, 2]
        self.y_max = y_coords.max()

        # Create a vertex list to be used for rendering
        self.vlist = pyglet.graphics.vertex_list(
            3 * self.num_faces,
            ('v3f', list_verts.reshape(-1)),
            ('t2f', list_texcs.reshape(-1)),
            ('n3f', list_norms.reshape(-1)),
            ('c3f', list_color.reshape(-1))
        )

        # Load the texture associated with this mesh
        file_name = os.path.split(file_path)[-1]
        tex_name = file_name.split('.')[0]
        tex_path = get_file_path('textures', tex_name, 'png')

        # Try to load the texture, if it exists
        if os.path.exists(tex_path):
            self.texture = load_texture(tex_path)
        else:
            self.texture = None

    def _load_mtl(self, model_path):
        mtl_path = model_path.split('.')[0] + '.mtl'

        if not os.path.exists(mtl_path):
            return {}

        print('loading materials from "%s"' % mtl_path)

        mtl_file = open(mtl_path, 'r')

        materials = {}
        cur_mtl = None

        # For each line of the input file
        for line in mtl_file:
            line = line.rstrip(' \r\n')

            # Skip comments
            if line.startswith('#') or line == '':
                continue

            tokens = line.split(' ')
            tokens = map(lambda t: t.strip(' '), tokens)
            tokens = list(filter(lambda t: t != '', tokens))

            prefix = tokens[0]
            tokens = tokens[1:]

            if prefix == 'newmtl':
                cur_mtl = {}
                materials[tokens[0]] = cur_mtl

            if prefix == 'Kd':
                vals = list(map(lambda v: float(v), tokens))
                vals = np.array(vals)
                cur_mtl['Kd'] = vals

        mtl_file.close()

        return materials

    def render(self):
        if self.texture:
            glEnable(GL_TEXTURE_2D)
            glBindTexture(self.texture.target, self.texture.id)
        else:
            glDisable(GL_TEXTURE_2D)

        self.vlist.draw(GL_TRIANGLES)

        glDisable(GL_TEXTURE_2D)
