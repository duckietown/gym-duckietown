from .graphics import rotate_point
import numpy as np

def duckie_boundbox(cur_pos, true_pos, theta, width, length):
    """
    Compute bounding box for duckie using its dimensions,
    current position, and angle of rotation
    """

    # halfwidth/length, posx/y
    hwidth = 0.5 * width
    hlength = 0.5 * length
    px = cur_pos[0]
    pz_rot = cur_pos[2]

    # Corners of the bounding box
    return np.array([
        rotate_point(px-hwidth, true_pos[2]-hlength, px, pz_rot, theta),
        rotate_point(px+hwidth, true_pos[2]-hlength, px, pz_rot, theta),
        rotate_point(px+hwidth, true_pos[2]+hlength, px, pz_rot, theta),
        rotate_point(px-hwidth, true_pos[2]+hlength, px, pz_rot, theta),
    ])

def tensor_sat_test(norm, corners):
    """
    Separating Axis Theorem (SAT) extended to >2D.
    Requires that both the inputs are stacked on axis 0.
    (each input ~ "a list of 2D matrices" = 3D Tensor)
    """
    dotval = np.matmul(norm, corners)
    return np.min(dotval, axis=-1), np.max(dotval, axis=-1)

def overlaps(min1, max1, min2, max2):
    """
    Helper function to check projection intervals (SAT)
    """
    return is_between_ordered(min2, min1, max1) or is_between_ordered(min1, min2, max2)

def is_between_ordered(val, lowerbound, upperbound):
    """
    Helper function to check projection intervals (SAT)
    """
    return lowerbound <= val and val <= upperbound

def generate_corners(pos, min_coords, max_coords, theta, scale):
    """
    Generates corners given obj pos, extents, scale, and rotation
    """
    px = pos[0]
    pz = pos[-1]
    return np.array([
        rotate_point(min_coords[0]*scale+px, min_coords[-1]*scale+pz, px, pz, theta),
        rotate_point(max_coords[0]*scale+px, min_coords[-1]*scale+pz, px, pz, theta),
        rotate_point(max_coords[0]*scale+px, max_coords[-1]*scale+pz, px, pz, theta),
        rotate_point(min_coords[0]*scale+px, max_coords[-1]*scale+pz, px, pz, theta),
    ])

def generate_norm(corners):
    """
    Generates both (orthogonal, 1per face) normal vectors
    for rectangle given vertices *in a particular order* (see generate_corners)
    """
    width = corners[0] - corners[1]
    length = corners[0] - corners[2]
    return np.array([
      [-1*width[1], width[0]], [length[1], -1*length[0]]
    ])

def intersects(duckie, objs_stacked, duckie_norm, norms_stacked):
    """
    Helper function for Tensor-based OBB intersection.
    Variable naming: SAT requires checking of the projection of all normals
    to all sides, which is where we use tensor_sat_test (gives the mins and maxs)
    of each projection pair. The variables are named as:
    {x's norm + projected on + min/max}.
    """
    duckduck_min, duckduck_max = tensor_sat_test(duckie_norm, duckie.T)
    objduck_min, objduck_max = tensor_sat_test(duckie_norm, objs_stacked)
    duckobj_min, duckobj_max = tensor_sat_test(norms_stacked, duckie.T)
    objobj_min, objobj_max = tensor_sat_test(norms_stacked, objs_stacked)

    # Iterate through each object we are checking against
    for idx in range(objduck_min.shape[0]):
        # If any interval doesn't overlap, immediately know objects don't intersect
        if not overlaps(
            duckduck_min[0], duckduck_max[0], objduck_min[idx][0], objduck_max[idx][0]):
            continue
        if not overlaps(
            duckduck_min[1], duckduck_max[1], objduck_min[idx][1], objduck_max[idx][1]):
            continue
        if not overlaps(
            duckobj_min[idx][0], duckobj_max[idx][0], objobj_min[idx][0], objobj_max[idx][0]):
            continue
        if not overlaps(
            duckobj_min[idx][1], duckobj_max[idx][1], objobj_min[idx][1], objobj_max[idx][1]):
            continue
        # All projection intervals overlap, collision with an object
        return True

    return False
