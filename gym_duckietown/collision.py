from .graphics import rotate_point
import numpy as np
import math

def agent_boundbox(true_pos, width, length, f_vec, r_vec):
    """
    Compute bounding box for agent using its dimensions,
    current position, and angle of rotation
    Order of points in bounding box:
    (front)
    4 - 3    
    |   |
    1 - 2
    """

    # halfwidth/length
    hwidth = 0.5 * width
    hlength = 0.5 * length

    # Indexing to make sure we only get the x/z dims
    corners = np.array([
        true_pos - hwidth*r_vec - hlength * f_vec,
        true_pos + hwidth*r_vec - hlength * f_vec,
        true_pos + hwidth*r_vec + hlength * f_vec,
        true_pos - hwidth*r_vec + hlength * f_vec
    ])[:, [0, 2]]

    return corners

def tensor_sat_test(norm, corners):
    """
    Separating Axis Theorem (SAT) extended to >2D.
    Requires that both the inputs are stacked on axis 0.
    (each input ~ "a list of 2D matrices" = 3D Tensor)
    """
    dotval = np.matmul(norm, corners)
    mins = np.min(dotval, axis=-1)
    maxs = np.max(dotval, axis=-1)

    return mins, maxs

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

def tile_corners(pos, width):
    """
    Generates the absolute corner coord for a tile, given grid pos and tile width
    """
    px = pos[0]
    pz = pos[-1]

    return np.array([
        [px*width-width, pz*width-width],
        [px*width+width, pz*width-width],
        [px*width+width, pz*width+width],
        [px*width-width, pz*width+width]
    ])

def generate_norm(corners):
    """
    Generates both (orthogonal, 1 per axis) normal vectors
    for rectangle given vertices *in a particular order* (see generate_corners)
    """
    ca = np.cov(corners, y = None, rowvar = 0, bias = 1)
    _, vect = np.linalg.eig(ca)
    return vect.T

def find_candidate_tiles(pos, mesh, angle, scale, tile_size):
    """
    Finds all of the tiles that a object could intersect with
    Returns the norms and corners of any of those that are drivable
    """

    # Find corners and normal vectors assoc w. object
    obj_corners = generate_corners(pos, mesh.min_coords, mesh.max_coords, angle, scale)
    obj_norm = generate_norm(obj_corners)

    # Find min / max x&y tile coordinates of object
    minx, miny = np.floor(
        np.amin(obj_corners, axis=0) / tile_size
    ).astype(int)

    maxx, maxy = np.floor(
        np.amax(obj_corners, axis=0) / tile_size
    ).astype(int)

    # The max number of tiles we need to check is every possible
    # combination of x and y within the ranges, so enumerate
    xr = list(range(minx, maxx+1))
    yr = list(range(miny, maxy+1))

    possible_tiles = np.array([(x, y) for x in xr for y in yr])
    return obj_corners, obj_norm, possible_tiles

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

def safety_circle_intersection(d, r1, r2):
    """
    Checks if  two circles with centers separated by d and centered
    at r1 and r2 either intesect or are enveloped (one inside of other)
    """
    intersect = np.logical_and(
        np.less_equal(np.power(r1 - r2, 2), np.power(d, 2)),
        np.less_equal(np.power(d, 2), np.power(r1 + r2, 2))
    )

    enveloped = np.less(d, abs(r1 - r2))

    return np.any(intersect) or np.any(enveloped)

def safety_circle_overlap(d, r1, r2):
    """
    Returns a proxy for area (see issue #24) 
    of two circles with centers separated by d 
    and centered at r1 and r2
    """
    scores = d - r1 - r2
    return np.sum(scores[np.where(scores < 0)])

def calculate_safety_radius(mesh, scale):
    """
    Returns a safety radius for an object, and scales
    it according to the YAML file's scale param for that obj
    """
    x, _, z = np.max([abs(mesh.min_coords), abs(mesh.max_coords)], axis=0)
    return np.linalg.norm([x, z]) * scale 

def heading_vec(angle):
    """
    Vector pointing in the direction the agent is looking
    """

    x = math.cos(angle)
    z = -math.sin(angle)
    return np.array([x, 0, z])
