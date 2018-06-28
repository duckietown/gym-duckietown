from .graphics import rotate_point
import numpy as np
import time 

def distance(a, b):
  return np.linalg.norm(a-b)

def duckie_boundbox(cur_pos, theta, width, length):
  hwidth = 0.5 * width
  hlength = 0.5 * length
  px = cur_pos[0]
  pz = cur_pos[2]

  return np.array([
    rotate_point(px-hwidth, pz-hlength, px, pz, theta),
    rotate_point(px+hwidth, pz-hlength, px, pz, theta),
    rotate_point(px+hwidth, pz+hlength, px, pz, theta),
    rotate_point(px-hwidth, pz+hlength, px, pz, theta),
  ])

def sat_test(norm, corners):
  dotval = np.matmul(norm, corners.T)
  mins = np.min(dotval, axis=1)
  maxs = np.max(dotval, axis=1)
  return mins[0], maxs[0], mins[1], maxs[1]

def overlaps(min1, max1, min2, max2):
  return is_between_ordered(min2, min1, max1) or is_between_ordered(min1, min2, max2)

def is_between_ordered(val, lowerbound, upperbound):
  return lowerbound <= val and val <= upperbound

def generate_corners(pos, min_coords,max_coords, theta, scale):    
  return np.array([
    rotate_point(min_coords[0]*scale+pos[0], min_coords[-1]*scale+pos[-1], pos[0], pos[-1], theta),
    rotate_point(max_coords[0]*scale+pos[0], min_coords[-1]*scale+pos[-1], pos[0], pos[-1], theta),
    rotate_point(max_coords[0]*scale+pos[0], max_coords[-1]*scale+pos[-1], pos[0], pos[-1], theta),
    rotate_point(min_coords[0]*scale+pos[0], max_coords[-1]*scale+pos[-1], pos[0], pos[-1], theta),
  ])

def generate_norm(corners):
  ca = np.cov(corners,y = None,rowvar = 0,bias = 1)
  _, vect = np.linalg.eig(ca)
  return vect.T

def intersects(corners1, corners2):    
  # norms of each side
  norm1 = generate_norm(corners1)
  norm2 = generate_norm(corners2)
  
  shape1a_min, shape1a_max, shape1b_min, shape1b_max = sat_test(norm1, corners1)
  shape2a_min, shape2a_max, shape2b_min, shape2b_max = sat_test(norm1, corners2)

  if not overlaps(shape1a_min, shape1a_max, shape2a_min, shape2a_max):
    return False

  if not overlaps(shape1b_min, shape1b_max, shape2b_min, shape2b_max):
    return False

  shape1a_min, shape1a_max, shape1b_min, shape1b_max = sat_test(norm2, corners1)
  shape2a_min, shape2a_max, shape2b_min, shape2b_max = sat_test(norm2, corners2)

  if not overlaps(shape1a_min, shape1a_max, shape2a_min, shape2a_max):
    return False

  if not overlaps(shape1b_min, shape1b_max, shape2b_min, shape2b_max):
    return False
  
  return True
