from .graphics import rotate_point
import numpy as np
import time 

def duckie_boundbox(cur_pos, theta, width, length):
  hwidth = 0.5 * width
  hlength = 0.5 * length
  px = cur_pos[0]
  pz = cur_pos[2]

  # TODO: Check dimensions
  return np.array([
    rotate_point(px-hwidth, pz-hlength, px, pz, theta),
    rotate_point(px+hwidth, pz-hlength, px, pz, theta),
    rotate_point(px+hwidth, pz+hlength, px, pz, theta),
    rotate_point(px-hwidth, pz+hlength, px, pz, theta),
  ])

def sat_test(axis, corners):
  min_along = np.inf
  max_along = -np.inf
  
  for c in corners:
    dotval = c.dot(axis)
   
    if dotval < min_along: min_along = dotval
    if dotval > max_along: max_along = dotval   
  
  return min_along, max_along

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
  v, vect = np.linalg.eig(ca)

  return np.transpose(vect)

def intersects(corners1, corners2):    
  # norms of each side
  norm1 = generate_norm(corners1)
  norm2 = generate_norm(corners2)
  
  for norm in norm1:
    shape1_min, shape1_max = sat_test(norm, corners1)
    shape2_min, shape2_max = sat_test(norm, corners2)
  
  if not overlaps(shape1_min, shape1_max, shape2_min, shape2_max):
    return False
  
  for norm in norm2:
    shape1_min, shape1_max = sat_test(norm, corners1)
    shape2_min, shape2_max = sat_test(norm, corners2)
    
  if not overlaps(shape1_min, shape1_max, shape2_min, shape2_max):
    return False
  
  return True
