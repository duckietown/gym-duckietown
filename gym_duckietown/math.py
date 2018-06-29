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

def tensor_sat_test(norm, corners):
  dotval = np.matmul(norm, corners)
  return np.min(dotval, axis=-1), np.max(dotval, axis=-1)

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
  width = corners[0] - corners[1]
  length = corners[0] - corners[2]
  return np.array([[-1* width[1], width[0]], [length[1], -1
    *length[0]]])

def intersects(duckie, objs_stacked, duckie_norm, norms_stacked):      
  duckduck_min, duckduck_max = tensor_sat_test(duckie_norm, duckie.T)
  objduck_min, objduck_max = tensor_sat_test(duckie_norm, objs_stacked)
  duckobj_min, duckobj_max = tensor_sat_test(norms_stacked, duckie.T)
  objobj_min, objobj_max = tensor_sat_test(norms_stacked, objs_stacked)

  for idx in range(objduck_min.shape[0]):
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
    return True
    
  return False
