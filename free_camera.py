#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""
import numpy as np

import argparse
import sys

import gym
from pyglet import app, clock
from pyglet.window import key

from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.simulator import Simulator

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

assert isinstance(env.unwrapped, Simulator)


@env.unwrapped.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        env.reset()
        env.render()
        return
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Camera movement
    env.unwrapped.cam_offset = env.unwrapped.cam_offset.astype('float64')
    cam_offset, cam_angle = env.unwrapped.cam_offset, env.unwrapped.cam_angle
    if symbol == key.W:
        cam_angle[0] -= 5
    elif symbol == key.S:
        cam_angle[0] += 5
    elif symbol == key.A:
        cam_angle[1] -= 5
    elif symbol == key.D:
        cam_angle[1] += 5
    elif symbol == key.Q:
        cam_angle[2] -= 5
    elif symbol == key.E:
        cam_angle[2] += 5
    elif symbol == key.UP:
        cam_offset[0] = cam_offset[0]+0.1
    elif symbol == key.DOWN:
        cam_offset[0]  = cam_offset[0] - .1
    elif symbol == key.LEFT:
        cam_offset[2] -= 0.1
    elif symbol == key.RIGHT:
        cam_offset[2] += .1
    elif symbol == key.O:
        cam_offset[1] += .1
    elif symbol == key.P:
        cam_offset[1] -= .1

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage depencency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     try:
    #         from experiments.utils import save_img
    #         save_img('screenshot.png', img)
    #     except BaseException as e:
    #         print(str(e))


def update(dt):
    env.render('free_cam')


# Main event loop
clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)
app.run()

env.close()
