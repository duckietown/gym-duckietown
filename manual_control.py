#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='SimpleSim-v0')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
args = parser.parse_args()

if args.env_name == 'SimpleSim-v0':
    env = SimpleSimEnv(
        map_name = args.map_name,
        draw_curve = args.draw_curve,
        draw_bbox = args.draw_bbox,
        domain_rand = args.domain_rand
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

@env.window.event
def on_key_press(symbol, modifiers):
    """
    This handler processes keyboard commands that
    control the simulation
    """

    if symbol == key.BACKSPACE or symbol == key.SLASH:
        print('RESET')
        env.reset()
        env.render()
    elif symbol == key.PAGEUP:
        env.cam_angle = 0
        env.render()
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.window.push_handlers(key_handler)

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    action = None

    if key_handler[key.UP]:
        action = np.array([0.70, 0.70])
    if key_handler[key.DOWN]:
        action = np.array([-0.40, -0.40])
    if key_handler[key.LEFT]:
        action = np.array([0.00, 0.40])
    if key_handler[key.RIGHT]:
        action = np.array([0.40, 0.00])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    if action is not None:
        # Speed boost
        if key_handler[key.LSHIFT]:
            action *= 1.5

        obs, reward, done, info = env.step(action)
        print('step_count = %s, reward=%.3f' % (env.step_count, reward))

        if done:
            print('done!')
            env.reset()
            env.render()

    env.render()

pyglet.clock.schedule_interval(update, 0.1)

# Enter main event loop
pyglet.app.run()

env.close()
