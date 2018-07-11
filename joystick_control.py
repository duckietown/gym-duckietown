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
def on_joybutton_press(joystick, button):
    if button == 12:
        print('RESET')
        env.reset()
        env.render()
    if button == 8:
        print('STOP RECORDING')
    if button == 9:
        print('START RECORDING')
            
def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    action = np.zeros(2)
    if round(joystick.x, 2) == 1.0:
        action[0] = 0.40 
    if round(joystick.x, 2) == -1.0:
        action[0] = -0.40
    if round(joystick.y, 2) == -1.0:
        action += 0.80
    if round(joystick.y, 2) == 1.0:
        action -= 0.80

    if joystick.buttons[5]:
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.step_count, reward))

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 0.1)

joysticks = pyglet.input.get_joysticks()
assert joysticks, 'No joystick device is connected'
joystick = joysticks[0]
joystick.open()
joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()

env.close()
