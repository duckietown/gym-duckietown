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
import json

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

# global variables for recording
positions = []
actions = []
demos = []
recording = False

def process_recording():
    global positions, actions, demos

    p = list(map(lambda p: [ p[0].tolist(), p[1] ], positions))
    a = list(map(lambda a: a.tolist(), actions))

    demo = {
        'positions': p,
        'actions': a
    }

    demos.append(demo)

    # Store the trajectories in a JSON file
    with open('experiments/demos_{}.json'.format(args.map_name), 'w') as outfile:
        json.dump({ 'demos': demos }, outfile)

@env.window.event
def on_joybutton_press(joystick, button):
    """
    Event Handler for Controller Button Inputs
    Relevant Button Definitions:
    1 - B
    2 - A
    8 - Select
    9 - Start
    12 - Home

    Triggers on button presses to control recording capabilities
    """
    global recording, positions, actions

    if button == 12: # Home Button
        print('RESET')
        env.reset()
        env.render()

    # Select Button
    if button == 8 and recording:
        recording = False
        print('Stop Recording, Press A to confirm, B to delete')

    # Start Button
    if button == 9 and not recording: 
        print('Start Recording, Press Select to Finish')
        recording = True
        positions.append((env.cur_pos, env.cur_angle))

    # A Button
    if button == 2 and len(positions) != 0 and not recording:
        print('Saved Recording')
        process_recording()
        positions = []
        actions = []

    # B Button
    if button == 1 and len(positions) != 0 and not recording:
        print('Deleted Recording')
        positions = []
        actions = []
            

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    # No actions took place
    if round(joystick.x, 2) == 0.0 and round(joystick.y, 2) == 0.0:
        return

    action = np.zeros(2)
    if round(joystick.x, 2) == 1.0: # RIGHT
        action[0] = 0.40 
    if round(joystick.x, 2) == -1.0: # LEFT
        action[0] = -0.40
    if round(joystick.y, 2) == -1.0: # UP
        action += 0.80
    if round(joystick.y, 2) == 1.0: # DOWN
        action -= 0.80

    if joystick.buttons[5]: # RTrigger, Boost
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.step_count, reward))

    if recording:
        positions.append((env.cur_pos, env.cur_angle))
        actions.append(action)

    if done:
        print('done!')
        env.reset()
        env.render()

    env.render()

pyglet.clock.schedule_interval(update, 0.1)

# Registers joysticks and recording controls
joysticks = pyglet.input.get_joysticks()
assert joysticks, 'No joystick device is connected'
joystick = joysticks[0]
joystick.open()
joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()

env.close()
