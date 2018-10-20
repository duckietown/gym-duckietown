#!/usr/bin/env python

"""
This script allows you to manually control the simulator or Duckiebot
using a Logitech Game Controller, as well as record trajectories.
"""

import sys
import argparse
import math
import json
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default=None)
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
args = parser.parse_args()

if args.env_name is None:
    env = DuckietownEnv(
        map_name = args.map_name,
        domain_rand = args.domain_rand,
        max_steps = np.inf
    )
else:
    env = gym.make(args.env_name)

env.reset()
env.render()

# global variables for demo recording
positions = []
actions = []
demos = []
recording = False

def write_to_file(demos):
    num_steps = 0
    for demo in demos:
        num_steps += len(demo['actions'])
    print('num demos:', len(demos))
    print('num steps:', num_steps)

    # Store the trajectories in a JSON file
    with open('experiments/demos_{}.json'.format(args.map_name), 'w') as outfile:
        json.dump({ 'demos': demos }, outfile)

def process_recording():
    global positions, actions, demos

    if len(positions) == 0:
        # Nothing to delete
        if len(demos) == 0:
            return

        # Remove the last recorded demo
        demos.pop()
        write_to_file(demos)
        return

    p = list(map(lambda p: [ p[0].tolist(), p[1] ], positions))
    a = list(map(lambda a: a.tolist(), actions))

    demo = {
        'positions': p,
        'actions': a
    }

    demos.append(demo)

    # Write all demos to this moment
    write_to_file(demos)

@env.unwrapped.window.event
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
        env.unwrapped.cam_angle[0] = 0
        env.render()
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

@env.unwrapped.window.event
def on_joybutton_press(joystick, button):
    """
    Event Handler for Controller Button Inputs
    Relevant Button Definitions:
    1 - A - Starts / Stops Recording
    0 - X - Deletes last Recording
    2 - Y - Resets Env.

    Triggers on button presses to control recording capabilities
    """
    global recording, positions, actions

    # A Button
    if button == 1:
        if not recording:
            print('Start recording, Press A again to finish')
            recording = True
        else:
            recording = False
            process_recording()
            positions = []
            actions = []
            print('Saved recording')

    # X Button
    elif button == 0:
        recording = False
        positions = []
        actions = []
        process_recording()
        print('Deleted last recording')

    # Y Button
    elif button == 3:
        print('RESET')
        env.reset()
        env.render()

    # Any other button thats not boost prints help
    elif button != 5:
        helpstr1 = "A - Starts / Stops Recording\nX - Deletes last Recording\n"
        helpstr2 = "Y - Resets Env.\nRB - Hold for Boost"

        print("Help:\n{}{}".format(helpstr1, helpstr2))

def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """
    global recording, positions, actions

    # No actions took place
    if round(joystick.x, 2) == 0.0 and round(joystick.y, 2) == 0.0:
        return

    x = round(joystick.y, 2)
    z = round(joystick.x, 2)

    action = np.array([-x, -z])

    # Right trigger, speed boost
    if joystick.buttons[5]:
        action *= 1.5

    if recording:
        positions.append((env.unwrapped.cur_pos, env.unwrapped.cur_angle))
        actions.append(action)

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if done:
        print('done!')
        env.reset()
        env.render()

        if recording:
            process_recording()
            positions = []
            actions = []
            print('Saved Recoding')

    env.render()

pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Registers joysticks and recording controls
joysticks = pyglet.input.get_joysticks()
assert joysticks, 'No joystick device is connected'
joystick = joysticks[0]
joystick.open()
joystick.push_handlers(on_joybutton_press)

# Enter main event loop
pyglet.app.run()

env.close()
