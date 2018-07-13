#!/usr/bin/env python3

"""
This script allows you to manually control the simulator or Duckiebot
using a Logitech Game Controller, as well as record trajectories.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
import gym
import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper
import json

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='SimpleSim-v0')
parser.add_argument('--map-name', default='udem1')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', action='store_true', help='enable domain randomization')
parser.add_argument('--save-all', action='store_true', help='Write all trajectories rather than just the most recent one to the file', default=False)
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

env = HeadingWrapper(env)

env.reset()
env.render()

# global variables for recording
positions = []
actions = []
demos = []
recording = False


def write_to_file(demos):
    # Store the trajectories in a JSON file
    # Only add the last demo
    print('lenDemos2', len(demos))
    if args.save_all:
        with open('experiments/demos_{}.json'.format(args.map_name), 'w') as outfile:
            json.dump({ 'demos': demos }, outfile)
    else:
        with open('experiments/demos_{}.json'.format(args.map_name), 'w') as outfile:
            json.dump({ 'demos': demos }, outfile)


def process_recording():
    global positions, actions, demos

    if len(positions) == 0 and args.save_all:
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
    if args.save_all:
        write_to_file(demos)

    # Write only last recorded demo
    else:
        write_to_file(demo)

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
    if button == 1 and not recording: 
        print('Start Recording, Press A again to Finish')
        recording = True
        positions.append((env.unwrapped.cur_pos, env.unwrapped.cur_angle))       

    # B Button
    elif button == 1 and recording:
        recording = False
        process_recording()
        print('Saved Recording')

        positions = []
        actions = []

    # X Button
    elif button == 0 and not recording:
        positions = []
        actions = []
        process_recording()

        print('Deleted Last Recording')

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

    # No actions took place
    if round(joystick.x, 2) == 0.0 and round(joystick.y, 2) == 0.0:
        return

    x = round(joystick.y, 2)
    z = round(joystick.x, 2)

    action = np.array([-x, -z])

    if joystick.buttons[5]: # RTrigger, Boost
        action *= 1.5

    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f' % (env.unwrapped.step_count, reward))

    if recording:
        positions.append((env.unwrapped.cur_pos, env.unwrapped.cur_angle))
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
