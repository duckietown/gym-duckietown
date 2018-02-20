#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import argparse

import numpy
import gym
import gym_duckietown
import pyglet

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckie-SimpleSim-v0')
args = parser.parse_args()

env = gym.make(args.env_name)

env.reset()
env.render()

@env.window.event
def on_key_press(symbol, modifiers):
    from pyglet.window import key

    action = None
    if symbol == key.LEFT:
        print('left')
        action = numpy.array([0.00, 0.40])
    elif symbol == key.RIGHT:
        print('right')
        action = numpy.array([0.40, 0.00])
    elif symbol == key.UP:
        print('forward')
        action = numpy.array([0.40, 0.40])
    elif symbol == key.SLASH:
        print('RESET')
        action = None
        env.reset()
        env.render()
    elif symbol == key.SPACE:
        action = numpy.array([0, 0])
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)
    else:
        return

    if action is not None:
        print('stepping')
        obs, reward, done, info = env.step(action)

        print('stepCount = %s, reward=%.3f' % (env.stepCount, reward))

        env.render()

        if done:
            print('done!')
            env.reset()
            env.render()

# Enter main event loop
pyglet.app.run()

env.close()
