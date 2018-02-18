#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy

import gym
import gym_duckietown
from gym_duckietown.envs import DuckiebotEnv
import pyglet

def main():
    #env = gym.make('Duckiebot-v0')
    env = gym.make('Duckie-SimpleSim-v0')
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

if __name__ == "__main__":
    main()
