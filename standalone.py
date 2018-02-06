#!/usr/bin/env python3

from __future__ import division, print_function

import numpy
import gym

import gym_duckietown
from gym_duckietown.envs import DuckiebotEnv
import pyglet

def main():

    env = gym.make('Duckiebot-v0')
    env.reset()

    env.render()
    @env.window.event
    def on_key_press(symbol, modifiers):
        from pyglet.window import key

        action = None
        if symbol == key.LEFT:
            print('left')
            action = numpy.array([-0.15, 0.15])
        elif symbol == key.RIGHT:
            print('right')
            action = numpy.array([0.15, -0.15])
        elif symbol == key.UP:
            print('forward')
            action = numpy.array([0.2, 0.2])
        elif symbol == key.DOWN:
            print('back')
            action = numpy.array([-0.1, -0.1])
        elif symbol == key.SLASH:
            print('RESET')
            action = None
            env.reset()
        elif symbol == key.SPACE:
            action = numpy.array([0, 0])
        else:
            return

        if action is not None:
            print('stepping')
            obs, reward, done, info = env.step(action)
            print('stepped')

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
