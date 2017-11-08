#!/usr/bin/env python2

from __future__ import division, print_function

import numpy
import gym

from gym_duckietown.envs import DuckietownEnv
import pyglet

def main():

    env = gym.make('Duckietown-v0')
    env.reset()

    env.render()
    @env.window.event
    def on_key_press(symbol, modifiers):
        from pyglet.window import key

        action = None
        if symbol == key.LEFT:
            print('left')
            action = numpy.array([-1, 1])
        elif symbol == key.RIGHT:
            print('right')
            action = numpy.array([1, -1])
        elif symbol == key.UP:
            print('forward')
            action = numpy.array([1, 1])
        elif symbol == key.DOWN:
            print('back')
            action = numpy.array([-1, -1])
        elif symbol == key.SLASH:
            print('RESET')
            action = None
            env.reset()
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

if __name__ == "__main__":
    main()
