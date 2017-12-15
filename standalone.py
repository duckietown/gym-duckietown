#!/usr/bin/env python3

from __future__ import division, print_function

import sys
import numpy
import gym

from gym_duckietown.envs import DuckietownEnv, DiscreteEnv
import pyglet

def main():

    env = gym.make('Duckie-SimpleSim-v0')
    env = DiscreteEnv(env)
    env.reset()

    env.render('app')
    window = env.unwrapped.window

    @window.event
    def on_key_press(symbol, modifiers):
        from pyglet.window import key

        action = None
        if symbol == key.LEFT:
            print('left')
            action = 0
        elif symbol == key.RIGHT:
            print('right')
            action = 1
        elif symbol == key.UP:
            print('forward')
            action = 2
        elif symbol == key.SPACE:
            print('RESET')
            env.reset()
            env.render('app')
            return
        elif symbol == key.ESCAPE:
            sys.exit(0)
        else:
            return

        obs, reward, done, info = env.step(action)

        print('stepCount = %s, reward=%.3f' % (env.unwrapped.stepCount, reward))

        if done:
            print('done!')
            env.reset()

        env.render('app')

    @window.event
    def on_close():
        sys.exit(0)

    # Enter main event loop
    pyglet.app.run()

if __name__ == "__main__":
    main()
