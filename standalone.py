#!/usr/bin/env python3

from __future__ import division, print_function

import numpy
import gym

from gym_duckietown.envs import DuckietownEnv, DiscreteEnv
import pyglet

def main():

    env = gym.make('Duckie-SimpleSim-v0')
    env = DiscreteEnv(env)
    env.reset()

    env.render('app')
    @env.unwrapped.window.event
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
        #elif symbol == key.DOWN:
        #    print('back')
        #    action = numpy.array([-1, -1])
        elif symbol == key.SPACE:
            print('RESET')
            action = None
            env.reset()
        else:
            return

        if action is not None:
            print('stepping')
            obs, reward, done, info = env.step(action)

            print('stepCount = %s, reward=%.3f' % (env.unwrapped.stepCount, reward))

            if done:
                print('done!')
                env.reset()

        env.render('app')

    # Enter main event loop
    pyglet.app.run()

if __name__ == "__main__":
    main()
