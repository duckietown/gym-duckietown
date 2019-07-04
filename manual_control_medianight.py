#!/usr/bin/env python
# manual

"""
This script allows you to manually control the simulator or Duckiebot
using the keyboard arrows.
"""

import sys
import argparse
import pyglet
from pyglet.window import key
import numpy as np
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper
from learning.utils.env import launch_env
import gym
import gym_duckietown
from gym_duckietown.envs import DuckietownEnv
from gym_duckietown.wrappers import UndistortWrapper

# from experiments.utils import save_img

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', default='Duckietown')
parser.add_argument('--map-name', default='medianight')
parser.add_argument('--distortion', default=False, action='store_true')
parser.add_argument('--draw-curve', action='store_true', help='draw the lane following curve')
parser.add_argument('--draw-bbox', action='store_true', help='draw collision detection bounding boxes')
parser.add_argument('--domain-rand', default=False, action='store_true', help='enable domain randomization')
parser.add_argument('--frame-skip', default=1, type=int, help='number of frames to skip')
parser.add_argument('--seed', default=1, type=int, help='seed')
args = parser.parse_args()

env = DuckietownEnv(
    seed=123,  # random seed
    map_name="medianight",
    max_steps=500001,  # we don't want the gym to reset itself
    domain_rand=0,
    camera_width=640,
    camera_height=480,
    accept_start_angle_deg=4,  # start close to straight
    full_transparency=True,
    distortion=False,
    start_angle_deg=-1.5,
    start_pos=np.array([0.77, 0.0, 1.73])
)

# env = ResizeWrapper(env)
# env = NormalizeWrapper(env)
env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
# env = ActionWrapper(env)
env = DtRewardWrapper2(env)
# env = DiscreteWrapper(env)

"""
if args.env_name and args.env_name.find('Duckietown') != -1:
    env = DuckietownEnv(
        seed=args.seed,
        map_name=args.map_name,
        draw_curve=args.draw_curve,
        draw_bbox=args.draw_bbox,
        domain_rand=args.domain_rand,
        frame_skip=args.frame_skip,
        distortion=args.distortion,
    )
else:
    env = gym.make(args.env_name)
"""

env.reset()
env.render()


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
    elif symbol == key.ESCAPE:
        env.close()
        sys.exit(0)

    # Take a screenshot
    # UNCOMMENT IF NEEDED - Skimage dependency
    # elif symbol == key.RETURN:
    #     print('saving screenshot')
    #     img = env.render('rgb_array')
    #     save_img('screenshot.png', img)


# Register a keyboard handler
key_handler = key.KeyStateHandler()
env.unwrapped.window.push_handlers(key_handler)

rewards = 0


def update(dt):
    """
    This function is called at every frame to handle
    movement/stepping and redrawing
    """

    # action = 3

    # if key_handler[key.UP]:
    #    action = 2
    # if key_handler[key.LEFT]:
    #    action = 0
    # if key_handler[key.RIGHT]:
    #    action = 1

    action = np.array([0.0, 0.0])

    if key_handler[key.UP]:
        action = np.array([0.44, 0.0])
    if key_handler[key.DOWN]:
        action = np.array([-0.44, 0])
    if key_handler[key.LEFT]:
        action = np.array([0.35, +1])
    if key_handler[key.RIGHT]:
        action = np.array([0.35, -1])
    if key_handler[key.SPACE]:
        action = np.array([0, 0])

    # Speed boost
    # if key_handler[key.LSHIFT]:
    #    action *= 1.5

    global rewards
    obs, reward, done, info = env.step(action)
    print('step_count = %s, reward=%.3f, epr=%.3f' % (env.unwrapped.step_count, reward, rewards))

    rewards += reward

    if key_handler[key.RETURN]:
        from PIL import Image
        im = Image.fromarray(obs)
        im.save('screen.png')

    if done:
        print('done!')
        print('------------- Episode Rewards:', rewards)
        rewards = 0
        print('------------- New Episode Rewards:', rewards)
        env.reset()

    env.render()


pyglet.clock.schedule_interval(update, 1.0 / env.unwrapped.frame_rate)

# Enter main event loop
pyglet.app.run()

env.close()
