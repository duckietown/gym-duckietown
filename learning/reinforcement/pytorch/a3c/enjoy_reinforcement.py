import ast
import argparse
import logging
import os
import numpy as np

import torch

# Duckietown Specific
import gym
from learning.reinforcement.pytorch.a3c import a3c_continuous_simple as a3c
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, ActionClampWrapper


def _enjoy(args):
    env = gym.make('Duckietown-udem1-v0')
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    env = ActionWrapper(env)
    env = ActionClampWrapper(env)  # clip to range [-1, 1]
    #env = PreventBackwardsWrapper(env)  # prevent duckiebot driving backwards..
    env = DtRewardWrapper(env)

    shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    shape_action_space = env.action_space.shape[0]  # (2,)

    # Initialize policy

    # Load model
    cwd = os.getcwd()
    path = os.path.join(cwd, args.model_dir, args.model_file)
    print('Loading model from:', path)

    global_net = torch.load(path)
    global_net.eval()

    #global_net.load_state_dict(state_dict.Net)

    obs = env.reset()
    done = False

    while True:
        while not done:
            action = global_net.choose_action(np.array(obs))
            # Perform action
            obs, reward, done, _ = env.step(action)
            env.render()
        done = False
        obs = env.reset()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument('--model-file', type=str, default='2019-06-10_19-28-47_a3c-cont.pth')  # Name of the model file
    _enjoy(parser.parse_args())
