import ast
import argparse
import logging
import os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

# Duckietown Specific
import gym
from learning.reinforcement.pytorch.a3c import a3c_cnn_discrete_gru as a3c
from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, DiscreteWrapper_a6


def preprocess_state(obs):
    from scipy.misc import imresize
    return imresize(obs.mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.


def _enjoy(args):
    env = launch_env()
    # env = ResizeWrapper(env)
    # env = NormalizeWrapper(env)
    env = ImgWrapper(env)  # to make the images from 160x120x3 into 3x160x120
    # env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    env = DiscreteWrapper_a6(env)

    shape_obs_space = env.observation_space.shape  # (3, 120, 160)
    shape_action_space = env.action_space.n  # (2,)

    # Initialize policy

    # Load model
    cwd = os.getcwd()
    path = os.path.join(cwd, args.model_dir, args.model_file)
    print('Loading model from:', path)

    checkpoint = torch.load(path)
    global_net = a3c.Net(channels=1, num_actions=shape_action_space)
    global_net.load_state_dict(checkpoint['model_state_dict'])
    #global_net.load_state_dict(checkpoint)
    global_net.eval()

    state = torch.tensor(preprocess_state(env.reset()))
    done = True

    while True:
        with torch.no_grad():
            if done:
                hx = torch.zeros(1, 256)
            else:
                hx = hx.detach()

            # Inference
            value, logit, hx = global_net.forward((state.view(-1, 1, 80, 80), hx))
            action_log_probs = F.log_softmax(logit, dim=-1)

            # Take action with highest probability
            action = action_log_probs.max(1, keepdim=True)[1].numpy().squeeze()

            # Perform action
            state, reward, done, _ = env.step(action)
            state = torch.tensor(preprocess_state(state))

            env.render()

            if done:
                state = torch.tensor(preprocess_state(env.reset()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', type=str, default='models')  # Name of the directory where the models are saved
    parser.add_argument('--model-file', type=str,
                        default='model-worker-7-7200000.pth')  # Name of the model file
    _enjoy(parser.parse_args())
