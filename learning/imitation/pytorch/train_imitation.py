#!/usr/bin/env python3

"""
This script will train a CNN model using imitation learning from a PurePursuit Expert.
"""

import time
import random
import argparse
import math
import json
from functools import reduce
import operator

import numpy as np
import torch
import torch.optim as optim

from utils.env import launch_env
from utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper, MetricsWrapper
from utils.teacher import PurePursuitExpert

from imitation.pytorch.model import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env)
    env = ImgWrapper(env)
    env = DtRewardWrapper(env)
    env = MetricsWrapper(env)
    env = ActionWrapper(env)
    print("Initialized Wrappers")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []

    # let's collect our samples
    for episode in range(0, args.episodes):
        print("Starting episode", episode)
        for steps in range(0, args.steps):
            # use our 'expert' to predict the next action.
            action = expert.predict(None)
            observation, reward, done, info = env.step(action)
            observations.append(observation)
            actions.append(action)
        env.reset()
    env.close()

    actions = np.array(actions)
    observations = np.array(observations)

    model = Model(action_dim=2, max_action=1.)
    model.train().to(device)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    avg_loss = 0
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

        model_actions = model(obs_batch)

        loss = (model_actions - act_batch).norm(2).mean()
        loss.backward()
        optimizer.step()

        loss = loss.data[0]
        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(model.state_dict(), 'imitation/pytorch/models/imitate.pt')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=1234, type=int, help="Sets Gym, TF, and Numpy seeds")
    parser.add_argument("--episodes", default=3, type=int, help="Number of epsiodes for experts")
    parser.add_argument("--steps", default=50, type=int, help="Number of steps per episode")
    parser.add_argument("--batch-size", default=32, type=int, help="Training batch size")
    parser.add_argument("--epochs", default=1, type=int, help="Number of training epochs")
    parser.add_argument("--model-directory", default="models/", type=str, help="Where to save models")

    args = parser.parse_args()

    _train(args)
