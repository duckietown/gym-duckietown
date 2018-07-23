#!/usr/bin/env python3

"""
This script will train a CNN model using imitation learning.
You should first start the gen_demos.py script to generate a dataset of
demonstrations, then start this script to begin training.
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
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

from gym_duckietown.envs import SimpleSimEnv
from gym_duckietown.wrappers import HeadingWrapper

from utils import *

class Model(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, 6, stride=3),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),

            nn.Conv2d(32, 32, 4, stride=2),
            nn.BatchNorm2d(32),
            nn.LeakyReLU()
        )

        self.enc_to_vels = nn.Sequential(
            nn.Linear(32 * 8 * 11, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Linear(128, 2),
            #nn.Tanh()
        )

        self.apply(init_weights)

    def forward(self, img):
        img = img / 255

        x = self.encoder(img)
        x = x.view(x.size(0), -1)
        vels = self.enc_to_vels(x)

        return vels

positions = []
actions = []

def load_data(map_name):
    global positions
    global actions

    file_name = 'experiments/demos_%s.json' % map_name
    try:
        with open(file_name, 'r') as f:
            data = json.load(f)
    except:
        print('failed to load data file "%s"' % file_name)
        return

    demos = data['demos']
    positions = map(lambda d: d['positions'], demos)
    actions = map(lambda d: d['actions'], demos)

    positions = sum(positions, [])
    actions = sum(actions, [])

    assert len(positions) == len(actions)

def gen_data():
    idx = random.randint(0, len(positions) - 1)
    cur_pos = np.array(positions[idx][0])
    cur_angle = positions[idx][1]
    vels = np.array(actions[idx])

    env.unwrapped.cur_pos = cur_pos
    env.unwrapped.cur_angle = cur_angle

    obs = env.unwrapped.render_obs().copy()
    obs = obs.transpose(2, 0, 1)

    return obs, vels

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--map-name', required=True)
    args = parser.parse_args()

    load_data(args.map_name)

    env = SimpleSimEnv(map_name=args.map_name)
    env = HeadingWrapper(env)

    model = Model()
    model.train()
    if torch.cuda.is_available():
        model.cuda()
    print_model_info(model)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.SGD(
        model.parameters(),
        lr=0.0004,
        weight_decay=1e-3
    )

    avg_loss = 0
    num_epochs = 2000000

    for epoch in range(1, num_epochs+1):
        optimizer.zero_grad()

        env.reset()
        obs, vels = gen_batch(gen_data)

        model_vels = model(obs)

        loss = (model_vels - vels).norm(2).mean()
        loss.backward()
        optimizer.step()

        loss = loss.data[0]
        avg_loss = avg_loss * 0.995 + loss * 0.005

        print('epoch %d, loss=%.3f' % (epoch, avg_loss))

        #print('gen time: %d ms' % genTime)
        #print('train time: %d ms' % trainTime)

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(model.state_dict(), 'trained_models/imitate.pt')

        # Periodically reload the training data
        if epoch % 200 == 0:
            load_data(args.map_name)
