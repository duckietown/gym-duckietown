#!/usr/bin/env python3

import time
from functools import reduce

import gym_duckietown
from gym_duckietown.envs import SimpleSimEnv

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.distributions import Categorical

class Model(nn.Module):
    def __init__(self, obs_space):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv1_drop = torch.nn.Dropout2d(p=0.2)

        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv2_drop = torch.nn.Dropout2d(p=0.2)

        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3_drop = torch.nn.Dropout2d(p=0.2)

        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.linear1_drop = nn.Dropout(p=0.5)
        self.linear1 = nn.Linear(32 * 10 * 10, 256)

        self.linear2 = nn.Linear(256, 1)

        # L1 loss, absolute value of element-wise difference
        self.lossFn = nn.L1Loss()

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.4
        )

    def forward(self, image, string):
        batch_size = image.size(0)

        x = self.conv1(image)
        #x = self.conv1_drop(x)
        x = F.leaky_relu(x)

        x = self.conv2(x)
        #x = self.conv2_drop(x)
        x = F.leaky_relu(x)

        x = self.conv3(x)
        #x = self.conv3_drop(x)
        x = F.leaky_relu(x)

        x = self.conv4(x)
        x = F.leaky_relu(x)

        x = x.view(-1, 32 * 10 * 10)
        x = self.linear1_drop(x)
        x = self.linear1(x)
        x = F.leaky_relu(x)

        x = self.linear2(x)

        return x

        """
        x = torch.cat((rnn_hidden, img_out), 1)
        x = F.relu(self.fc2(x))
        class_scores = self.fc3(x)
        class_probs = F.softmax(class_scores, dim=1)

        return class_probs
        """

    def train(self, image, dist):
        """
        Expects image, string and labels to be in tensor form
        """

        image = Variable(torch.from_numpy(image).float())
        dist = Variable(torch.from_numpy(dist).float())

        # Zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        output = self(image, string)
        loss = self.lossFn(output, dist)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().data[0]







env = SimpleSimEnv()

Model = Model(env.observation_space)






for i in range(0, 100):
    print(i)

    obs = env.reset().transpose(2, 0, 1)
    dist, dotDir = env.getLanePos()
