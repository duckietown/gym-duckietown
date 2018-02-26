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






        self.fc2 = nn.Linear(64 + 64, 64)
        self.fc3 = nn.Linear(64, 2)

        self.lossFn = nn.CrossEntropyLoss()

        self.optimizer = optim.SGD(
            self.parameters(),
            lr=0.001,
            momentum=0.4
        )

    def forward(self, image, string):
        batch_size = image.size(0)




        """
        x = torch.cat((rnn_hidden, img_out), 1)
        x = F.relu(self.fc2(x))
        class_scores = self.fc3(x)
        class_probs = F.softmax(class_scores, dim=1)

        return class_probs
        """

    def train(self, image, string, label):
        """
        Expects image, string and labels to be in tensor form
        """

        image = Variable(torch.from_numpy(image).float())
        label = Variable(torch.from_numpy(label).long())


        # zero the parameter gradients
        self.optimizer.zero_grad()

        # forward + backward + optimize
        outputs = self(image, string)
        loss = self.lossFn(outputs, label)
        loss.backward()
        self.optimizer.step()

        return loss.cpu().data[0]














env = SimpleSimEnv()

Model = Model(env.observation_space)






for i in range(0, 100):
    print(i)

    obs = env.reset()
    dist, dotDir = env.getLanePos()
