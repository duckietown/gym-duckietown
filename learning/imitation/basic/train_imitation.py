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

from gail.models import *
from gail.dataloader import *

from learning.utils.env import launch_env
from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    DtRewardWrapper, ActionWrapper, ResizeWrapper
from learning.utils.teacher import PurePursuitExpert

from learning.imitation.basic.model import Model
from gail.models import Generator
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
    env = launch_env()
    env = ResizeWrapper(env)
    env = NormalizeWrapper(env) 
    env = ImgWrapper(env)
    env = ActionWrapper(env)
    env = DtRewardWrapper(env)
    print("Initialized Wrappers")

    observation_shape = (None, ) + env.observation_space.shape
    action_shape = (None, ) + env.action_space.shape

    # Create an imperfect demonstrator
    expert = PurePursuitExpert(env=env)

    observations = []
    actions = []

    writer = SummaryWriter(comment='imitate/{}'.format(args.training_name))


    # let's collect our samples
    if args.get_samples:
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
        torch.save(actions, '{}/data_a_t.pt'.format(args.data_directory))
        torch.save(observations, '{}/data_o_t.pt'.format(args.data_directory))    

    else:
        observations = torch.load('{}/data_o_t.pt'.format(args.data_directory))
        actions = torch.load('{}/data_a_t.pt'.format(args.data_directory))
        

        # data = ExpertTrajDataset(args)
        # for i in range(args.episodes):
        #     observations.append(data[i]['observation'][0])
        #     actions.append(data[i]['action'][0])


    actions = np.array(actions)
    observations = np.array(observations)

    # model = Model(action_dim=2, max_action=1.)
    model = Generator(action_dim=2)
    # state_dict = torch.load('models/G_imitate_2.pt', map_location=device)
    # model.load_state_dict(state_dict)
    model.train().to(device)

    # weight_decay is L2 regularization, helps avoid overfitting
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lrG,
        # weight_decay=1e-3
    )

    avg_loss = 0
    last_loss = 0
    writer.add_graph(model,input_to_model=torch.from_numpy(observations).float().to(device), verbose=False)
    for epoch in range(args.epochs):
        optimizer.zero_grad()

        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))
        obs_batch = torch.from_numpy(observations[batch_indices]).float().to(device)
        act_batch = torch.from_numpy(actions[batch_indices]).float().to(device)

        model_actions = model.select_action(obs_batch)

        loss = (model_actions - act_batch).norm(2).mean()
        
        loss.backward()
        optimizer.step()

        loss = loss.data.item()
        avg_loss = avg_loss * 0.995 + loss * 0.005
        
        writer.add_scalar("G/loss", loss, epoch) #should go towards -inf?

        print('epoch %d, loss=%.3f' % (epoch, loss))
        
        # Periodically save the trained model
        if epoch - 200 == 0 or epoch % 1000 == 0:
            torch.save(model.state_dict(), '{}/G_{}_epoch_{}.pt'.format(args.model_directory, args.training_name, epoch))
        # if epoch % 1000 == 0:
        #     torch.save(model.state_dict(), '{}/G_{}_epoch{}.pt'.format(args.model_directory,args.training_name,epoch))
        #     # torch.save(G.state_dict(), '{}/G_{}_epoch{}.pt'.format(args.model_directory,args.training_name,epoch))
        if abs(last_loss - loss) < args.eps:
            break
        last_loss = loss
        torch.save(model.state_dict(), '{}/G_{}.pt'.format(args.model_directory, args.training_name))

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