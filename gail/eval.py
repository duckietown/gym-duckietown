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

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"
def _eval(args):
    args.data_directory = "D:/Michael/Learning/duckietown_evaldata"
    args.episodes = 20

    if args.get_samples:
        
        observations = []    
        actions = []

        from learning.utils.env import launch_env
        from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
            DtRewardWrapper, ActionWrapper, ResizeWrapper
        from learning.utils.teacher import PurePursuitExpert

        env = launch_env()
        env = ResizeWrapper(env)
        env = NormalizeWrapper(env) 
        env = ImgWrapper(env)
        env = ActionWrapper(env)
        env = DtRewardWrapper(env)
        expert = PurePursuitExpert(env=env)

        for episode in range(0, args.eval_episodes):
            print("Starting episode", episode)
            for steps in range(0, args.eval_steps):
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

    actions = np.array(actions)
    observations = np.array(observations)

    G = Generator(2)
    state_dict = torch.load('models/G_{}.pt'.format(args.training_name), map_location=device)
    G.load_state_dict(state_dict)
    G.to(device)

    G_random = Generator(2)
    # r_state_dict = torch.load('models/G_random.pt', map_location=device)
    # G_random.load_state_dict(r_state_dict)
    G_random.to(device)
    # torch.save(G.state_dict(), '{}/G_random.pt'.format(args.model_directory))


    model_scores = []
    random_scores = []

    with torch.no_grad():
        observations = torch.FloatTensor(observations).to(device)
        
        expert_actions = torch.FloatTensor(actions).to(device)
        model_actions = G.select_action(observations).to(device)
        random_actions = G_random.select_action(observations).to(device)
        
        random_scores = (expert_actions-random_actions).abs()
        model_scores = (expert_actions-model_actions).abs()

        # expert_score = (np.mean(expert_actions-random_actions) - np.mean(expert_actions-model_actions))/np.mean(expert_actions-random_actions)
        scores = torch.ones(model_scores.shape) - (model_scores/random_scores)

    print(scores.mean(), scores.std())

    return scores.mean(), scores.std()
    # print(random_score-model_score)
        

