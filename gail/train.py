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

def _train(args):
    # from learning.utils.env import launch_env
    # from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
    #     DtRewardWrapper, ActionWrapper, ResizeWrapper
    # from learning.utils.teacher import PurePursuitExpert

    if args.get_samples:
        generate_expert_trajectorys(args)
    
    data = ExpertTrajDataset(args)

    G = Generator(action_dim=2).to(device)
    D = Discriminator(action_dim=2).to(device)
    state_dict = torch.load('models/G_imitate_2.pt'.format(args.checkpoint), map_location=device)
    G.load_state_dict(state_dict)
    # if args.use_checkpoint:
    #     state_dict = torch.load('models/G_{}.pt'.format(args.checkpoint), map_location=device)
    #     G.load_state_dict(state_dict)
    #     state_dict = torch.load('models/D_{}.pt'.format(args.checkpoint), map_location=device)
    #     D.load_state_dict(state_dict)

    D_optimizer = optim.SGD(
        D.parameters(), 
        lr = args.lrD,
        weight_decay=1e-3
        )

    G_optimizer = optim.SGD(
        G.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    avg_loss = 0
    avg_g_loss = 0
    loss_fn = nn.BCEWithLogitsLoss()
    # loss_fn = nn.BCELoss()

    validation = args.epochs-1
    writer = SummaryWriter(comment='gail')
    for epoch in range(args.epochs):
        if epoch % int(args.epochs/(args.episodes*0.7)) == 0 and args.batch_size < args.steps: #if divisible by 7 sample new trajectory?
            rand_int = np.random.randint(0,(args.episodes*0.7))
            observations = torch.FloatTensor(data[rand_int]['observation']).to(device)
            actions =  torch.FloatTensor(data[rand_int]['action']).to(device)

        
        batch_indices = np.random.randint(0, observations.shape[0], (args.batch_size))

        obs_batch = observations[batch_indices]
        act_batch = actions[batch_indices]

        model_actions = G(obs_batch)

        ## Update D

        exp_label = torch.full((args.batch_size,1), 1, device=device).float()
        policy_label = torch.full((args.batch_size,1), 0, device=device).float()

        ##Making labels soft
        # exp_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1.1
        # exp_label = exp_label.clamp(0.7,1.2)

        # policy_label = torch.randn((args.batch_size,1), device=device).float()*0.1 + 1
        # policy_label = policy_label.clamp(0,0.3)
        ##
        for _ in range(20):

            D_optimizer.zero_grad()

            prob_expert = D(obs_batch,act_batch)
            expert_loss = loss_fn(prob_expert, exp_label)
            # writer.add_scalar("expert D loss", expert_loss, epoch)
            writer.add_scalar("expert D loss", torch.mean(prob_expert), epoch)

            prob_policy = D(obs_batch,model_actions)
            policy_loss = loss_fn(prob_policy, policy_label)
            # writer.add_scalar("policy D loss", policy_loss, epoch)
            writer.add_scalar("policy D loss", torch.mean(prob_policy), epoch)

            # loss = (expert_loss + policy_loss)

            loss = -(torch.mean(prob_expert) - torch.mean(prob_policy))

            writer.add_scalar("D/loss", loss, epoch)
            # if epoch % 10:
            loss.backward(retain_graph=True)
            D_optimizer.step()

            for p in D.parameters():
                p.data.clamp_(-0.01,0.01)

        ## Update G

        G_optimizer.zero_grad()

        loss_g = -(torch.mean(D(obs_batch,model_actions)))
        # loss_g = loss_g.mean()
        loss_g.backward()
        G_optimizer.step()

        avg_g_loss = loss_g.item()
        avg_loss = loss.item() 

        writer.add_scalar("G/loss", avg_g_loss, epoch) #should go towards -inf?
        print('epoch %d, D loss=%.3f, G loss=%.3f' % (epoch, avg_loss, avg_g_loss))

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(D.state_dict(), '{}/D2.pt'.format(args.model_directory))
            torch.save(G.state_dict(), '{}/G2.pt'.format(args.model_directory))
        if epoch % 1000 == 0:
            torch.save(D.state_dict(), '{}/D_epoch{}.pt'.format(args.model_directory,epoch))
            torch.save(G.state_dict(), '{}/G_epoch{}.pt'.format(args.model_directory,epoch))
        torch.cuda.empty_cache()
    torch.save(D.state_dict(), '{}/D2.pt'.format(args.model_directory))
    torch.save(G.state_dict(), '{}/G2.pt'.format(args.model_directory))
    # writer.add_graph("generator", G)
    # writer.add_graph("discriminator",D)

