import time
import random
import argparse
import math
import json
from functools import reduce
import operator
import sys

import numpy as np
import torch
import torch.optim as optim

from gail.models import *
from gail.dataloader import *

from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _train(args):
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

    observations = []    
    actions = []

    expert = PurePursuitExpert(env=env)
    # let's collect our samples
    if args.get_samples:
        # for episode in range(0, args.episodes):
        #     print("Starting episode", episode)
        #     for steps in range(0, args.steps):
        #         # use our 'expert' to predict the next action.
        #         action = expert.predict(None)
        #         observation, reward, done, info = env.step(action)
        #         actions.append(action)
        #     env.reset()
        # env.close()
        observations, actions, _, _, _, _, _ = generate_trajectories(env, steps=args.steps, episodes=args.episodes, expert=expert)
        torch.save(actions, '{}/data_a_t.pt'.format(args.data_directory))
        torch.save(observations, '{}/data_o_t.pt'.format(args.data_directory))    

    else:
        observations = torch.load('{}/data_o_t.pt'.format(args.data_directory))
        actions = torch.load('{}/data_a_t.pt'.format(args.data_directory))

    expert_actions = torch.FloatTensor(actions)
    expert_observations = torch.stack(observations)

    G = Generator(action_dim=2).train().to(device)
    D = Discriminator(action_dim=2).train().to(device)
    # V = Value(action_dim=2).train().to(device)

    # state_dict = torch.load('models/G_imitate_2.pt'.format(args.checkpoint), map_location=device)
    # G.load_state_dict(state_dict)

        # state_dict = torch.load('models/D_{}.pt'.format(args.checkpoint), map_location=device)
        # D.load_state_dict(state_dict)

    D_optimizer = optim.SGD(
        D.parameters(), 
        lr = args.lrD,
        weight_decay=1e-3
        )

    G_optimizer = optim.Adam(
        G.parameters(),
        lr = args.lrG,
        weight_decay=1e-3,
    )

    avg_loss = 0
    avg_g_loss = 0
    # loss_fn = nn.BCEWithLogitsLoss()
    loss_fn = nn.BCELoss()

    writer = SummaryWriter(comment='gail/{}'.format(args.training_name))
    if args.pretrain_D:
        for i in range(args.pretrain_D):
            batch_indices = np.random.randint(0, expert_actions.shape[0], (args.batch_size))
            
            obs_batch = expert_observations[batch_indices].float().to(device).data
            act_batch = expert_actions[batch_indices].float().to(device).data
            
            model_actions = G.select_action(obs_batch)
            prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, loss_fn, D, obs_batch, act_batch, model_actions, writer, -1)
    
    if args.checkpoint:
        try:
            state_dict = torch.load('models/G_{}.pt'.format(args.checkpoint), map_location=device)
            G.load_state_dict(state_dict)
        except:
            pass

    for epoch in range(args.epochs):
        batch_indices = np.random.randint(0, expert_observations.shape[0], (args.batch_size))
        obs_batch = expert_observations[batch_indices].float().to(device).data
        act_batch = expert_actions[batch_indices].float().to(device).data

        model_actions = G.select_action(obs_batch)

        ## Update D

        if args.D_train_eps > 0 and epoch % args.D_train_eps == 0:
            prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch)
        # elif args.D_train_eps>0:
        #     D_train_eps = args.D_train_eps
        #     prob_expert, prob_policy, loss = update_discriminator(args, D_optimizer, D_train_eps, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch)
        
        # Update G
        if args.do_ppo:
            observations, actions, log_probs, rewards, masks, values, next_value = generate_trajectories(env, steps=args.steps, episodes=1, policy=G, d=D)

            returns = compute_gae(next_value, rewards, masks, values, args)

            returns   = torch.cat(returns).detach().view(len(returns),1)
            log_probs = torch.stack(log_probs).detach()
            values    = torch.cat(values).detach()

            states    = torch.stack(observations)
            actions   = torch.cat(actions)
            advantage = returns - values
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            
            do_ppo_step(states, actions, log_probs, returns, advantage, args, G, G_optimizer)
        else:
            loss_g = do_policy_gradient(args, G, D, env, obs_batch, model_actions)
            loss_g.backward()
            G_optimizer.step()

        avg_g_loss = loss_g.item()
        avg_loss = loss.item() 
        # avg_g_loss = D(obs_batch,model_actions).mean()
        writer.add_scalar("G/loss", avg_g_loss, epoch) #should go towards -inf?
        print('epoch %d, D loss=%.5f, G loss=%.5f' % (epoch, avg_loss, avg_g_loss))

        # Periodically save the trained model
        if epoch % 200 == 0:
            torch.save(D.state_dict(), '{}/D_{}_epoch_{}.pt'.format(args.model_directory,args.training_name,epoch))
            torch.save(G.state_dict(), '{}/G_{}_epoch_{}.pt'.format(args.model_directory,args.training_name,epoch))
        torch.save(D.state_dict(), '{}/D_{}.pt'.format(args.model_directory,args.training_name))
        torch.save(G.state_dict(), '{}/G_{}.pt'.format(args.model_directory,args.training_name))
        torch.cuda.empty_cache()

    # writer.add_graph("generator", G)
    # writer.add_graph("discriminator",D)


def generate_trajectories(env, steps=10, policy=None, d=None,  episodes=1, expert=None):
    observations = []
    actions = []
    log_probs = []
    rewards = []
    masks = []
    values = []
    next_value = 0

    for episode in range(0, episodes): 
        while True:
            try:
                o = []
                a = []
                l = []
                r = []
                m = []
                v = []
                obs = env.reset()
                for step in range(0, steps):
                    # use our 'expert' to predict the next action.
                    o.append(torch.FloatTensor(obs).to(device))
                    
                    if expert:
                        action = expert.predict(None)
                        a.append(action)
                    else:
                        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)

                        action = policy.select_action(obs)
                        a.append(action)
                        r.append(d(obs,action).unsqueeze(1).to(device)) 

                        action = action.squeeze().data.cpu().numpy()
                    
                        # a.append(torch.FloatTensor(action)).to(device)
                        v_dist, s_dist, value = policy(obs)
                        l.append(torch.FloatTensor([v_dist.log_prob(action[0]), s_dist.log_prob(action[1])]))


                        v.append(value)

                    obs, reward, done, info = env.step(action)

                    mask = 0 if done or step == steps-1 else 1
                    m.append(torch.FloatTensor([mask]).unsqueeze(1).to(device))

                observations += o
                actions += a
                log_probs += l
                rewards += r
                masks += m
                values += v
                break
            except ValueError: 
                print(o,a,l,r,m,v)
                break
            except KeyboardInterrupt:
                break
            except:
                print("Unexpected error:", sys.exc_info()[0])
                pass

    if not expert:
        obs = torch.from_numpy(obs).float().to(device).unsqueeze(0)
        _ , _ , next_value = policy(obs)
    return observations, actions, log_probs, rewards, masks, values, next_value
            
    # return torch.FloatTensor(observations).to(device), \
    #         torch.FloatTensor(actions).to(device), \
    #         torch.FloatTensor(log_probs).to(device), \
    #         torch.FloatTensor(rewards).to(device), \
    #         torch.FloatTensor(masks).to(device), \
    #         torch.FloatTensor(values).to(device)

def do_policy_gradient(args, G, D, env, obs_batch, model_actions):
    if args.rollout:
        obs, acts, _, _, _, vals, _ = generate_trajectories(env, steps=10, episodes=1, policy=G, d=D)
        obs = torch.stack(obs)
        acts = torch.cat(acts)
    else:
        obs = obs_batch
        acts = model_actions

    # loss_g = D(obs,acts).log().mean()
    loss_g = torch.cat([d*args.gamma**i for i,d in enumerate(D(obs,acts).log()-torch.cat(vals))]).mean()

    return loss_g

def update_discriminator(args, D_optimizer, loss_fn, D, obs_batch, act_batch, model_actions, writer, epoch):
    D_optimizer.zero_grad()

    exp_label = torch.full((args.batch_size,1), 0, device=device).float()
    policy_label = torch.full((args.batch_size,1), 1, device=device).float()

    prob_expert = D(obs_batch,act_batch)
    expert_loss = loss_fn(prob_expert, exp_label)

    prob_policy = D(obs_batch,model_actions)
    policy_loss = loss_fn(prob_policy, policy_label)

    loss = (expert_loss + policy_loss)

    # loss = -(prob_expert.mean() - prob_policy.mean())

    # if epoch % 10:
    loss.backward(retain_graph=True)
    D_optimizer.step()

    # for p in D.parameters():
    #     p.data.clamp_(-0.01,0.01)

    if epoch != -1:
        writer.add_scalar("expert D probability", torch.mean(prob_expert), epoch)
        writer.add_scalar("policy D probability", torch.mean(prob_policy), epoch)
        writer.add_scalar("D/loss", loss, epoch)

    return prob_expert.data, prob_policy.data, loss.data



def ppo_iter(states, actions, log_probs, returns, advantage):
    batch_size = states.shape[0]
    # generates random mini-batches until we have covered the full batch
    for _ in range(batch_size // 5):
        rand_ids = np.random.randint(0, batch_size, 5)
        yield states[rand_ids, :], actions[rand_ids, :], log_probs[rand_ids, :], returns[rand_ids, :], advantage[rand_ids, :]
        

def do_ppo_step(states, actions, log_probs, returns, advantages, args, G, G_optimizer):
    '''
    from https://github.com/colinskow/move37/blob/f57afca9d15ce0233b27b2b0d6508b99b46d4c7f/ppo/ppo_train.py#L63
    '''
    for e in range(args.ppo_epochs):
        for state, action, old_log_probs, return_, advantage in ppo_iter(states, actions, log_probs, returns, advantages):
            v_dist, s_dist, value = G(state)
            entropy = v_dist.entropy().mean() + s_dist.entropy().mean()

            new_log_probs = torch.cat((v_dist.log_prob(action[:,0].view(action.shape[0],1)),s_dist.log_prob(action[:,1].view(action.shape[0],1))),1)
            ratio = (new_log_probs.to(device) - old_log_probs.to(device)).exp()

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - args.clip_param, 1.0 + args.clip_param) * advantage

            actor_loss  = - torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()

            loss = args.critic_discount* critic_loss + actor_loss - args.entropy_beta * entropy

            G_optimizer.zero_grad()
            loss.backward(retain_graph=True)
            G_optimizer.step()



            # track statistics
            # sum_returns += return_.mean()
            # sum_advantage += advantage.mean()
            # sum_loss_actor += actor_loss
            # sum_loss_critic += critic_loss
            # sum_loss_total += loss
            # sum_entropy += entropy
            
            # count_steps += 1



def get_ppo_loss(old_dist, old_value, new_dist, new_value, actions, rewards, advantages, args):
    '''
    referenced from https://github.com/adik993/ppo-pytorch/blob/master/agents/ppo.py
    '''
    value_old_clipped = old_value + (new_value - old_value).clamp(-args.v_clip_range, args.v_clip_range)
    v_old_loss_clipped = (rewards - value_old_clipped).pow(2)
    v_loss = (rewards - new_value).pow(2)
    value_loss = torch.min(v_old_loss_clipped, v_loss).mean()

    # Policy loss
    advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-8)
    advantage.detach_()
    log_prob = new_dist.log_prob(actions)
    log_prob_old = old_dist.log_prob(actions)
    ratio = (log_prob - log_prob_old).exp().view(-1)

    surrogate = advantage * ratio
    surrogate_clipped = advantage * ratio.clamp(1 - args.clip_range, 1 + args.clip_range)
    policy_loss = torch.min(surrogate, surrogate_clipped).mean()

    # Entropy
    entropy = new_dist.entropy().mean()

    # Total loss
    losses = policy_loss + args.c_entropy * entropy - args.c_value * value_loss
    total_loss = -losses
    return total_loss


    