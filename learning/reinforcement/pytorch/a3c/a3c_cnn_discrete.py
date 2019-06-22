#!/usr/bin/env python
# manual

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.autograd import Variable
import numpy as np


def preprocess_state(obs):
    from scipy.misc import imresize
    return imresize(obs.mean(2), (60, 80)).astype(np.float32).reshape(1, 60, 80) / 255.


class Net(nn.Module):
    def __init__(self, channels, num_actions):
        super(Net, self).__init__()
        self.channels = channels
        self.num_actions = num_actions

        # Conv Filter output size:
        # o = output
        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1
        # p = padding
        # k = kernel_size
        # s = stride
        # d = dilation

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=2,
                               padding=1)  # (16,30,40)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)  # (16,15,20)

        self.dense_size = int(16 * (60 * .5 * .5) * (80 * .5 * .5))
        self.pi = nn.Linear(self.dense_size, self.num_actions)  # actor
        self.v = nn.Linear(self.dense_size, 1)  # critic

        # Weight & bias initialization
        for layer in [self.conv1, self.conv2, self.pi, self.v]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, x):
        x = F.elu(self.conv1(x))
        x = F.elu(self.conv2(x))
        x = x.contiguous().view(-1, self.dense_size)

        pi = self.pi(x)
        values = self.v(x)
        return pi, values

    def choose_action(self, s):
        self.eval()
        s = v_wrap(s)
        s.unsqueeze_(0)
        logits, _ = self.forward(s)
        probabilities = F.softmax(logits, dim=1)
        action_distribution = self.distribution(probabilities)
        return action_distribution.sample().numpy()[0]

    def loss_func(self, s, a, v_t):
        self.train()
        logits, values = self.forward(s)
        td = v_t - values  # Temporal Difference as in DQN
        critic_loss = td.pow(2)

        probabilities = F.softmax(logits, dim=1)
        action_distribution = self.distribution(probabilities)
        a_loss = -action_distribution.log_prob(a) * td.detach().squeeze()
        total_loss = (critic_loss + a_loss).mean()

        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, global_episode, global_episode_reward, res_queue, name,
                 env_name='Breakout-v0', graphical_output=False, max_episodes=20, max_steps_per_episode=100,
                 sync_frequency=100, gamma=0.9):
        super(Worker, self).__init__()
        self.name = 'Worker %s' % name
        self.env_name = env_name
        self.global_episode, self.global_episode_reward = global_episode, global_episode_reward
        self.res_queue = res_queue
        self.global_net, self.optimizer = global_net, optimizer
        self.total_step = 0
        self.max_episodes = max_episodes
        self.max_steps_per_episode = max_steps_per_episode
        self.update_global_net_frequency = sync_frequency
        self.graphical_output = graphical_output
        self.gamma = gamma

    def run(self):
        import gym

        # We have to initialize the gym here, otherwise the multiprocessing will crash
        self.env = gym.make(self.env_name).unwrapped

        self.shape_obs_space = self.env.observation_space.shape
        self.num_actions = self.env.action_space.n

        self.local_net = Net(1, self.num_actions)  # local network, no shared weights

        # Start training
        while self.global_episode.value < self.max_episodes:

            # init
            obs = torch.from_numpy(preprocess_state(self.env.reset()))

            values = []
            log_probs = []
            rewards = []
            entropies = []

            t = 0
            episode_reward = 0
            done = False

            # sync local model with shared model
            self.local_net.load_state_dict(self.global_net.state_dict())

            # Play an episode
            while done is False:
                t += 1
                self.total_step += 1

                if self.name == "Worker 0" and self.graphical_output:
                    self.env.render()

                logits, value = self.local_net.forward(Variable(obs.unsqueeze(0)))

                prob = F.softmax(logits, dim=0)
                log_prob = F.log_softmax(logits, dim=0)

                entropy = -(log_prob * prob).sum(1, keepdim=True)
                entropies.append(entropy)

                action = prob.multinomial(num_samples=1).data
                log_prob = log_prob.gather(1, Variable(action))

                # Step environment
                obs, reward, done, _ = self.env.step(action.numpy())
                obs = torch.from_numpy(preprocess_state(obs))

                done = done or t >= self.max_steps_per_episode

                # Clamp reward to [-1,1]
                episode_reward += max(min(reward, 1), -1)

                # Book keeping
                values.append(value)
                log_probs.append(log_prob)
                rewards.append(episode_reward)
            # --- End Episode

            # The episode has ended but we still have to add the terminal reward.
            R = torch.zeros(1, 1)
            if not done:
                _, value = self.local_net.forward(Variable(obs.unsqueeze(0)))
                R = value.data
            values.append(Variable(R))

            policy_loss = 0
            value_loss = 0

            R = Variable(R)

            gae = torch.zeros(1, 1)
            for i in reversed(range(len(rewards))):
                R = self.gamma * R + rewards[i]
                advantage = R - values[i]
                value_loss += 0.5 * advantage.pow(2)

                # Generalized Advantage Estimation (gae)
                delta_t = rewards[i] + self.gamma * values[i + 1].data - values[i].data
                gae = gae * self.gamma + delta_t

                policy_loss += -(log_probs[i] * Variable(gae) + 0.01 * entropies[i])

            self.optimizer.zero_grad()
            (policy_loss + 0.5 * value_loss).backward()
            torch.nn.utils.clip_grad_norm(self.local_net.parameters(), 40)

            ensure_shared_grads(self.local_net, self.global_net)

            self.optimizer.step()

            # Book keeping of the training statistics
            with self.global_episode.get_lock():
                self.global_episode.value += 1

            with self.global_episode_reward.get_lock():
                # Moving average
                if self.global_episode_reward.value == 0.:
                    self.global_episode_reward.value = episode_reward
                else:
                    self.global_episode_reward.value = self.global_episode_reward.value * 0.9 + \
                                                       episode_reward * 0.1
            self.res_queue.put(self.global_episode_reward.value)

            print(self.name, 'Global Episode:', self.global_episode.value,
                  '| Global Episode R: %.0f' % self.global_episode_reward.value)

        self.env.close()
        self.res_queue.put(None)


def ensure_shared_grads(model, shared_model):
    for param, shared_param in zip(model.parameters(), shared_model.parameters()):
        if shared_param.grad is not None:
            return
        shared_param._grad = param.grad
