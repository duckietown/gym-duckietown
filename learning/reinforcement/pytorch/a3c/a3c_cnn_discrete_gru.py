#!/usr/bin/env python
# manual

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import numpy as np

os.environ['OMP_NUM_THREADS'] = '1'


def preprocess_state(obs):
    from scipy.misc import imresize
    return imresize(obs.mean(2), (80, 80)).astype(np.float32).reshape(1, 80, 80) / 255.


class Net(nn.Module):
    def __init__(self, channels, num_actions):
        super(Net, self).__init__()
        self.channels = channels
        self.num_actions = num_actions

        # Conv Filter output size:
        # o = output
        # p = padding
        # k = kernel_size
        # s = stride
        # d = dilation

        # o = [i + 2*p - k - (k-1)*(d-1)]/s + 1

        self.conv1 = nn.Conv2d(in_channels=self.channels, out_channels=16, kernel_size=3, stride=2,
                               padding=1)  # (16,30,40)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=2, padding=1)  # (16,15,20)

        self.gru = nn.GRUCell(int(16 * (80 * .5 * .5) * (80 * .5 * .5)), 256)
        # self.dense_size = int(16 * (60 * .5 * .5) * (80 * .5 * .5))
        self.pi = nn.Linear(256, self.num_actions)  # actor
        self.v = nn.Linear(256, 1)  # critic

        # Weight & bias initialization
        for layer in [self.conv1, self.conv2, self.pi, self.v]:
            nn.init.normal_(layer.weight, mean=0., std=0.1)
            nn.init.constant_(layer.bias, 0.)

    def forward(self, inputs):
        inputs, hx = inputs
        x = F.elu(self.conv1(inputs))
        x = F.elu(self.conv2(x))
        gru_input = x.view(-1, int(16 * (80 * .5 * .5) * (80 * .5 * .5)))
        hx = self.gru(gru_input, hx)

        pi = self.pi(hx)
        values = self.v(hx)
        return values, pi, hx

    def loss_func(self, s, a, v_t):
        logits, values, self.hx = self.forward((s, self.hx))
        td = v_t - values
        critic_loss = td.pow(2)

        probs = F.softmax(logits, dim=1)
        m = self.distribution(probs)
        exp_v = m.log_prob(a) * td.detach().squeeze()
        a_loss = -exp_v
        total_loss = (critic_loss + a_loss).mean()

        return total_loss


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, args, info, identifier):
        super(Worker, self).__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.args = args
        self.info = info
        self.identifier = identifier
        self.name = 'Worker %s' % identifier
        self.total_step = 0
        self.hx = None

    def run(self):
        import gym

        # We have to initialize the gym here, otherwise the multiprocessing will crash
        self.env = gym.make(self.args.env).unwrapped

        # Set seeds so we can reproduce our results
        self.env.seed(self.args.seed + self.identifier)
        torch.manual_seed(self.args.seed + self.identifier)

        self.shape_obs_space = self.env.observation_space.shape

        self.local_net = Net(1, self.env.action_space.n)  # local network

        start_time = last_disp_time = time.time()

        state = torch.tensor(preprocess_state(self.env.reset()))
        episode_length, epr, eploss, done = 0, 0, 0, True  # bookkeeping

        while self.info['frames'][0] <= 8e7:  # 80 millione steps.

            # Sync parameters from global net
            self.local_net.load_state_dict(self.global_net.state_dict())

            # Reset hidden state of GRU cell / Remove hidden state from computational graph
            hx = torch.zeros(1, 256) if done else hx.detach()

            # Values used to compute gradients
            values, log_probs, actions, rewards = [], [], [], []

            for step in range(self.args.steps_until_sync):
                episode_length += 1

                # Inference
                value, logit, hx = self.local_net.forward((state.view(-1, 1, 80, 80), hx))
                action_log_probs = F.log_softmax(logit, dim=-1)

                # Sample an action from the distribution
                action = torch.exp(action_log_probs).multinomial(num_samples=1).data[0]

                state, reward, done, _ = self.env.step(action.numpy()[0])
                state = torch.tensor(preprocess_state(state))
                epr += reward
                reward = np.clip(reward, -1, 1)  # clip to -1 and 1
                done = done or episode_length >= 1e4

                self.info['frames'].add_(1)

                num_frames = int(self.info['frames'].item())

                if done:  # update shared data
                    self.info['episodes'] += 1

                    # Moving average statistics
                    interp_factor = 1 if self.info['episodes'][0] == 1 else 1 - 0.99
                    self.info['run_epr'].mul_(1 - interp_factor).add_(interp_factor * epr)
                    self.info['run_loss'].mul_(1 - interp_factor).add_(interp_factor * eploss)

                # print training info every minute
                if self.identifier == 0 and time.time() - last_disp_time > 60:
                    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    print('time {}, episodes {:.0f}, frames {:.1f}M, mean epr {:.2f}, run loss {:.2f}'
                          .format(elapsed, self.info['episodes'].item(), num_frames / 1e6,
                                  self.info['run_epr'].item(), self.info['run_loss'].item()))
                    last_disp_time = time.time()

                if done:
                    episode_length, epr, eploss = 0, 0, 0
                    state = torch.tensor(preprocess_state(self.env.reset()))

                values.append(value)
                log_probs.append(action_log_probs)
                actions.append(action)
                rewards.append(reward)

            # Terminal value
            next_value = torch.zeros(1, 1) if done else self.local_net.forward((state.unsqueeze(0), hx))[0]
            values.append(next_value.detach())

            # Calculate loss
            loss = self.calc_loss(self.args, torch.cat(values), torch.cat(log_probs), torch.cat(actions),
                                  np.asarray(rewards))
            eploss += loss.item()

            # Calculate gradient
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.local_net.parameters(), 40)

            # sync gradients with global network
            for param, shared_param in zip(self.local_net.parameters(), self.global_net.parameters()):
                if shared_param.grad is None:
                    shared_param._grad = param.grad

            # Backpropagation
            self.optimizer.step()

    def calc_loss(self, args, values, log_probs, actions, rewards):
        np_values = values.view(-1).data.numpy()

        # Actor loss: Generalized Advantage Estimation
        delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]

        logpys = log_probs.gather(1,
                                  torch.tensor(actions).view(-1, 1))  # Select logps of the actions the agent executed
        gen_adv_est = discount(delta_t, args.gamma * args.tau)
        policy_loss = -(logpys.view(-1) * torch.FloatTensor(gen_adv_est.copy())).sum()

        # Critic loss: l2 loss over value estimator
        rewards[-1] += args.gamma * np_values[-1]
        discounted_r = discount(np.asarray(rewards), args.gamma)
        discounted_r = torch.tensor(discounted_r.copy(), dtype=torch.float32)
        value_loss = .5 * (discounted_r - values[:-1, 0]).pow(2).sum()

        # Entropy - Used for regularization
        entropy_loss = -log_probs.sum()
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss


def sync_nets(optimizer, local_net, global_net, done, next_state, state_buffer, action_buffer, reward_buffer, gamma):
    if done:
        v_next_state = 0.  # terminal
    else:
        flattened_input = v_wrap(next_state[None, :])
        v_next_state = local_net.forward((flattened_input, local_net.hx))[1].data.numpy()[0, 0]

    buffer_v_target = []
    for r in reward_buffer[::-1]:  # reverse buffer r
        v_next_state = r + gamma * v_next_state
        buffer_v_target.append(v_next_state)
    buffer_v_target.reverse()

    loss = local_net.loss_func(
        v_wrap(np.array(state_buffer)),
        v_wrap(np.array(action_buffer), dtype=np.int64) if action_buffer[0].dtype == np.int64 else v_wrap(
            np.vstack(action_buffer)),
        v_wrap(np.array(buffer_v_target)[:, None]))

    # calculate local gradients
    optimizer.zero_grad()
    loss.backward()


def discount(x, gamma):
    from scipy.signal import lfilter
    return lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner


def v_wrap(np_array, dtype=np.float32):
    if np_array.dtype != dtype:
        np_array = np_array.astype(dtype)
    return torch.from_numpy(np_array)
