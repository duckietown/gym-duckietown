import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
from learning.reinforcement.pytorch.a3c import CustomOptimizer

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

        self.conv1 = nn.Conv2d(channels, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 32, 3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(32, 32, 3, stride=2, padding=1)

        self.gru = nn.GRUCell(32 * 5 * 5, 256)
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
        x = F.elu(self.conv3(x))
        x = F.elu(self.conv4(x))
        gru_input = x.view(-1, 32 * 5 * 5)
        hx = self.gru(gru_input, hx)

        pi = self.pi(hx)
        values = self.v(hx)
        return values, pi, hx


class Worker(mp.Process):
    def __init__(self, global_net, optimizer, args, info, identifier, logger):
        super(Worker, self).__init__()
        self.global_net = global_net
        self.optimizer = optimizer
        self.args = args
        self.info = info
        self.identifier = identifier
        self.name = f'worker-{identifier}'
        self.total_step = 0
        self.args = args
        self.ckpt_dir, self.ckpt_path, self.log_dir = logger.get_log_dirs()

    def calc_loss(self, args, values, log_probs, actions, rewards):
        np_values = values.view(-1).data.numpy()

        # Actor loss: Generalized Advantage Estimation A = R(lamdda) - V(s), Schulman
        # Paper:  High-Dimensional Continuous Control Using Generalized Advantage Estimation
        delta_t = np.asarray(rewards) + args.gamma * np_values[1:] - np_values[:-1]
        advantage = discount(delta_t, args.gamma)

        # Select log probabilities of the actions the agent executed
        action_log_probabilities = log_probs.gather(1, torch.tensor(actions).view(-1, 1))
        policy_loss = -(action_log_probabilities.view(-1) * torch.FloatTensor(advantage.copy())).sum()

        # Critic loss: l2 loss over value estimator
        rewards[-1] += args.gamma * np_values[-1]
        discounted_reward = discount(np.asarray(rewards), args.gamma)
        discounted_reward = torch.tensor(discounted_reward.copy(), dtype=torch.float32)
        value_loss = .5 * (discounted_reward - values[:-1, 0]).pow(2).sum()

        # Entropy - Used for regularization
        # Entropy is a metric for the distribution of probabilities
        # -> We want to maximize entropy to encourage exploration
        entropy_loss = (-log_probs * torch.exp(log_probs)).sum()
        return policy_loss + 0.5 * value_loss - 0.01 * entropy_loss

    def run(self):
        from learning.utils.env import launch_env
        from learning.utils.wrappers import NormalizeWrapper, ImgWrapper, \
            DtRewardWrapper2, ActionWrapper, ResizeWrapper, DiscreteWrapper

        # We have to initialize the gym here, otherwise the multiprocessing will crash
        self.env = launch_env()
        #self.env = ResizeWrapper(self.env)
        #self.env = NormalizeWrapper(self.env)
        self.env = ImgWrapper(self.env)  # to make the images from 160x120x3 into 3x160x120
        #self.env = ActionWrapper(self.env)
        self.env = DtRewardWrapper2(self.env)
        self.env = DiscreteWrapper(self.env)

        # Set seeds so we can reproduce our results
        self.env.seed(self.args.seed + self.identifier)
        torch.manual_seed(self.args.seed + self.identifier)

        self.local_net = Net(1, self.env.action_space.n)  # local network
        state = torch.tensor(preprocess_state(self.env.reset()))

        # bookkeeping
        start_time = last_disp_time = time.time()
        episode_length, epr, eploss, done = 0, 0, 0, True

        render_this_episode = self.args.render_env

        while self.info['frames'][0] <= self.args.max_steps:
            render_this_episode = self.args.graphical_output and (render_this_episode or (self.info['episodes'] % 10 == 0 and self.identifier == 0))

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
                np_action = action.numpy()[0]

                done = False
                for x in range(self.args.action_update_steps):
                    if done == False:
                        state, reward, done, _ = self.env.step(np_action)
                        reward += reward

                state = torch.tensor(preprocess_state(state))
                epr += reward
                #reward = np.clip(reward, -1, 1)
                done = done or episode_length >= self.args.max_episode_steps

                if render_this_episode:
                    self.env.render()
                    #print('Action: ', np_action)

                self.info['frames'].add_(1)
                num_frames = int(self.info['frames'].item())

                if done:  # update shared data
                    self.info['episodes'] += 1

                    # Moving average statistics:
                    # Linear interpolation between the current average and the new value
                    # Allows us to better estimate quality of results with high variance
                    interp_factor = 1 if self.info['episodes'][0] == 1 else 1 - 0.99
                    self.info['run_epr'].mul_(1 - interp_factor).add_(interp_factor * epr)
                    self.info['run_loss'].mul_(1 - interp_factor).add_(interp_factor * eploss)

                    elapsed = time.time() - start_time

                    with open(f"{self.log_dir}/performance-{self.name}.txt", "a") as myfile:
                        myfile.write(f"{self.info['episodes'].item():.0f} {num_frames} {epr} {self.info['run_loss'].item()} {elapsed}\n")
                    
                    if self.info['episodes'].item() % 1000 == 0 and self.args.save_models:
                        optimizer = CustomOptimizer.SharedAdam(self.global_net.parameters(), lr=self.args.learning_rate)
                        info = {k: torch.DoubleTensor([0]).share_memory_() for k in ['run_epr', 'run_loss', 'episodes', 'frames']}

                        torch.save({
                            'model_state_dict': self.global_net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'info': info
                        }, f"{self.ckpt_dir}/model-{self.name}-{self.info['episodes'].item()}")

                        print("Saved model to:",  f"{self.ckpt_dir}/model-{self.name}-{self.info['episodes'].item()}")

                # print training info every minute
                if self.identifier == 0 and time.time() - last_disp_time > 60:
                    elapsed = time.strftime("%Hh %Mm %Ss", time.gmtime(time.time() - start_time))
                    print(f"[time]: {elapsed}, [episodes]: {self.info['episodes'].item():.0f}, [frames]: {num_frames:.0f},"+
                        f"[mean epr]:{self.info['run_epr'].item():.2f}, [run loss]: {self.info['run_loss'].item():.2f}")
                
                    last_disp_time = time.time()

                # reset buffers / environment
                if done:
                    episode_length, epr, eploss = 0, 0, 0
                    state = torch.tensor(preprocess_state(self.env.reset()))

                values.append(value)
                log_probs.append(action_log_probs)
                actions.append(action)
                rewards.append(reward)

            # Reached sync step -> We need a terminal value
            # If the episode did not end use estimation of V(s) to bootstrap
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

def discount(x, gamma):
    from scipy.signal import lfilter
    return lfilter([1], [1, -gamma], x[::-1])[::-1]  # discounted rewards one liner
