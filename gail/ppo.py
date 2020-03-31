import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class PPO():
    '''
    referenced from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail/blob/master/a2c_ppo_acktr/algo/ppo.py
    '''
    def __init__(self,
                 generator_network,
                 value_network,
                 clip_param,
                 ppo_epoch,
                 num_mini_batch,
                 value_loss_coef,
                 entropy_coef,
                 lr=None,
                 eps=None,
                 max_grad_norm=Norm,
                 use_clipped_value_train=True
                 ):
        self.generator = generator_network
        self.value_network = value_network
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

        self.G_optimizer = optim.Adam(generator_network.parameters(), lr=lr, eps=eps)
        self.V_optimizer = optim.Adam(value_network.parameters(), lr=lr, eps=eps)

    def update(self, rollouts):
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        
        value_loss_epoch = 0
        action_loss_epoch = 0
        dist_entropy_epoch = 0

        for e in range(self.ppo_epoch):
            values, action_log_probs, dist = 0