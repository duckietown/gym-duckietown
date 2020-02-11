import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class Model(nn.Module):
    def __init__(self, action_dim, max_action):
        super(Model, self).__init__()

        # ONLY TRU IN CASE OF DUCKIETOWN:
        flat_size = 32 * 9 * 14

        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sigm = nn.Sigmoid()

        self.conv1 = nn.Conv2d(3, 32, 8, stride=2)
        self.conv2 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv3 = nn.Conv2d(32, 32, 4, stride=2)
        self.conv4 = nn.Conv2d(32, 32, 4, stride=1)

        self.bn1 = nn.BatchNorm2d(32)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(32)
        self.bn4 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(.5)

        self.lin1 = nn.Linear(flat_size, 512)
        self.lin2 = nn.Linear(512, action_dim)

        self.max_action = max_action

    def forward(self, x):
        x = self.bn1(self.lr(self.conv1(x)))
        x = self.bn2(self.lr(self.conv2(x)))
        x = self.bn3(self.lr(self.conv3(x)))
        x = self.bn4(self.lr(self.conv4(x)))
        x = x.view(x.size(0), -1)  # flatten
        x = self.dropout(x)
        x = self.lr(self.lin1(x))

        # because we don't want our duckie to go backwards
        x = self.lin2(x)
        x[:, 0] = self.max_action * self.sigm(x[:, 0])  # because we don't want the duckie to go backwards
        x[:, 1] = self.tanh(x[:, 1])

        return x