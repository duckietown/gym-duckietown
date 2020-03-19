import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self, action_dim):
        super(Generator,self).__init__()
        
        self.flat_size = 144768 + 1000 #only for ducky town!!
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=1)
        self.conv2 = nn.Conv2d(32,32, 3, stride=2)

        self.flatten = Flatten()
        
        self.resnet50 = models.resnet50(pretrained=True)
        
        self.lin1 = nn.Linear(self.flat_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, action_dim)
        
    def forward(self,x):
        y = self.resnet50(x)
        y = self.flatten(y)

        x = self.tanh(self.conv1(x))
        x = self.tanh(self.conv2(x))
        x = self.flatten(x)
                
        x = torch.cat((x,y),1)
        
        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)

        return x

class Discriminator(nn.Module):
    def __init__(self, action_dim):
        super(Discriminator,self).__init__()
        
        self.flat_size = 34048 +action_dim #only for ducky town!!
        self.lr = nn.LeakyReLU()
        
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2)
        self.conv2 = nn.Conv2d(32,64, 3, stride=2)
        self.conv3 = nn.Conv2d(64,128,3, stride=2)

        self.flatten = Flatten()
        
#         self.resnet50 = models.resnet50(pretrained=True)
        
        self.lin1 = nn.Linear(self.flat_size, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128, 1)
        
    def forward(self,observations, actions):
        x = self.lr(self.conv1(observations))
        x = self.lr(self.conv2(x))
        x = self.lr(self.conv3(x))
        x = self.flatten(x)
        
        x = torch.cat((x,actions),1)
        
        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(x))
        x = self.lr(self.lin3(x))

        
        return x
    