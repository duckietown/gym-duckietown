import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.distributions import LogNormal, Normal

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self, action_dim, deterministic=True):
        super(Generator,self).__init__()
        
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()

        self.is_deterministic=True

        self.conv1 = nn.Sequential( nn.Conv2d(3, 32, 7, stride=4, padding=3),
                                    self.lr,
                                    nn.Conv2d(32,32, 5, stride=4, padding=2),
                                    self.lr,
                                    nn.Conv2d(32,32, 3, stride=2, padding=1)
                                    # self.lr()
                                    )
        # self.conv1 = nn.Conv2d(3, 32, 7, stride=4, padding=3)
        # self.conv2 = nn.Conv2d(32,32, 5, stride=4, padding=2)
        # self.conv3 = nn.Conv2d(32,32, 3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(3,32, 20, stride=30)   

        self.conv5 = nn.Conv2d(2112, 32, 3, stride=1, padding=1)   

        self.flatten = Flatten()
        self.dropout = nn.Dropout(.5)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.resnet.parameters():
            param.requires_grad = False
          
        self.lin1 = nn.Linear(32*5*4, 256)
        self.lin2 = nn.Linear(256, 128)

        self.mu_v_head = nn.Linear(128,1)
        self.mu_s_head = nn.Linear(128,1)
        self.sig_v_head = nn.Linear(128,1)
        self.sig_s_head = nn.Linear(128,1)
        
        for p in self.mu_s_head.parameters():
            p.data.clamp_(-0.01,0.01)

        for p in self.sig_s_head.parameters():
            p.data.clamp_(-0.01,0.01)

    def forward(self,x):
        r = self.lr(self.resnet(x))

        c = self.lr(self.conv1(x))
        # c = self.lr(self.conv2(c))
        # c = self.lr(self.conv3(c))

        f = self.lr(self.conv4(x))

        x = torch.cat((r,c,f),1)

        x = self.lr(self.conv5(x))
        x = self.dropout(self.flatten(x))

        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(x))
        
        mu_v = self.mu_v_head(x)
        sig_v = self.sig_v_head(x)

        mu_s = self.mu_s_head(x)
        sig_s = self.sig_s_head(x)

        return mu_v, sig_v, mu_s, sig_s


    def select_action(self, observation):
        mu_v, sig_v, mu_s, sig_s = self.forward(observation)

        if self.is_deterministic:
            model_actions = torch.cat((mu_v,mu_s),1)
        else:
            velocity_dist = Normal(mu_v, sig_v)
            steer_dist = Normal(mu_s, sig_s)
            velocities = Variable(velocity_dist.sample(), requires_grad=True)
            steering_angles = Variable(steer_dist.sample(), requires_grad=True)

            model_actions = torch.cat((velocities,steering_angles),1)

        return model_actions

class Discriminator(nn.Module):
    def __init__(self, action_dim):
        super(Discriminator,self).__init__()
        
        self.lr = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.conv1 = nn.Conv2d(3, 32, 3, stride=4, padding=2)
        self.conv2 = nn.Conv2d(32,64, 3, stride=4, padding=2)
        self.conv3 = nn.Conv2d(64,128,3, stride=4, padding=2)

        self.flatten = Flatten()
        self.softmax = nn.Softmax()
        # self.resnet = models.resnet50(pretrained=True)
        # for param in self.resnet.parameters():
        #     param.requires_grad = False
        # self.resnet.fc = nn.Linear(2048,1000)
        
        self.lin1 = nn.Linear(128*4*3+action_dim, 256)
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
        x = self.sig(self.lin3(x))
        # x = self.lr(self.lin3(x))
        # x = self.lin3(x)

        return x
    
class Value(nn.Module):
    def __init__(self, action_dim):
        super(Value,self).__init__()
        
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()


        self.conv1 = nn.Conv2d(3, 32, 7, stride=4, padding=3)
        self.conv2 = nn.Conv2d(32,32, 5, stride=4, padding=2)
        self.conv3 = nn.Conv2d(32,32, 3, stride=2, padding=1)

        self.conv4 = nn.Conv2d(3,32, 20, stride=30)   

        self.conv5 = nn.Conv2d(2112, 32, 3, stride=1, padding=1)   

        self.flatten = Flatten()
        self.dropout = nn.Dropout(.5)
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-2])

        for param in self.resnet.parameters():
            param.requires_grad = False
          
        self.lin1 = nn.Linear(32*5*4, 256)
        self.lin2 = nn.Linear(256, 128)
        self.lin3 = nn.Linear(128,1)
        # self.lin4 = nn.Linear(128,1)

    def forward(self,x):
        r = self.lr(self.resnet(x))

        c = self.lr(self.conv1(x))
        c = self.lr(self.conv2(c))
        c = self.lr(self.conv3(c))

        f = self.lr(self.conv4(x))

        x = torch.cat((r,c,f),1)

        x = self.lr(self.conv5(x))
        x = self.dropout(self.flatten(x))

        x = self.lr(self.lin1(x))
        x = self.lr(self.lin2(x))
        x = self.lr(self.lin3(x))

        return x


if __name__ == "__main__":

    pass