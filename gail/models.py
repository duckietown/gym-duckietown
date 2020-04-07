import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models

from torch.distributions import LogNormal, Normal, MultivariateNormal, Independent
from functools import reduce

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class Generator(nn.Module):
    def __init__(self,observation_dim, action_dim, use_VAE=False):
        super(Generator,self).__init__()
        
        self.use_VAE = use_VAE
        self.lr = nn.LeakyReLU()
        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.softmax = nn.Softmax()
        self.flatten = Flatten()
        if self.use_VAE:
            self.encoder = Encoder(observation_dim)
            state_dict = torch.load('duckietown/vae-encoder')
            self.encoder.load_state_dict(state_dict)
            for param in self.encoder.parameters():
                param.requires_grad = False
        else:
            self.encoder =  models.resnet50(pretrained=True)
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder = nn.Sequential(self.encoder,
                                        self.lr,
                                        nn.Linear(1000,256),
                                        self.lr)
        self.lin1 = nn.Linear(256, 256)

        self.mu_head = nn.Linear(256, action_dim)
        self.sig_head = nn.Linear(256, action_dim)
        self.value_head = nn.Linear(256, 1)


        if reduce(lambda x,y: x*y, observation_dim) == 4:
            self.encoder = nn.Sequential(nn.Linear(4,4),
                                    nn.LeakyReLU(),
                                    nn.Linear(4,4),
                                    nn.LeakyReLU())

            self.mu_head = nn.Linear(4, action_dim)
            self.sig_head = nn.Linear(4, action_dim)


            self.value_head = nn.Linear(4, 1)

    def forward(self,x):

        x = self.lr(self.encoder(x))
        x = self.flatten(x)
        x = self.lr(self.lin1(x))

        value = self.lr(self.value_head(x))
        mu = self.mu_head(x)
        sig = abs(self.sig_head(x)) + 1e-10
        dist = Normal(*[mu, sig])

        return dist, value

    def sample_action(self, observation):
        dist, _ = self.forward(observation)

        actions = dist.rsample()

        return actions
    
    def get_means(self, observation):
        dist, value = self.forward(observation)

        return dist.mean


class Discriminator(nn.Module):
    def __init__(self, observation_dim, action_dim, use_VAE=False):
        super(Discriminator,self).__init__()
        self.use_VAE = bool(use_VAE)
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.lr = nn.LeakyReLU()
        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()

        self.flatten = Flatten()
        self.softmax = nn.Softmax()

        if reduce(lambda x,y: x*y, observation_dim) == 4:
            self.encoder = nn.Sequential(nn.Linear(4,4),
                                    nn.LeakyReLU(),
                                    nn.Linear(4,4),
                                    nn.LeakyReLU())
            self.lin1 = nn.Linear(4+action_dim, 2)
            self.lin2 = nn.Linear(2,1)

        else:
            if self.use_VAE:
                self.encoder = Encoder(observation_dim)
                state_dict = torch.load('duckietown/vae-encoder')
                self.encoder.load_state_dict(state_dict)
                for param in self.encoder.parameters():
                    param.requires_grad = False
            else:
                self.encoder =  models.resnet50(pretrained=True)
                for param in self.encoder.parameters():
                    param.requires_grad = False
                self.encoder = nn.Sequential(self.encoder,
                                            self.lr,
                                            nn.Linear(1000,40),
                                            self.lr)

            self.lin1 = nn.Linear(40+action_dim,128)
            self.lin2 = nn.Linear(128, 1)

        
    def forward(self,observations, actions):

        x = self.encoder(observations)
        x = torch.cat((x,actions),1)
        x = self.lr(self.lin1(x))
        x = self.sig(self.lin2(x))
        return x

class Encoder(nn.Module):
    def __init__(self, observation_dim):
        super(Encoder,self).__init__()
        self.flatten = Flatten()
        self.lr = nn.LeakyReLU()
        self.fc = nn.Sequential(self.flatten,
                                nn.Linear(reduce(lambda x,y: x*y, observation_dim), 1024),
                                self.lr,
                                nn.Linear(1024,512),
                                self.lr,
                                )
        self.mu_head = nn.Linear(512,256)
        self.sig_head = nn.Linear(512,256)
        
    def forward(self, x):
        mu = self.mu_head(self.fc(x))
        sig = abs(self.sig_head(self.fc(x))) + 1e-10

        dist = Normal(*[mu, sig])
        sampled = dist.rsample()

        return sampled

class Decoder(nn.Module):
    def __init__(self, observation_dim):
        super(Decoder,self).__init__()
        self.flatten = Flatten()
        self.lr = nn.LeakyReLU()
        self.fc = nn.Sequential(nn.Linear(256,512),
                                self.lr,
                                nn.Linear(512,1024),
                                self.lr,
                                nn.Linear(1024,reduce(lambda x,y: x*y, observation_dim)),
                                )
        
    def forward(self, x):
        return(self.fc(x))

if __name__ == "__main__":

    pass