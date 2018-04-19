import os
import math
import argparse

# QKFIX: Add the parent path to PYTHONPATH
import sys
sys.path.insert(0, 'tristan')
sys.path.append("..")

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision
#from datasets.dsprites import DSprites
from torchvision import datasets, transforms

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=50, metavar='N',
                    help='Number of epochs to train (default: 10)')
parser.add_argument('--beta', type=str, default='1',
                    help='Value for beta (default: 1)')
parser.add_argument('--softmax', type=float, default=0.0,
                    help='Sum of the betas (default: 0)')
parser.add_argument('--entropy', action='store_true', default=False,
                    help='Add a penalty on the entropy')
parser.add_argument('--obs', type=str, default='normal',
                    help='Type of the observation model (in [normal, bernoulli], '
                         'default: normal)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataset
data_transform = transforms.Compose([transforms.ToTensor()])
                                     #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), 'data/'),
                                     transform=data_transform)

# Data loader
data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)


def to_var(x, **kwargs):
    if args.cuda:
        x = x.cuda()
    return Variable(x, **kwargs)


def log_sum_exp(value):
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


output_folder = 'beta-vae'
if 'SLURM_JOB_ID' in os.environ:
    output_folder += '-{0}'.format(os.environ['SLURM_JOB_ID'])


# VAE model
class VAE(nn.Module):
    def __init__(self, x_dim=128, z_dim=20, output_dim=400):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.x_dim = x_dim
        self.output_dim = output_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),  # 64x60x80
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),  # 128x30x40
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),  # 128x15x20
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),  # 256x7x10
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 4),  # 512x4x7
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )
        self.conv_z = nn.Conv2d(512 * 4 * 7, self.output_dim, 1)

        self.encoder_mean = nn.Linear(self.output_dim, self.z_dim)
        self.encoder_logvar = nn.Sequential(
            nn.Linear(self.output_dim, self.z_dim),
            nn.Softplus())

        self.decoder = nn.Sequential(
            #nn.ConvTranspose2d(self.output_dim, 512, 1, 0, 0),
            #nn.BatchNorm2d(512),
            #nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4),  # 256x7x10
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, padding=(1, 1), output_padding=(1, 0)),  # 128x15x20
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 128x30x40
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        self.deconv_z = nn.ConvTranspose2d(self.z_dim, 512 * 4 * 7, 1, 1)

    def reparametrize(self, mu, log_var):
        """"z = mean + eps * sigma where eps is sampled from N(0, 1)."""
        eps = to_var(torch.randn(mu.size(0), mu.size(1)))
        z = mu + eps * torch.exp(0.5 * log_var) # 0.5 to convert var to std
        return z

    def forward(self, x):
        h = x.view(-1, 3, 120, 160)
        h = self.encoder(h)
        h = self.conv_z(h.view(x.size(0), -1, 1, 1)).squeeze()
        mu, log_var = self.encoder_mean(h), self.encoder_logvar(h)

        z = self.reparametrize(mu, log_var)
        z = z.view(h.size(0), self.z_dim, 1, 1)
        z = self.deconv_z(z)
        z = z.view(h.size(0), -1, 4, 7)
        logits = self.decoder(z)

        return logits, mu, log_var

    def sample(self, z):
        return self.decoder(z)

vae = VAE()
if args.cuda:
    vae.cuda()
parameters = list(vae.parameters())

if args.beta == 'learned':
    beta_ = to_var(torch.FloatTensor(vae.z_dim).uniform_(-1., 1.), requires_grad=True)
    parameters.append(beta_)

optimizer = torch.optim.Adam(parameters, lr=0.001)
iter_per_epoch = len(data_loader)

writer = SummaryWriter('./.logs/{0}'.format(output_folder))

# fixed inputs for debugging
fixed_x, _ = next(iter(data_loader))
fixed_grid = torchvision.utils.make_grid(fixed_x, normalize=True, scale_each=True)
writer.add_image('vae/original', fixed_grid, 0)
fixed_x = to_var(fixed_x)

for epoch in range(50):
    for i, (images, _) in enumerate(data_loader):

        images = to_var(images)
        logits, mu, log_var = vae(images)

        # Compute reconstruction loss and kl divergence
        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
        if args.obs == 'normal':
            # QKFIX: We assume here that the image is in B&W
            reconst_loss = F.mse_loss(F.sigmoid(logits), images, size_average=False)
        elif args.obs == 'bernoulli':
            reconst_loss = F.binary_cross_entropy_with_logits(logits, images, size_average=False)
        else:
            raise ValueError('Argument `obs` must be in [normal, bernoulli]')

        if args.beta == 'learned':
            if args.softmax:
                beta_norm_ = F.softmax(beta_)
                beta = args.softmax * vae.z_dim * beta_norm_
            else:
                beta = 1. + F.softplus(beta_)
            kl_divergence = torch.sum(0.5 * torch.matmul((mu ** 2 + torch.exp(log_var) - log_var - 1),
                beta.unsqueeze(1)))
        else:
            beta = int(args.beta)
            kl_divergence = torch.sum(0.5 * beta * (mu ** 2 + torch.exp(log_var) - log_var - 1))

        # Backprop + Optimize
        total_loss = reconst_loss + kl_divergence
        if args.beta == 'learned' and args.softmax and args.entropy:
            entropy = - torch.sum(beta_ * beta_norm_) + log_sum_exp(beta_)
            total_loss += math.log(vae.z_dim) - entropy
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        writer.add_scalar('loss', total_loss.data[0], epoch * iter_per_epoch + i)
        if args.beta == 'learned':
            writer.add_histogram('beta', beta.data, epoch * iter_per_epoch + i)

        if i % 100 == 0:
            print ("Epoch[%d/%d], Step [%d/%d], Total Loss: %.4f, "
                   "Reconst Loss: %.4f, KL Div: %.7f"
                   %(epoch + 1, 50, i + 1, iter_per_epoch, total_loss.data[0],
                     reconst_loss.data[0], kl_divergence.data[0]))

    # Save the reconstructed images
    reconst_logits, _, _ = vae(fixed_x)
    reconst_grid = torchvision.utils.make_grid(F.sigmoid(reconst_logits).data,
        normalize=True, scale_each=True)
    writer.add_image('vae/reconstruction', reconst_grid, epoch)

    # Save the checkpoint
    state = {
        'epoch': epoch + 1,
        'model': vae.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss': {
            'total': total_loss.data[0],
            'reconstruction': reconst_loss.data[0],
            'kl_divergence': kl_divergence.data[0]
        },
        'args': args,
        'beta': beta.data if args.beta == 'learned' else int(beta)
    }
    if not os.path.exists('./.saves/{0}'.format(output_folder)):
        os.makedirs('./.saves/{0}'.format(output_folder))
    torch.save(state, './.saves/{0}/beta-vae_{1:d}.ckpt'.format(output_folder, epoch + 1))