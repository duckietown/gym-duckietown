import argparse
import math
import os

from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision

from representation_analysis.DuckieDataset import DuckieDataset
from representation_analysis.models import VAE


def log_sum_exp(value):
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)

parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--num_steps', type=int, default=1000, metavar='M',
                    help='Number of steps to train (default: 1000)')
parser.add_argument('--beta', type=str, default='1',
                    help='Value for beta (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
parser.add_argument('--output_folder', type=str, default='beta-vae', metavar='O',
                    help='Output folder (default beta-vae)')
parser.add_argument('--save_every', type=int, default=1, metavar='K',
                    help='Save after this many steps (default: 100)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_dataset = DuckieDataset(args.num_steps*args.batch_size)
data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

vae = VAE(use_cuda=args.cuda)
if args.cuda:
    vae.cuda()
parameters = list(vae.parameters())

if args.beta == 'learned':
    if not args.cuda:
        beta_ = Variable(torch.FloatTensor(vae.z_dim).uniform_(-1., 1.), requires_grad=True)
    else:
        beta = Variable(torch.FloatTensor(vae.z_dim).uniform_(-1., 1.).cuda(), requires_grad=True)
    parameters.append(beta_)

optimizer = torch.optim.Adam(parameters, lr=0.001)

total_losses = []
reconst_losses = []
kl_divergences = []

# fixed inputs for debugging
fixed_x = next(iter(data_loader))
fixed_grid = torchvision.utils.make_grid(fixed_x)
if not args.cuda:
    fixed_x = Variable(fixed_x)
else:
    fixed_x = Variable(fixed_x.cuda())

#  train
for i, images in enumerate(data_loader):
    if not args.cuda:
        images = Variable(images)
    else:
        images = Variable(images.cuda())

    logits, mu, log_var = vae(images)

    # Compute reconstruction loss and kl divergence
    # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
    # QKFIX: We assume here that the image is in B&W
    reconst_loss = F.mse_loss(F.sigmoid(logits), images, size_average=False)

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

    total_losses.append(total_loss.data[0])
    reconst_losses.append(reconst_loss.data[0])
    kl_divergences.append(kl_divergence.data[0])

    print("Step {}/{}, Total Loss: {:.4f}, "
          "Reconst Loss: {:.4f}, KL Div: {:.7f}"
          .format(i + 1, args.num_steps, total_loss.data[0],
                  reconst_loss.data[0], kl_divergence.data[0]))

    if (i+1) % args.save_every == 0:
        # Save the checkpoint
        state = {
            'step': i,
            'model': vae.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': {
                'total': total_losses,
                'reconstruction': reconst_losses,
                'kl_divergence': kl_divergences
            },
            'args': args,
            'beta': beta.data if args.beta == 'learned' else int(beta)
        }
        if not os.path.exists('representation_analysis/saves/{0}'.format(args.output_folder)):
            os.makedirs('representation_analysis/saves/{0}'.format(args.output_folder))
        torch.save(state, 'representation_analysis/saves/{0}/beta-vae_{1:d}.ckpt'.format(args.output_folder, i + 1))
        print('Got to step {}. Checkpoint saved!'.format(i))

        # Save the reconstructed images
        reconst_logits, _, _ = vae(fixed_x)
        reconst_grid = torchvision.utils.make_grid(F.sigmoid(reconst_logits).data)
        if not os.path.exists('representation_analysis/reconstructions/{0}'.format(args.output_folder)):
            os.makedirs('representation_analysis/reconstructions/{0}'.format(args.output_folder))
        torchvision.utils.save_image(reconst_grid, 'representation_analysis/reconstructions/{0}/{1}.jpg'
                                     .format(args.output_folder, i))
