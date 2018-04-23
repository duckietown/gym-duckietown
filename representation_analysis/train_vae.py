import argparse
import math
import os
import sys

#from torch.utils.data import DataLoader
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from representation_analysis.models import VAE
from representation_analysis.helpers import compute_total_correlation, compute_dim_wise_KL

sys.path.append(os.getcwd())

def log_sum_exp(value):
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                    help='Input batch size for training (default: 128)')
parser.add_argument('--num_steps', type=int, default=1718, metavar='M',
                    help='Number of steps to train (default: 1718)')
parser.add_argument('--beta', type=str, default='15',
                    help='Value for beta (default: 1)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Enables CUDA training')
parser.add_argument('--seed', type=int, default=7691, metavar='S',
                    help='Random seed (default: 7691)')
parser.add_argument('--output_folder', type=str, default='beta-vae', metavar='O',
                    help='Output folder (default beta-vae)')
parser.add_argument('--save_every', type=int, default=100, metavar='K',
                    help='Save after this many steps (default: 100)')
parser.add_argument('--state_size', type=int, default=100,
                    help='Size of latent code (default: 100)')
parser.add_argument('--saved_model', type=str, help='Save file to use')
parser.add_argument('--model_type', type=str, help='Model to train (beta-tcvae or vae)', default='beta-tcvae')

#'--saved_model representation_analysis/saves/beta-vae/beta-vae_300.ckpt'

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

# Dataset
data_transform = transforms.Compose([transforms.ToTensor()])
                                     #transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

train_dataset = datasets.ImageFolder(root=os.path.join(os.getcwd(), 'representation_analysis/data/'),
                                     transform=data_transform)

data_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)

if args.saved_model:
    try:
        loaded_state = torch.load(args.saved_model)
        step = loaded_state['step']
        model = loaded_state['model']
        vae = VAE(z_dim=args.state_size, use_cuda=args.cuda)
        vae.load_state_dict(model)
        optimizer_states = loaded_state['optimizer']

        total_losses = loaded_state['loss']['total']
        reconst_losses = loaded_state['loss']['reconstruction']
        kl_divergences = loaded_state['loss']['kl_divergence']
        TC_losses = loaded_state['loss']['TC_losses']
        args = loaded_state['args']
        beta = loaded_state['beta']
        fixed_x = loaded_state['fixed_x']

        parameters = list(vae.parameters())
        if args.cuda:
            vae.cuda()

        optimizer = torch.optim.Adam(parameters, lr=0.001)
        optimizer.load_state_dict(optimizer_states)

        print('model found and loaded successfully... resuming training from step {}'.format(step))
    except:
        print('problem loading model! Check model file!')
        exit(1)

else:
    print('creating new model ...')
    vae = VAE(z_dim=args.state_size, use_cuda=args.cuda)
    if args.cuda:
        vae.cuda()

    parameters = list(vae.parameters())

    if args.beta == 'learned':
        if not args.cuda:
            beta_ = Variable(torch.FloatTensor(vae.z_dim).uniform_(-1., 1.), requires_grad=True)
        else:
            beta = Variable(torch.FloatTensor(vae.z_dim).uniform_(-1., 1.).cuda(), requires_grad=True)
        parameters.append(beta_)

    optimizer = torch.optim.Adam(parameters, lr=5e-4)

    total_losses = []
    reconst_losses = []
    kl_divergences = []
    TC_losses = []

    # fixed inputs for debugging
    fixed_x, _ = next(iter(data_loader))
    # fixed_x = next(iter(data_loader))
    fixed_grid = torchvision.utils.make_grid(fixed_x)
    if not args.cuda:
        fixed_x = Variable(fixed_x)
    else:
        fixed_x = Variable(fixed_x.cuda())
    torchvision.utils.save_image(fixed_grid, 'representation_analysis/reconstructions/{0}/original.jpg'
                                 .format(args.output_folder))
    step = 0

#  train
for i, (images, _) in enumerate(data_loader, start=step):
    try:
        if not args.cuda:
            images = Variable(images)
        else:
            images = Variable(images.cuda())

        logits, mu, log_var, z = vae(images)

        # Compute reconstruction loss and kl divergence
        # For kl_divergence, see Appendix B in the paper or http://yunjey47.tistory.com/43
        reconst_loss = F.mse_loss(F.sigmoid(logits), images, size_average=False)
        reconst_loss /= args.batch_size

        if args.model_type == 'vae':
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

            kl_divergence /= args.batch_size

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

        if args.model_type == 'beta-tcvae':
            kl_divergence = torch.sum(0.5 * 1 * (mu ** 2 + torch.exp(log_var) - log_var - 1))
            kl_divergence /= args.batch_size
            #print('KL divergence: {:.4f}'.format(kl_divergence.data[0]))

            TC = compute_total_correlation(len(data_loader)*args.batch_size, mu, log_var, z)
            rest = kl_divergence - TC

            #dim_wise_KL = compute_dim_wise_KL(len(data_loader)*args.batch_size, mu, log_var, z)
            #index_code_MI = kl_divergence - TC - dim_wise_KL
            #print('TC={:.4f} dim_wise_KL={:.4f} index_code_MI={:.4f}'.format(TC.data[0], dim_wise_KL.data[0], index_code_MI.data[0]))

            # Backprop + Optimize
            beta = int(args.beta)
            total_loss = reconst_loss + (beta-1)*TC
            print('Total Loss={:.4f} TC={:.4f} Reconstruction Loss={:.4f}'
                  .format(total_loss.data[0], TC.data[0], reconst_loss.data[0]))
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            total_losses.append(total_loss.data[0])
            reconst_losses.append(reconst_loss.data[0])
            kl_divergences.append(kl_divergence.data[0])
            TC_losses.append(TC.data[0])

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
                'beta': beta.data if args.beta == 'learned' else int(beta),
                'fixed_x': fixed_x
            }
            if not os.path.exists('representation_analysis/saves/{0}'.format(args.output_folder)):
                os.makedirs('representation_analysis/saves/{0}'.format(args.output_folder))
            torch.save(state, 'representation_analysis/saves/{0}/beta-vae_{1:d}.ckpt'.format(args.output_folder, i + 1))
            print('Got to step {}. Checkpoint saved!'.format(i))

            # Save the reconstructed images
            reconst_logits, _, _, _ = vae(fixed_x)
            reconst_grid = torchvision.utils.make_grid(F.sigmoid(reconst_logits).data)
            if not os.path.exists('representation_analysis/reconstructions/{0}'.format(args.output_folder)):
                os.makedirs('representation_analysis/reconstructions/{0}'.format(args.output_folder))
            torchvision.utils.save_image(reconst_grid, 'representation_analysis/reconstructions/{0}/{1}.jpg'
                                         .format(args.output_folder, i))

    except KeyboardInterrupt:
        print('Training stopped! Saving progress ...')
        # Save the checkpoint
        state = {
            'step': i,
            'model': vae.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': {
                'total': total_losses,
                'reconstruction': reconst_losses,
                'kl_divergence': kl_divergences,
                'TC_losses': TC_losses
            },
            'args': args,
            'beta': beta.data if args.beta == 'learned' else int(beta),
            'fixed_x': fixed_x
        }
        if not os.path.exists('representation_analysis/saves/{0}'.format(args.output_folder)):
            os.makedirs('representation_analysis/saves/{0}'.format(args.output_folder))
        torch.save(state, 'representation_analysis/saves/{0}/beta-vae_{1:d}.ckpt'.format(args.output_folder, i + 1))
        print('Got to step {}. Checkpoint saved!'.format(i))

        # Save the reconstructed images
        reconst_logits, _, _, _ = vae(fixed_x)
        reconst_grid = torchvision.utils.make_grid(F.sigmoid(reconst_logits).data)
        if not os.path.exists('representation_analysis/reconstructions/{0}'.format(args.output_folder)):
            os.makedirs('representation_analysis/reconstructions/{0}'.format(args.output_folder))
        torchvision.utils.save_image(reconst_grid, 'representation_analysis/reconstructions/{0}/{1}.jpg'
                                     .format(args.output_folder, i))
        break
