import torch.nn as nn
import torch
from torch.autograd import Variable


# VAE model
class VAE(nn.Module):
    def __init__(self, x_dim=128, z_dim=20, output_dim=400, use_cuda=True):
        super(VAE, self).__init__()

        self.z_dim = z_dim
        self.x_dim = x_dim
        self.output_dim = output_dim
        self.use_cuda = use_cuda

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
        eps = torch.randn(mu.size(0), mu.size(1))
        if not self.use_cuda:
            eps = Variable(eps)
        else:
            eps = Variable(eps.cuda())
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