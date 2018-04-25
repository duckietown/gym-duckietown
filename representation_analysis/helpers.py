import torch
from torch import distributions
import numpy as np
import pyro.distributions as dist
import pyro
from torch.autograd import Variable


def log_sum_exp(value):
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


def logsumexp(inputs, dim=None, keepdim=False):
    """Numerically stable logsumexp.

    Args:
        inputs: A Variable with any shape.
        dim: An integer.
        keepdim: A boolean.

    Returns:
        Equivalent of log(sum(exp(inputs), dim=dim, keepdim=keepdim)).
    """
    # For a 1-D array x (any array along a single dimension),
    # log sum exp(x) = s + log sum exp(x - s)
    # with s = max(x) being a common choice.
    # taken from somewhere
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs



def compute_total_correlation(N, mu, log_var, z):
    M = z.shape[0]
    sample = []
    for i in range(M):
        sample.append(dist.MultivariateNormal(loc=mu[i], covariance_matrix=torch.exp(0.5 * log_var)[i].diag()).sample())
    sample = torch.stack(sample, dim=0)[:, -1, :]
    q_z = distributions.Normal(mean=mu, std=torch.exp(0.5*log_var))
    total_array = []
    for i in range(M):
        total_array.append(q_z.log_prob(sample[i]))
    q_mat = torch.stack(total_array, dim=0)

    H = - (logsumexp(q_mat.sum(dim=2), dim=0) - np.log(N * M)).mean()
    CE = - (logsumexp(q_mat, dim=1).sum(dim=1) - np.log(N * M)).mean()
    #print('entropy: {}'.format(H.data[0]))
    #print('cross entropy: {}'.format(CE.data[0]))
    #print('log_var: {}'.format(log_var.data[0]))
    return - (CE - H)

def compute_dim_wise_entropy(N, mu, log_var, z):
    #M = z.shape[0]
    #K = z.shape[1]
    #estimate = 0
    #q_z = distributions.Normal(mean=mu, std=torch.exp(0.5 * log_var))
    #for i in range(M):
    #    q_z_given_x = q_z.log_prob(z[i])
    #    for k in range(K):
    #        estimate += log_sum_exp(q_z_given_x[:, k]) - np.log(M * N)
    #return estimate/M

    M = z.shape[0]
    K = z.shape[1]
    q_z = distributions.Normal(mean=mu, std=torch.exp(0.5 * log_var))
    estimate = 0
    for k in range(K):
        for i in range(M):
            q_z_given_x = q_z.log_prob(z[i])[:, k]
            estimate += log_sum_exp(q_z_given_x)
        estimate -= np.log(N * M)
    return estimate/M


def compute_dim_wise_ce(N, z):
    M = z.shape[0]
    K = z.shape[1]
    q_z = distributions.Normal(mean=0, std=1)
    estimate = 0
    for k in range(K):
        q_z_given_x = q_z.log_prob(z)
        estimate += log_sum_exp(q_z_given_x)
        estimate -= np.log(N * M)

    #M = z.shape[0]
    #estimate = 0
    #q_z = distributions.Normal(mean=0, std=1)
    #for i in range(M):
    #    q_z_given_x = q_z.log_prob(z[i])
    #    estimate += log_sum_exp(q_z_given_x) - np.log(M * N)
    return estimate/M


def compute_dim_wise_KL(N, mu, log_var, z):
    estimate = compute_dim_wise_entropy(N, mu, log_var, z) - compute_dim_wise_ce(N, z)
    return estimate

def save_curve(total_losses, TC_losses):
    from matplotlib import pyplot as plt
    plt.plot(total_losses, color='red', label='Total Loss')
    plt.plot(TC_losses, color='green', label='TC Loss')
    plt.savefig('representation_analysis/results/training_curve')
