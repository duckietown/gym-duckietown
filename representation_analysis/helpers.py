import torch
from torch import distributions
import numpy as np

def log_sum_exp(value):
    m = torch.max(value)
    sum_exp = torch.sum(torch.exp(value - m))
    return m + torch.log(sum_exp)


def minibatch_importance_sample(N, mu, log_var, z):
    M = z.shape[0]
    q_z = distributions.Normal(mean=mu, std=torch.exp(0.5*log_var))
    estimate = 0
    for i in range(M):
        q_z_given_x = q_z.log_prob(z[i]).sum(1)
        q_z_given_x = log_sum_exp(q_z_given_x) - np.log(M * N)
        estimate += q_z_given_x
    return estimate/M


def compute_cross_entropy_minibatch_IS(N, mu, log_var, z):
    M = z.shape[0]
    q_z = distributions.Normal(mean=mu, std=torch.exp(0.5*log_var))
    estimate = 0
    for i in range(M):
        q_z_given_x = q_z.log_prob(z[i]).sum(0)
        q_z_given_x = log_sum_exp(q_z_given_x) - np.log(M * N)
        estimate += q_z_given_x
    return - estimate/M


def compute_total_correlation(N, mu, log_var, z):
    entropy_q = minibatch_importance_sample(N, mu, log_var, z)
    cross_entropy_q_factorized = compute_cross_entropy_minibatch_IS(N, mu, log_var, z)
    return (cross_entropy_q_factorized - entropy_q)/z.shape[0]


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
