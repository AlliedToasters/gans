import numpy as np
import torch

def get_random_vecs(batch_size, latent_dim):
    """Returns randomly generated latent vectors"""
    rand = np.random.randn(batch_size, latent_dim)
    norms = np.linalg.norm(rand, axis=1).reshape(-1, 1)
    values = rand/norms
    shaped = values.reshape(batch_size, latent_dim)
    return torch.tensor(shaped).float()

def get_interp_batch(q1, q2, steps=100):
    """
    gets a batch of latent vectors corresponding to
    SLERP interpolation between two vectors q1, q2
    """
    mus = torch.linspace(0, 1, steps=steps)
    vecs = [interp(q1, q2, mu).reshape(1, -1) for mu in mus]
    vecs = torch.cat(vecs, dim=0)
    return vecs

def interp(q1, q2, mu):
    """
    Performs SLERP interpolation between vecs q1, q2
    at position 0 < mu < 1 along the great circle connecting
    the two points.
    """
    dot = torch.dot(q1, q2)
    theta = torch.acos(dot)
    t1 = ((torch.sin((torch.tensor(1).float() - mu)*theta))/torch.sin(theta))*q1
    t2 = (torch.sin(mu * theta)/torch.sin(theta)) * q2
    return t1 + t2

vecs = get_random_vecs(64, 2)
