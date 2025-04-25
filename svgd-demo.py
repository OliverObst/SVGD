#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:47:16 2025

@author: 30045063
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def logprob(x):
    """
    Example log-probability: mixture of three Gaussians along diagonal.
    x: Tensor [N, D]
    returns: Tensor [N]
    """
    means = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], device=device)
    sigmas = torch.tensor([0.5, 0.8, 0.2], device=device)
    inv_vars = 1.0 / (sigmas ** 2)
    pis = torch.ones(3, device=device) / 3.0

    # compute component densities
    # diffs: [N, 3, D]
    diffs = x.unsqueeze(1) - means.unsqueeze(0)
    sq_dist = (diffs ** 2).sum(dim=2)            # [N, 3]
    exponents = -0.5 * sq_dist * inv_vars.unsqueeze(0)
    gauss = torch.exp(exponents) / (2 * torch.pi * sigmas.unsqueeze(0)**2)
    weighted = pis.unsqueeze(0) * gauss           # [N,3]
    
    p = weighted.sum(dim=1)
    logp = torch.log(p + 1e-12)
    return logp

def kernel(x):
    """
    Anisotropic RBF kernel with per-dimension median heuristic bandwidths.
    x: Tensor [N, D], returns K: [N, N]
    """
    x_det = x.detach()
    # pairwise differences: (N, N, D)
    diff = x.unsqueeze(1) - x_det.unsqueeze(0)
    # squared distances per dimension: (N, N, D)
    sq_dist_dim = diff.pow(2)
    N, D = x.shape
    # flatten (N*N, D) and compute median per dimension
    sq_flat = sq_dist_dim.view(-1, D)
    m = torch.median(sq_flat, dim=0).values  # (D,)
    # per-dimension bandwidth h_d
    h = torch.sqrt(0.5 * m / torch.log(torch.tensor(N + 1.0, device=device)))  # (D,)
    # scale squared distances: diff^2 / (2 h_d^2)
    denom = 2 * h.pow(2)  # (D,)
    sq_scaled = sq_dist_dim / denom.view(1, 1, D)  # (N, N, D)
    # sum over dimensions to get anisotropic squared distances
    sq_dist = sq_scaled.sum(dim=2)  # (N, N)
    # kernel matrix
    return torch.exp(-sq_dist)  # (N,N))            # (N,N)


def svgd_grad(x, logprob, kernel):
    """
    Compute SVGD update via autograd.
    returns: phi [N, D]
    """
    x = x.detach().clone().requires_grad_(True)
    N, D = x.shape
    # kernel matrix
    K = kernel(x)  # [N, N]
    # log-prob gradient
    lp = logprob(x)           # [N]
    dlogp = torch.autograd.grad(lp.sum(), x)[0]  # [N, D]
    # repulsion: gradient of trace(K) w.r.t x
    # Compute gradient of sum K_ij for each i
    # Sum K over j to get s: [N], then grad_x (sum_j K_ij)
    sum_K = K.sum(dim=1)    # [N]
    # grad_K = grad_x (sum_j K_ij) = d(sum_K)/dx
    grad_K = torch.autograd.grad(sum_K.sum(), x)[0]  # [N, D]
    # SVGD phi
    phi = (K @ dlogp - grad_K) / N
    return phi

# Initialize particles
torch.manual_seed(23)
num_particles = 200
particles = torch.rand(num_particles, 2, device=device) * 8 - 4  # uniform [-4,4]

# Prepare background heatmap for visualization
grid_x = np.linspace(-4, 4, 200)
grid_y = np.linspace(-4, 4, 200)
xx, yy = np.meshgrid(grid_x, grid_y)
grid_points = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), device=device)
with torch.no_grad():
    logp_vals = logprob(grid_points)
    p_vals = torch.exp(logp_vals).cpu().numpy().reshape(xx.shape)

# Plot setup
dpi = 100
fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
ax.imshow(p_vals, origin='lower', extent=[-4, 4, -4, 4], cmap='viridis', alpha=0.6)
scat = ax.scatter(particles[:, 0].cpu(), particles[:, 1].cpu(), c='white', s=20)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

# Animation update
def update(frame):
    global particles
    phi = svgd_grad(particles, logprob, kernel)
    particles = particles + 0.1 * phi
    scat.set_offsets(particles.detach().numpy())
    ax.set_title(f'SVGD Iter {frame+1}')
    return scat,

ani = FuncAnimation(fig, update, frames=800, blit=True)
writer = FFMpegWriter(fps=30)
ani.save('svgd_autograd_demo.mp4', writer=writer)
print('Saved svgd_autograd_demo.mp4')

