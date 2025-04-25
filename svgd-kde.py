#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 26 08:19:44 2025

This script computes SVGD using an RBF kernel using a 
kernel density estimator (instead of autograd) 
on a 3-component Gaussian mixture, pretending we don't know 
the log p.

@author: oliver
"""
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mixture parameters (unknown to estimator)
means = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], device=device)
sigmas = torch.tensor([0.5, 0.8, 0.2], device=device)
pi = torch.tensor([1/3, 1/3, 1/3], device=device)

# Draw samples from the mixture (used only by KDE)
def sample_mixture(n):
    comps = torch.multinomial(pi, n, replacement=True)
    eps = torch.randn(n, 2, device=device)
    return means[comps] + eps * sigmas[comps].unsqueeze(1)

# Pre-sample data for KDE
data = sample_mixture(2000)  # samples for density estimation

# RBF kernel and repulsion gradient
def rbf_kernel(x):
    N, D = x.shape
    x2 = (x**2).sum(1, keepdim=True)
    sq = x2 + x2.T - 2*(x @ x.T)
    med = torch.median(sq)
    h = torch.sqrt(0.5 * med / torch.log(torch.tensor(N+1.0, device=device)))
    K = torch.exp(-sq/(2*h*h))
    sumK = K.sum(1, keepdim=True)
    gradK = 2*(x*sumK - K @ x)/(h*h)
    return K, gradK

# KDE-based score estimator (attraction term)
def kde_score(x):
    N, D = x.shape
    M = data.shape[0]
    x2 = (x**2).sum(1, keepdim=True)
    d2 = x2 + (data**2).sum(1) - 2*(x @ data.T)
    with torch.no_grad():
        dd2 = (data**2).sum(1, keepdim=True) + (data**2).sum(1) - 2*(data @ data.T)
        med = torch.median(dd2)
        h = torch.sqrt(0.5 * med / torch.log(torch.tensor(M+1.0, device=device)))
    Kd = torch.exp(-d2/(2*h*h))
    sumKd = Kd.sum(1, keepdim=True)
    diff = data.unsqueeze(0) - x.unsqueeze(1)
    grad_p = (Kd.unsqueeze(2) * diff).sum(1) / (h*h)
    score = grad_p / (sumKd + 1e-9)
    return score

# Single SVGD update combining attraction and repulsion
def svgd_step(particles, lr=0.1, repulsion_coeff=0.5):
    score = kde_score(particles)
    K, gradK = rbf_kernel(particles)
    phi = (K @ score + repulsion_coeff * gradK) / particles.shape[0]
    return (particles + lr * phi).detach()

# Initialize inference particles
torch.manual_seed(0)
N = 200
particles = torch.rand(N, 2, device=device) * 8.0 - 4.0  # uniform in [-4,4]^2

# True density heatmap for visualization
grid_x = torch.linspace(-4, 4, 200, device=device)
grid_y = torch.linspace(-4, 4, 200, device=device)
xx, yy = torch.meshgrid(grid_x, grid_y, indexing='xy')
grid = torch.stack([xx.ravel(), yy.ravel()], dim=1)
with torch.no_grad():
    diffs = grid.unsqueeze(1) - means.unsqueeze(0)
    sq = (diffs**2).sum(2)
    gauss = torch.exp(-0.5 * sq / (sigmas.unsqueeze(0)**2)) / (2 * torch.pi * sigmas.unsqueeze(0)**2)
    true_p = (pi.unsqueeze(0) * gauss).sum(1)
p_vals = true_p.view(200, 200).cpu().numpy()

# Plot setup
fig, ax = plt.subplots(figsize=(6, 6))
ax.imshow(p_vals, origin='lower', extent=[-4, 4, -4, 4], cmap='viridis', alpha=0.6)
scat = ax.scatter(particles[:, 0].cpu(), particles[:, 1].cpu(), c='white', s=20)
ax.set_xlim(-4, 4)
ax.set_ylim(-4, 4)

def update(frame):
    global particles
    particles = svgd_step(particles)
    scat.set_offsets(particles.cpu().numpy())
    ax.set_title(f'SVGD KDE-based on mixture samples (Frame {frame+1})')
    return scat,

# Animate and save
dpi = 100
ani = FuncAnimation(fig, update, frames=800, blit=True)
writer = FFMpegWriter(fps=30)
ani.save('svgd_kde_demo.mp4', writer=writer)
print('Saved svgd_kde_demo.mp4')

