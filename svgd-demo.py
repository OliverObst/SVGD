#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 25 14:47:16 2025

This script compares standard SVGD using an RBF kernel (white particles) 
with an RFF approximation (red particles) on a 3-component Gaussian mixture.
It profiles and visualises particles for each method, in a video.
Average per-step times for both kernels are printed at the end.

@author: oliver
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import time
import imageio

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Log-probability: mixture of three Gaussians 
def logprob(x):
    means = torch.tensor([[-2.0, -2.0], [0.0, 0.0], [2.0, 2.0]], device=device)
    sigmas = torch.tensor([0.5, 0.8, 0.2], device=device)
    inv_vars = 1.0 / (sigmas ** 2)
    pis = torch.ones(3, device=device) / 3.0

    diffs = x.unsqueeze(1) - means.unsqueeze(0)
    sq_dist = (diffs ** 2).sum(dim=2)
    exponents = -0.5 * sq_dist * inv_vars.unsqueeze(0)
    gauss = torch.exp(exponents) / (2 * torch.pi * sigmas.unsqueeze(0)**2)
    p = (pis.unsqueeze(0) * gauss).sum(dim=1)
    return torch.log(p + 1e-12)

# Autograd-based SVGD gradient (RBF kernel with median heuristic)
def kernel(x):
    x_det = x.detach()
    diff = x.unsqueeze(1) - x_det.unsqueeze(0)
    sq_dist = diff.pow(2).sum(dim=2)
    m = torch.median(sq_dist)
    h = torch.sqrt(0.5 * m / torch.log(torch.tensor(x.shape[0] + 1.0, device=device)))
    return torch.exp(-sq_dist / (2 * h**2))

def svgd_grad_rbf(x):
    x = x.detach().clone().requires_grad_(True)
    N = x.shape[0]
    K = kernel(x)
    lp = logprob(x).sum()
    dlogp = torch.autograd.grad(lp, x)[0]
    sum_K = K.sum()
    grad_K = torch.autograd.grad(sum_K, x)[0]
    return (K @ dlogp - grad_K) / N

# Random Fourier Features setup
M = 100  # number of random features
# initialize base particles to compute bandwidth
torch.manual_seed(42)
num_particles = 500
base = torch.rand(num_particles, 2, device=device) * 8.0 - 4.0
with torch.no_grad():
    d0 = base.unsqueeze(1) - base.unsqueeze(0)
    sq0 = d0.pow(2).sum(dim=2)
    m0 = sq0.median()
    h0 = torch.sqrt(0.5 * m0 / torch.log(torch.tensor(num_particles + 1.0, device=device)))

W = torch.randn(M, 2, device=device) / h0
b = 2 * np.pi * torch.rand(M, device=device)
two_over_M = (2.0 / M) ** 0.5

# RFF feature map
def rff_features(x):
    proj = x @ W.t() + b.unsqueeze(0)  # [N, M]
    return two_over_M * torch.cos(proj)

# SVGD gradient using RFF approximation
def svgd_grad_rff(x):
    x = x.detach().clone().requires_grad_(True)
    N = x.shape[0]
    lp = logprob(x).sum()
    dlogp = torch.autograd.grad(lp, x)[0]
    Z = rff_features(x)
    temp = Z.t() @ dlogp
    attr = Z @ temp
    s = Z.sum(dim=0)
    proj = x @ W.t() + b.unsqueeze(0)
    sinp = torch.sin(proj)
    grad_phi = -two_over_M * sinp.unsqueeze(2) * W.unsqueeze(0)
    rep = torch.einsum('k,nkd->nd', s, grad_phi)
    return (attr - rep) / N

# Initialize two particle sets
torch.manual_seed(42)
particles_rbf = base.clone()
particles_rff = base.clone()
times_rbf, times_rff = [], []

# Prepare background heatmap
grid_x = np.linspace(-4, 4, 200)
grid_y = np.linspace(-4, 4, 200)
xx, yy = np.meshgrid(grid_x, grid_y)
grid = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), device=device)
with torch.no_grad():
    p_vals = torch.exp(logprob(grid)).cpu().numpy().reshape(xx.shape)

# Plot setup
dpi = 100
fig, ax = plt.subplots(figsize=(6,6), dpi=dpi)
ax.imshow(p_vals, origin='lower', extent=[-4,4,-4,4], cmap='viridis', alpha=0.6)
scat_rbf = ax.scatter(particles_rbf[:,0].detach().cpu().numpy(), particles_rbf[:,1].detach().cpu().numpy(), c='white', s=20, label='RBF')
scat_rff = ax.scatter(particles_rff[:,0].detach().cpu().numpy(), particles_rff[:,1].detach().cpu().numpy(), c='red', s=20, label='RFF')
ax.legend(loc='upper right')
ax.set_xlim(-4,4)
ax.set_ylim(-4,4)

# Animation update
def update(frame):
    global particles_rbf, particles_rff
    t0 = time.perf_counter()
    phi_rbf = svgd_grad_rbf(particles_rbf)
    particles_rbf = particles_rbf + 0.1 * phi_rbf
    t1 = time.perf_counter()
    times_rbf.append(t1 - t0)

    t0 = time.perf_counter()
    phi_rff = svgd_grad_rff(particles_rff)
    particles_rff = particles_rff + 0.1 * phi_rff
    t1 = time.perf_counter()
    times_rff.append(t1 - t0)

    scat_rbf.set_offsets(particles_rbf.detach().cpu().numpy())
    scat_rff.set_offsets(particles_rff.detach().cpu().numpy())
    ax.set_title(f'Iter {frame+1}')
    return scat_rbf, scat_rff

# Animate and save
dpi = 100
ani = FuncAnimation(fig, update, frames=800, blit=True)
writer = FFMpegWriter(fps=30)
ani.save('svgd_compare_demo.mp4', writer=writer)
print('Saved svg_compare_demo.mp4')

if False:
    mp4_path = 'svgd_compare_demo.mp4'
    gif_path = 'svgd_compare_demo.gif'

    reader = imageio.get_reader(mp4_path)
    fps = reader.get_meta_data().get('fps', 30)
    writer = imageio.get_writer(gif_path, fps=fps)
    for frame in reader:
        writer.append_data(frame)
    reader.close()
    writer.close()
    print(f'Animated GIF saved as {gif_path}')

# Profiling results
print(f'Average RBF step time: {np.mean(times_rbf):.4f}s')
print(f'Average RFF step time: {np.mean(times_rff):.4f}s')
