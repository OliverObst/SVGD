# SVGD Exploration Repository

Experiments and demonstrations of Stein Variational Gradient Descent (SVGD).

See also: [kimbente/svgd](https://github.com/kimbente/svgd)  
for a more polished SVGD repository.

## structure

- **`svgd-demo.py`**  
  Compares two SVGD variants:
  - Exact RBF kernel (white particles)
  - Random Fourier Features (red particles)
  - Profiles per-step runtime and saves an MP4.

- **`svgd-kde.py`**  
  SVGD without autograd:
  - RBF kernel
  - KDE-based estimation of log p
  - Less dreadful than expected.

## features

- Adaptive RBF kernel width via median heuristic.
- Random Fourier Features for scalable kernel approximation.
- Both autograd and manual (KDE) options.

## prerequisites

- Python 3.8+
- [PyTorch](https://pytorch.org/)
- [Matplotlib](https://matplotlib.org/)
- [Imageio](https://imageio.github.io/) (for MP4/GIF saving)

### Installation

```bash
pip install torch matplotlib imageio
```

### Running the Demos

1. **Compare kernels:**
   ```bash
   python svgd-demo.py
   ```
   Generates `svgd_compare_demo.mp4`, prints per-step timings.
   
2. **SVGD with KDE:**
   ```bash
   python svgd-kde.py
   ```
   Generates `svgd_kde_demo.mp4`, takes time.


## License

MIT License.

