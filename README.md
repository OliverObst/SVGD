# SVGD Exploration Repository

Experiments and demonstrations of Stein Variational Gradient Descent (SVGD)

See also: https://github.com/kimbente/svgd
for a nicer repo on SVGD

## Structure

- **`svgd-demo.py`**  
  Compares two SVGD implementations:
  - Exact RBF kernel (white particles)
  - RFF approximation (red particles)
  - Profiles per-step runtime and saves an MP4.
  
- **`svgd-kde.py`**
  Does the SVGD thing but without autograd
  - RBF kernel
  - uses KDE estimation for log p
  Not as terrible as I thought.

## Features

- Adaptive RBF kernel using the median heuristic.
- Random Fourier Features for scalable kernel approximation.
- autograd and KDE

### Prerequisites

- Python 3.8+  
- [PyTorch](https://pytorch.org/)  
- [Matplotlib](https://matplotlib.org/)  
- [Imageio](https://imageio.github.io/) (for GIF conversion)

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

