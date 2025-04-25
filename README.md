# SVGD Exploration Repository

Experiments and demonstrations of Stein Variational Gradient Descent (SVGD)

## Structure

- **`svgd-demo.py`**  
  Compares two SVGD implementations:
  - Exact RBF kernel (white particles)
  - RFF approximation (red particles)
  Profiles per-step runtime and saves an MP4.

## Features

- Adaptive RBF kernel using the median heuristic.
- Random Fourier Features for scalable kernel approximation.

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

## License

MIT License.

