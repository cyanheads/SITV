# Self-Inverse Task Vectors (SITV)
### Exploring Loss Landscapes Along Task Vector Directions

Inspired by [*Walks in Rotation Spaces Return Home when Doubled and Scaled*](https://arxiv.org/abs/2502.14367) (Eckmann & Tlusty, 2025), this project searches for self-inverse scaling factors in neural network parameter space.

## Overview

This project visualizes how neural network loss changes as we move along task vector directions: `L(M_base + αT)`, where `T = M_finetuned - M_base` is the task vector.

### Research Question

**What does the loss landscape look like along the task vector direction? Does L(α) cross L(M_base) at any α ≠ 0?**

We sweep α values from -3.0 to 3.0 and plot the resulting loss landscape to discover:
- **Zero-crossings**: Values of α where loss returns to baseline (analogous to rotation self-inverse points)
- **Optimal scaling**: Best α for task performance
- **Landscape shape**: Monotonic, periodic, or symmetric?

### Connection to Eckmann & Tlusty (2025)

Their work proves that rotation group walks have abundant special λ values where `[W(λ)]² = I` (self-inverse property) - when W(λ) is a 180° rotation, squaring returns to identity.

**Our Exploration**: We test whether neural loss landscapes exhibit analogous special scaling factors where loss functionally returns to baseline, even though task vectors lack the group structure of rotations.

## Installation

Requires Python 3.12 or higher.

```bash
# Clone the repository
git clone https://github.com/yourusername/SITV.git
cd SITV

# Install dependencies
pip install -e .

# For development
pip install -e ".[dev]"
```

## Usage

Run the main experiment:

```bash
python main.py
```

This will:
1. Load a base language model (default: `google/gemma-3-12b-it`)
2. Fine-tune it on a sentiment analysis task
3. Compute the task vector `T = M_finetuned - M_base`
4. Sweep α from -3.0 to 3.0 (100 samples)
5. Evaluate `L(M_base + αT)` for each α
6. Generate visualizations and save results to `./outputs/`

### Output Files

- `loss_landscape_sweep.png` - Visualization of loss landscape L(α) vs α
- `loss_landscape_results.json` - Detailed metrics for all α values tested

## Methodology

### Loss Landscape Sweep

For a given task vector `T = M_finetuned - M_base`, we:

1. **Sweep α values**: Sample 100 values uniformly from [-3.0, 3.0]
2. **Compute models**: For each α, create `M(α) = M_base + αT`
3. **Evaluate loss**: Measure `L(α)` on general and task-specific evaluation sets
4. **Find special points**:
   - **Zero-crossings**: α ≠ 0 where `|L(α) - L(M_base)| < threshold`
   - **Minimum general loss**: Best α for preserving general knowledge
   - **Minimum task loss**: Best α for task performance

### Performance Optimization

The implementation uses in-place parameter modification, achieving 10-100x speedup compared to model reloading for each α value.

### Metrics

- **Loss L(α)**: Primary metric - model loss at scaling factor α
- **Functional Return**: `|L(α) - L(M_base)|` - distance from baseline
- **Task Performance**: Task-specific loss at each α value
- **Zero-crossings**: Special α values where loss returns to baseline

## Project Structure

```
SITV/
├── main.py              # Main experiment implementation
├── pyproject.toml       # Project configuration and dependencies
├── CHANGELOG.md         # Version history
├── README.md            # This file
├── .gitattributes       # Git attributes configuration
└── .gitignore           # Git ignore patterns
```

## Interpretation

### If Zero-Crossings Found (L(α) ≈ L(M_base) for α ≠ 0)

✓ Suggests special scaling factors exist (analogous to 180° rotations)
✓ Rich geometric structure in loss landscapes
✓ Potential applications in model merging and multi-task learning
✓ Indicates non-monotonic loss behavior along task vector direction

### If No Zero-Crossings Found

✓ Loss is monotonic along task vector direction
✓ Task vectors lack rotation-like symmetry properties
✓ Optimal scaling is straightforward (minimum of monotonic curve)
✓ Still provides insights into parameter space geometry and optimal α values

### Practical Applications

- **Model Merging**: Identify optimal α for combining base and fine-tuned models
- **Task Vector Arithmetic**: Understand valid scaling ranges for task vectors
- **Multi-task Learning**: Balance general knowledge vs task-specific performance
- **Parameter Space Geometry**: Visualize loss landscape structure

## Requirements

- Python 3.12+
- PyTorch 2.5.0+
- Transformers 4.50.0+
- NumPy 2.0.0+
- Matplotlib 3.9.0+
- Accelerate 0.20.0+

See [pyproject.toml](pyproject.toml) for complete dependency specifications.

## Hardware Support

The code automatically detects and uses available hardware:
- **CUDA** (NVIDIA GPUs)
- **MPS** (Apple Silicon)
- **CPU** (fallback)

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run linter
ruff check .

# Run type checker
mypy main.py

# Run tests
pytest
```

## Citation

This work is inspired by:

```bibtex
@article{eckmann2025walks,
  title={Walks in Rotation Spaces Return Home when Doubled and Scaled},
  author={Eckmann, Jean-Pierre and Tlusty, Tsvi},
  journal={arXiv preprint arXiv:2502.14367v3},
  year={2025}
}
```

## License

[Add your license here]

## Acknowledgments

This research explores the fascinating connection between rotation group theory and neural network parameter spaces. Special thanks to Eckmann & Tlusty for their groundbreaking work on rotation walks.
