# Self-Inverse Task Vectors (SITV)

Exploring functional return in neural network parameter space, inspired by [*Walks in Rotation Spaces Return Home when Doubled and Scaled*](https://arxiv.org/abs/2502.14367) (Eckmann & Tlusty, 2025).

## Overview

This project investigates whether neural network loss landscapes exhibit "functional roots of identity" under task vector transformations, analogous to how rotation walks can return to identity when doubled and scaled.

### Research Question

**Do there exist non-zero λ values where "doubled" task vector transformations functionally return to the base model's loss?**

We test two interpretations:

1. **LINEAR**: `M_doubled = M_base + 2λT` (scalar multiplication)
2. **COMPOSITIONAL**: `M_comp = M_base + λT + λ²T` (iterative application - closer to the paper)

Where `T = M_finetuned - M_base` is the task vector.

### Key Insight from Eckmann & Tlusty (2025)

For rotation groups SO(3)/SU(2), almost any walk W can be scaled and doubled to return to identity: `[W(λ)]² = I` for special λ values. This occurs because 180° rotations are abundant (`f₁(π) = 2/π`), and squaring them gives identity: `R(n,π)² = I`.

**Critical Difference**: Rotations form a multiplicative group, while task vectors form an additive vector space. This is an empirical exploration without the group structure guarantees of the paper.

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
4. Search for λ values where functional return ≈ 0
5. Generate visualizations and save results to `./outputs/`

### Output Files

- `self_inverse_task_vectors.png` - Comprehensive visualization comparing linear vs compositional approaches
- `self_inverse_task_results.json` - Detailed metrics for all λ values tested

## Methodology

### Two Approaches

**LINEAR (simpler, but less analogous to paper):**
- Transform: `M_doubled = M_base + 2λT`
- Tests for periodicity/symmetry in loss landscape

**COMPOSITIONAL (closer to paper's W² = W ∘ W):**
- Transform: `M_comp = M_base + λT + λ²T`
- Mimics iterative application: apply λT, then apply induced change again

### Metrics

- **Geometric Distance** (reference): `||M_doubled - M_base||` in parameter space
- **Functional Return** (KEY): `|L(M_doubled) - L(M_base)|` in loss landscape
- **Task Performance**: Model's loss on task-specific evaluation data
- **Utility Score**: Combined metric balancing functional return and task performance

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

### If Non-Zero λ* Found with Small Functional Return

✓ Indicates rich geometric structure in loss landscapes
✓ Suggests special scaling factors for task vector arithmetic
✓ Applications in model merging, multi-task learning
✓ Hints at deeper algebraic structure in parameter space

### If No Such λ* Exist

✓ Indicates loss landscape is monotonic along task vector direction
✓ Suggests task vectors lack the symmetry properties of rotations
✓ Still provides insights into parameter space geometry

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
