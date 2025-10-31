<div align="center">
  <h1>Self-Inverse Task Vectors (SITV)</h1>
  <p><b>Loss landscape explorer for neural network task vectors. Sweeps L(M_base + αT) to find optimal scaling factors and visualize how model performance changes along the task vector direction.</b></p>
  <p>Inspired by <a href="https://arxiv.org/abs/2502.14367"><i>Walks in Rotation Spaces Return Home when Doubled and Scaled</i></a> (Eckmann & Tlusty, 2025)</p>
</div>

<div align="center">

[![Version](https://img.shields.io/badge/Version-0.2.0-blue.svg?style=flat-square)](./CHANGELOG.md) [![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=flat-square)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.5.0+-EE4C2C.svg?style=flat-square)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE) [![Status](https://img.shields.io/badge/Status-Research-yellow.svg?style=flat-square)](https://github.com/cyanheads/SITV)

</div>

---

## 🔬 Research Question

**What does the loss landscape look like along the task vector direction? Does L(α) cross L(M_base) at any α ≠ 0?**

For a task vector `T = M_finetuned - M_base`, I sweep α ∈ [-3.0, 3.0] and plot `L(M_base + αT)` to discover:

| Metric                | Description                                                     |
| :-------------------- | :-------------------------------------------------------------- |
| **Zero-crossings**    | Values of α where loss returns to baseline (self-inverse points) |
| **Optimal scaling**   | Best α for task performance vs general knowledge trade-off       |
| **Landscape shape**   | Monotonic, periodic, or symmetric loss behavior                  |

### Connection to Rotation Groups

Eckmann & Tlusty (2025) prove that rotation group walks have abundant special λ values where `[W(λ)]² = I` (self-inverse property). When W(λ) is a 180° rotation, squaring returns to identity.

**Exploration**: I test whether neural loss landscapes exhibit analogous special scaling factors where loss functionally returns to baseline, even though task vectors lack the group structure of rotations.

## 📊 Methodology

### Loss Landscape Sweep

| Step | Operation                                    | Description                                              |
| :--- | :------------------------------------------- | :------------------------------------------------------- |
| 1    | **Create task vector**                       | `T = M_finetuned - M_base`                               |
| 2    | **Sample α values**                          | 100 points uniformly distributed in [-3.0, 3.0]          |
| 3    | **Compute models**                           | For each α: `M(α) = M_base + αT`                         |
| 4    | **Evaluate loss**                            | Measure `L(α)` on general and task-specific datasets     |
| 5    | **Identify special points**                  | Zero-crossings, minimum loss, optimal task performance   |
| 6    | **Visualize landscape**                      | Generate 2x2 plot grid with loss curves and analysis     |

### Performance Optimization

The implementation uses **in-place parameter modification**, achieving **10-100x speedup** compared to model reloading for each α value. This makes sweeping large models (e.g., 12B parameters) practical.

### Metrics Tracked

| Metric                  | Formula                  | Interpretation                                   |
| :---------------------- | :----------------------- | :----------------------------------------------- |
| **Loss L(α)**           | Loss of `M_base + αT`    | Primary metric - model performance at scaling α  |
| **Functional Return**   | `\|L(α) - L(M_base)\|`   | Distance from baseline loss                      |
| **Task Performance**    | Task-specific loss       | Performance on fine-tuning task                  |
| **Zero-crossings**      | α where `\|L(α) - L_base\| < threshold` | Self-inverse scaling factors |

## 🚀 Getting Started

### Prerequisites

- **Python 3.12+** (required for modern type hints)
- **GPU recommended** (CUDA, Apple Silicon MPS, or CPU fallback)
- **~40GB disk space** (for model downloads and checkpoints)

### Installation

```bash
# Clone the repository
git clone https://github.com/cyanheads/SITV.git
cd SITV

# Install dependencies
pip install -e .

# For development (includes ruff, mypy, pytest)
pip install -e ".[dev]"
```

### Quick Start

Run the experiment with default settings:

```bash
python main.py
```

This will:
1. Load base model (`google/gemma-3-12b-it` by default)
2. Fine-tune on sentiment analysis task
3. Compute task vector `T = M_finetuned - M_base`
4. Sweep α from -3.0 to 3.0 (100 samples)
5. Evaluate `L(M_base + αT)` for each α
6. Generate visualizations and save to `./outputs/`

### Output Files

| File                            | Description                                          |
| :------------------------------ | :--------------------------------------------------- |
| `loss_landscape_sweep.png`      | 2x2 visualization grid showing loss curves           |
| `loss_landscape_results.json`   | Complete results with all α values and metrics       |

## 📈 Interpreting Results

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

## 🛠️ Use Cases

| Application                | Description                                                          |
| :------------------------- | :------------------------------------------------------------------- |
| **Model Merging**          | Identify optimal α for combining base and fine-tuned models          |
| **Task Vector Arithmetic** | Understand valid scaling ranges for task vector composition          |
| **Multi-task Learning**    | Balance general knowledge vs task-specific performance               |
| **Parameter Space Study**  | Visualize loss landscape geometry and functional structure           |
| **Transfer Learning**      | Guide fine-tuning strategies with landscape insights                 |

## ⚙️ Configuration

### Hardware Support

The code automatically detects and optimizes for available hardware:

| Hardware         | Detection                      | Notes                                  |
| :--------------- | :----------------------------- | :------------------------------------- |
| **CUDA**         | NVIDIA GPUs                    | Recommended for large models           |
| **MPS**          | Apple Silicon (M1/M2/M3)       | Native acceleration on macOS           |
| **CPU**          | Fallback                       | Slower but works universally           |

### Model Configuration

Edit `main.py` to configure:

```python
# Model selection
model_name = "google/gemma-3-12b-it"  # Or any HuggingFace model

# Sweep parameters
alpha_range = (-3.0, 3.0)  # Range of α values
num_samples = 100          # Number of points to sample

# Task configuration
task = "sentiment_analysis"  # Or implement custom tasks
```

## 📂 Project Structure

```
SITV/
├── main.py              # Main experiment implementation
├── pyproject.toml       # Project configuration and dependencies
├── CHANGELOG.md         # Version history and release notes
├── README.md            # This file
├── .gitattributes       # Git LFS and line ending configuration
├── .gitignore           # Python, model, and output file patterns
└── outputs/             # Generated visualizations and results (gitignored)
```

## 🧑‍💻 Development

### Code Quality

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy main.py
```

### Development Workflow

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes** following the existing patterns
4. **Run quality checks**: `ruff check . && mypy main.py`
5. **Commit with conventional commits**: `git commit -m "feat: add feature"`
6. **Push and open a Pull Request**

## 📚 Requirements

### Core Dependencies

| Package         | Version  | Purpose                                |
| :-------------- | :------- | :------------------------------------- |
| **Python**      | 3.12+    | Modern type hints and language features |
| **PyTorch**     | 2.5.0+   | Neural network operations              |
| **Transformers** | 4.50.0+ | Pre-trained language models            |
| **NumPy**       | 2.0.0+   | Numerical computing                    |
| **Matplotlib**  | 3.9.0+   | Visualization and plotting             |
| **Accelerate**  | 0.20.0+  | Multi-GPU and mixed precision training |

### Development Dependencies

- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **pytest** - Testing framework

See [pyproject.toml](pyproject.toml) for complete dependency specifications.

## 📖 Citation

If you use this code in your research, please cite both this repository and the inspiring paper:

```bibtex
@misc{sitv2025,
  title={Self-Inverse Task Vectors: Exploring Loss Landscapes Along Task Vector Directions},
  author={Hand, Casey},
  year={2025},
  publisher={GitHub},
  url={https://github.com/cyanheads/SITV}
}

@article{eckmann2025walks,
  title={Walks in Rotation Spaces Return Home when Doubled and Scaled},
  author={Eckmann, Jean-Pierre and Tlusty, Tsvi},
  journal={arXiv preprint arXiv:2502.14367v3},
  year={2025}
}
```

## 🤝 Contributing

Contributions are welcome! Areas of interest:

- **Additional tasks**: Implement more fine-tuning tasks beyond sentiment analysis
- **Visualization improvements**: Enhanced plots and interactive visualizations
- **Performance optimizations**: Further speedups for large-scale experiments
- **Theoretical analysis**: Mathematical insights into loss landscape geometry
- **Empirical studies**: Test on diverse model architectures and tasks

Please open an issue to discuss major changes before submitting a PR.

## 📜 License

This project is licensed under the MIT License. See the [LICENSE](./LICENSE) file for details.

## 🙏 Acknowledgments

This research explores the fascinating connection between rotation group theory and neural network parameter spaces. Special thanks to:

- **Jean-Pierre Eckmann & Tsvi Tlusty** for their groundbreaking work on rotation walks
- **The HuggingFace team** for Transformers and model hosting
- **The PyTorch team** for the deep learning framework

---

<div align="center">
  <p><i>Exploring the geometry of neural loss landscapes, one task vector at a time.</i></p>
  <p>
    <a href="https://github.com/cyanheads/SITV/issues">Report Bug</a> •
    <a href="https://github.com/cyanheads/SITV/issues">Request Feature</a> •
    <a href="https://github.com/sponsors/cyanheads">Sponsor</a>
  </p>
</div>
