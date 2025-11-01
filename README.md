<div align="center">
  <h1>Self-Inverse Task Vectors (SITV)</h1>
  <p><b>Exploring the loss landscape geometry of neural network task vectors. Maps L(M_base + αT) to find optimal scaling factors and investigate whether self-inverse properties from rotation groups emerge in parameter space.</b></p>
  <p>Loosely inspired by <a href="https://arxiv.org/abs/2502.14367"><i>Walks in Rotation Spaces Return Home when Doubled and Scaled</i></a> (Eckmann & Tlusty, 2025)</p>
</div>

<div align="center">

[![Version](https://img.shields.io/badge/Version-0.10.1-blue.svg?style=flat-square)](./CHANGELOG.md) [![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=flat-square)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-EE4C2C.svg?style=flat-square)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE) [![Status](https://img.shields.io/badge/Status-Research-yellow.svg?style=flat-square)](https://github.com/cyanheads/SITV)

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

# For Bayesian optimization sampling (optional)
pip install -e ".[bayesian]"
```

### Quick Start

Run the experiment with default settings:

```bash
python main.py
```

This will:
1. Load base model (`google/gemma-3-4b-it` by default)
2. Fine-tune on sentiment analysis task
3. Compute task vector `T = M_finetuned - M_base`
4. Sweep α from -3.0 to 3.0 (150 samples)
5. Evaluate `L(M_base + αT)` for each α
6. Generate visualizations and save to `./outputs/`
7. **Save models** for future re-analysis (new!)

### Command-Line Options

The CLI has been intentionally simplified to avoid configuration drift. All experiment parameters should be configured in `config.yaml`.

```bash
# Full experiment with default parameters
python main.py

# Use custom config file
python main.py --config my_config.yaml

# Analysis-only mode (skip fine-tuning, use saved models)
python main.py --analysis-only

# Combine custom config with analysis-only mode
python main.py --config my_config.yaml --analysis-only

# View all options
python main.py --help
```

**Analysis-Only Mode**: After running a full experiment, the base and fine-tuned models are saved to `outputs/saved_base_model/` and `outputs/saved_finetuned_model/`. Use `--analysis-only` to skip the time-consuming fine-tuning step and re-run the alpha sweep. To change experiment parameters (alpha range, sample count, etc.), edit `config.yaml` and run with `--analysis-only`. This is useful for:
- Testing different alpha ranges without re-training
- Increasing sample density for finer resolution
- Generating new visualizations with different settings

### Output Files

| File                            | Description                                          |
| :------------------------------ | :--------------------------------------------------- |
| `loss_landscape_sweep.png`      | 2x2 visualization grid showing loss curves           |
| `loss_landscape_results.json`   | Complete results with all α values and metrics       |
| `experiment_report.md`          | Comprehensive Markdown report for LLM analysis       |
| `experiment_metrics.json`       | Complete timing and performance metrics              |
| `saved_base_model/`             | Saved base model (for analysis-only mode)            |
| `saved_finetuned_model/`        | Saved fine-tuned model (for analysis-only mode)      |

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

### Configuration File (config.yaml)

The primary way to configure SITV is through `config.yaml` in the project root. All settings have sensible defaults optimized for modern GPUs.

**Example config.yaml:**

```yaml
# Reproducibility Configuration
reproducibility:
  seed: 42  # Random seed for reproducible experiments (Python, NumPy, PyTorch)
  # Set to null to use non-deterministic behavior

# Model Configuration
model:
  name: "google/gemma-3-4b-it"
  device: null  # auto-detect

# Task Configuration
task:
  name: "sentiment_positive"

# Evaluation Configuration
evaluation:
  general_dataset: "combined"  # Options: mixed_domain, wikitext, coding, common_knowledge, combined
  batch_size: 8  # Number of texts per forward pass (higher = faster but more memory)
  enable_mixed_precision: true  # Use FP16/BF16 for 1.5-2x speedup (auto-selects BF16 on Ampere+ GPUs)
  max_length: 1024  # Maximum sequence length for evaluation tokenization

# Output Configuration
output:
  dir: "outputs"
  analysis_only: false

# Fine-Tuning Configuration
fine_tuning:
  num_epochs: 2
  learning_rate: 5.0e-5
  batch_size: 16
  max_length: 512
  data_repetition_factor: 100  # Multiply training examples (30 unique × 100 = 3000 total)

# Alpha Sweep Configuration
alpha_sweep:
  alpha_min: -3.0
  alpha_max: 3.0
  num_samples: 150
  enable_squaring_test: true
  threshold: 0.1
  sampling_strategy: "uniform"  # Options: uniform, adaptive, bayesian
  # enable_gradient_analysis: false  # Set true to compute dL/dα and find critical points

# 2D Composition Configuration
composition_2d:
  enable: false
  alpha_min: -2.0
  alpha_max: 2.0
  beta_min: -2.0
  beta_max: 2.0
  num_samples_per_dim: 30
```

**Using custom config files:**

```bash
python main.py --config my_custom_config.yaml
```

**Configuration Philosophy:** All experiment settings are configured via YAML to maintain a single source of truth and avoid configuration drift. The only CLI options are `--config` (to specify a custom config file) and `--analysis-only` (to skip fine-tuning and use saved models).

### Hardware Support

The code automatically detects and optimizes for available hardware:

| Hardware         | Detection                      | Notes                                  |
| :--------------- | :----------------------------- | :------------------------------------- |
| **CUDA**         | NVIDIA GPUs                    | Recommended for large models           |
| **MPS**          | Apple Silicon (M1/M2/M3)       | Native acceleration on macOS           |
| **CPU**          | Fallback                       | Slower but works universally           |

## 📂 Project Structure

**v0.3.0+**: The project has been refactored into a modular package architecture with 30+ modules organized in 7 layers.

```
SITV/
├── main.py                      # Thin entry point (44 lines)
├── sitv/                        # Main package (30+ modules, 7,899 lines)
│   ├── data/                    # Data models and task definitions
│   │   ├── models.py            # AlphaSweepResult, ExperimentMetrics, etc.
│   │   ├── tasks.py             # Predefined task definitions
│   │   └── loader.py            # Dataset loader for text files
│   ├── core/                    # Core services (device, task vectors, evaluation)
│   │   ├── device.py            # Hardware detection and management
│   │   ├── task_vector.py       # Task vector operations
│   │   ├── evaluation.py        # Model evaluation and perplexity
│   │   ├── validation.py        # Input validation and error checking
│   │   └── error_handling.py    # Error handling and retry logic
│   ├── models/                  # Model management (loading, saving, fine-tuning)
│   │   ├── loader.py            # Model and tokenizer operations
│   │   └── fine_tuner.py        # Fine-tuning service with progress tracking
│   ├── experiments/             # Experiment orchestration and implementations
│   │   ├── base.py              # Abstract Experiment base class
│   │   ├── config.py            # Configuration classes
│   │   ├── alpha_sweep.py       # 1D alpha sweep experiment
│   │   ├── composition_2d.py    # 2D composition experiment
│   │   ├── orchestrator.py      # ExperimentOrchestrator
│   │   └── sampling/            # Sampling strategies (uniform, adaptive, bayesian)
│   ├── analysis/                # Results analysis
│   │   ├── analyzer.py          # Zero-crossing detection, min loss finding
│   │   └── gradient/            # Gradient analysis and critical point detection
│   ├── reporting/               # Report generation
│   │   └── markdown.py          # Markdown report generator
│   ├── visualization/           # Plotting and visualization
│   │   └── plotter.py           # 1D and 2D plotting utilities
│   ├── io/                      # File I/O operations
│   │   ├── file_manager.py      # JSON and figure saving
│   │   └── paths.py             # Path management
│   ├── utils/                   # Utilities
│   │   ├── console.py           # Console output formatting
│   │   ├── progress.py          # Progress tracking with ETA
│   │   └── timing.py            # Timing utilities
│   └── cli/                     # Command-line interface
│       └── args_parser.py       # Argument parsing
├── data/                        # Dataset files (text format)
│   ├── general/                 # General evaluation datasets
│   │   ├── mixed_domain.txt     # Mixed domain texts
│   │   ├── wikitext.txt         # Wikipedia-style texts
│   │   ├── coding.txt           # Programming-related texts
│   │   └── common_knowledge.txt # General knowledge texts
│   ├── tasks/                   # Task training datasets
│   │   ├── sentiment_positive.txt
│   │   ├── sentiment_negative.txt
│   │   ├── instruction_following.txt
│   │   └── qa_factual.txt
│   └── eval/                    # Task evaluation datasets
│       ├── sentiment_positive_eval.txt
│       ├── sentiment_negative_eval.txt
│       ├── instruction_following_eval.txt
│       └── qa_factual_eval.txt
├── tests/                       # Test suite
│   ├── conftest.py              # Pytest fixtures
│   ├── test_data_models.py      # Data model tests
│   ├── test_device.py           # Device management tests
│   ├── test_model_management.py # Model service tests
│   └── test_task_vector.py      # Task vector tests
├── archive/                     # Archived code
│   └── main_original.py         # Original 2,232-line monolithic version
├── pyproject.toml               # Project configuration and dependencies
├── CHANGELOG.md                 # Version history and release notes
├── README.md                    # This file
├── .gitattributes               # Git LFS and line ending configuration
├── .gitignore                   # Python, model, and output file patterns
└── outputs/                     # Generated visualizations and results (gitignored)
```

## 🧑‍💻 Development

### Testing

The project includes a test suite covering core functionality:

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data_models.py

# Run with coverage
pytest --cov=sitv
```

### Code Quality

```bash
# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy main.py sitv/

# Run all quality checks
ruff check . && ruff format --check . && mypy main.py sitv/ && pytest
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
| **SciPy**       | 1.11.0+  | Gradient analysis and smoothing        |
| **PyYAML**      | 6.0+     | Configuration file parsing             |

### Optional Dependencies

- **scikit-learn** 1.0+ - Bayesian optimization sampling (install with `pip install -e ".[bayesian]"`)

### Development Dependencies

- **ruff** - Fast Python linter and formatter
- **mypy** - Static type checking
- **pytest** - Testing framework
- **types-PyYAML** - Type stubs for YAML module

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
