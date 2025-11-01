<div align="center">
  <h1>Self-Inverse Task Vectors (SITV)</h1>
  <p><b>Exploring the loss landscape geometry of neural network task vectors. Maps L(M_base + αT) to find optimal scaling factors and investigate whether self-inverse properties from rotation groups emerge in parameter space.</b></p>
  <p>Loosely inspired by <a href="https://arxiv.org/abs/2502.14367"><i>Walks in Rotation Spaces Return Home when Doubled and Scaled</i></a> (Eckmann & Tlusty, 2025)</p>
</div>

<div align="center">

[![Version](https://img.shields.io/badge/Version-0.15.0-blue.svg?style=flat-square)](./CHANGELOG.md) [![Python](https://img.shields.io/badge/Python-3.12+-3776AB.svg?style=flat-square)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.8.0+-EE4C2C.svg?style=flat-square)](https://pytorch.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg?style=flat-square)](./LICENSE) [![Status](https://img.shields.io/badge/Status-Research-yellow.svg?style=flat-square)](https://github.com/cyanheads/SITV)

</div>

See [CHANGELOG.md](./CHANGELOG.md) for complete version history.

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
| 1    | **Create task vector(s)**                    | `T = M_finetuned - M_base` (1D, 2D, or 3D composition)   |
| 2    | **Sample α values**                          | 100+ points (uniform/adaptive/Bayesian sampling)         |
| 3    | **Compute models**                           | For each α: `M(α) = M_base + αT` or geodesic `exp_M(α·v)` |
| 4    | **Evaluate loss**                            | Measure `L(α)` on general and task-specific datasets     |
| 5    | **Analyze geometry** (optional)              | Fisher metrics, geodesics, curvature, symmetries         |
| 6    | **Identify special points**                  | Zero-crossings, minimum loss, optimal task performance   |
| 7    | **Visualize landscape**                      | 1D/2D/3D plots with interactive exploration              |

### Performance Optimization

The implementation uses multiple optimization techniques for efficient large-scale experiments:

- **In-place parameter modification**: 10-100x speedup by reusing single model instance
- **Batched evaluation**: Process multiple texts per forward pass (configurable batch size)
- **Mixed precision**: Auto-selects BF16/FP16 for 1.5-2x speedup on modern GPUs
- **Adaptive sampling**: 40-60% fewer samples with multi-resolution refinement
- **Bayesian optimization**: 80-90% speedup for finding optimal α values quickly

These optimizations make sweeping large models (e.g., 12B parameters) practical on consumer hardware.

### Metrics Tracked

| Metric                  | Formula                  | Interpretation                                   |
| :---------------------- | :----------------------- | :----------------------------------------------- |
| **Loss L(α)**           | Loss of `M_base + αT`    | Primary metric - model performance at scaling α  |
| **Functional Return**   | `\|L(α) - L(M_base)\|`   | Distance from baseline loss                      |
| **Task Performance**    | Task-specific loss       | Performance on fine-tuning task                  |
| **Zero-crossings**      | α where `\|L(α) - L_base\| < threshold` | Self-inverse scaling factors |

## 🧭 Riemannian Geometry on Parameter Manifolds

**v0.10.0+**: SITV implements proper Riemannian geometry to detect curvature and geometric structure in neural network parameter space.

### Why Riemannian Geometry?

Neural network parameters don't live in flat Euclidean space—they form a curved manifold where the "distance" between points depends on the loss landscape. Traditional task vectors use straight-line interpolation `M(α) = M_base + αT`, but the true shortest path (geodesic) may curve through parameter space.

**Core question**: Does parameter space have intrinsic curvature, like an "anthill" with branching tunnels?

### Fisher Information Matrix as Metric Tensor

The Fisher Information Matrix `g` defines a Riemannian metric on parameter manifolds:

- **Metric types**: `euclidean`, `fisher_diagonal`, `fisher_kfac`, `fisher_full`
- **Task vector norms**: `||T||_g = √(T^T · g · T)` (Riemannian) vs `||T||` (Euclidean)
- **Geodesic paths**: Computed via Runge-Kutta 4 integration of the geodesic equation

### Geometric Features

| Feature | Description | Use Case |
| :------ | :---------- | :------- |
| **Geodesic Integration** | Replace `M + αT` with exponential map `exp_M(α·v)` | True shortest paths in curved space |
| **Curvature Analysis** (v0.13.0) | Sectional, Ricci, scalar curvature computation | Detect if space is curved or flat |
| **Symmetry Detection** (v0.14.0) | Rotation/permutation/scaling symmetries | Remove parameter redundancy |
| **Christoffel Symbols** | Recomputed along geodesics to capture metric variation | Varying-metric geodesics for curvature detection |

### Configuration Example

```yaml
geometry:
  enabled: true
  metric_type: "fisher_diagonal"  # Fast approximation

  geodesic_integration:
    enabled: true
    num_steps: 20
    recompute_metric_every: 5  # Detect curvature!

  curvature_analysis:
    enabled: true
    compute_sectional: true  # K(X,Y) curvature of 2D planes

  symmetry_analysis:
    enabled: true
    quotient_space: true  # Work in canonical parameter form
```

**Interpretation**: If geodesics deviate from straight lines → **curved geometry detected** → parameter space has rich structure beyond Euclidean assumptions.

## 🎲 Multi-Dimensional Task Vector Composition

### 2D Composition (v0.5.0+)

Explore how two tasks interact via 2D sweeps: `L(M_base + α·T1 + β·T2)`

- **Grid resolution**: Configurable (20×20 to 60×60 evaluations)
- **Automatic analysis** (v0.12.0): Detects additive vs synergistic effects
- **Interaction metrics**: R² score, interaction RMS
- **Predictions**: Optimal 2D composition predicted from 1D task properties

### 3D Composition (v0.11.0)

**Three-task interaction analysis**: `L(M_base + α·T1 + β·T2 + γ·T3)`

- **Interactive visualizations**: Plotly 3D plots with rotation and zoom
- **Cross-sectional slices**: 2D views at fixed γ values
- **Computational cost**: Scales cubically (10³ = 1,000 evals typical, 20³ = 8,000 for high-res)
- **Use cases**: Understanding complex multi-task compositions

### Composition Analysis (v0.12.0)

**Automatic interaction detection** after 2D/3D sweeps:

- **Task independence**: R² score measures how well 2D optimal is predicted from 1D
- **Interaction patterns**: Additive, synergistic, or interference effects
- **Prediction strategies**: Independent, weighted average, balanced composition
- **Practical implications**: Guides multi-task learning and model merging

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

All outputs are saved to the `outputs/` directory (gitignored by default).

#### 1D Alpha Sweep Outputs
| File | Description |
| :--- | :---------- |
| `loss_landscape_sweep.png` | 2x2 (or 2x3 with squaring test) visualization grid |
| `loss_landscape_results.json` | All α values and metrics |
| `experiment_report.md` | Markdown report with geometry analysis (if enabled) |
| `experiment_metrics.json` | Timing and performance metrics |

#### 2D Composition Outputs
| File | Description |
| :--- | :---------- |
| `loss_landscape_2d_sweep.png` | 2D heatmap of loss landscape |
| `loss_landscape_2d_results.json` | All (α, β) values and losses |
| `composition_analysis_*.png` | Analysis visualizations (if `enable_analysis: true`) |
| `analysis_results.json` | Interaction detection and predictions |

#### 3D Composition Outputs (v0.11.0+)
| File | Description |
| :--- | :---------- |
| `loss_landscape_3d_*.png` | 3D surface plots and 2D cross-sections |
| `loss_landscape_3d_*.html` | Interactive 3D plots (Plotly) with rotation/zoom |
| `loss_landscape_3d_results.json` | All (α, β, γ) values and losses |

#### Multi-Task Comparison (v0.12.0+)
| File | Description |
| :--- | :---------- |
| `comparison_report.md` | Multi-task comparison report |
| `task_comparison_*.png` | Side-by-side task comparison plots |

#### Saved Models
| File | Description |
| :--- | :---------- |
| `saved_base_model/` | Base model (for --analysis-only mode) |
| `saved_finetuned_model/` | Fine-tuned model (for --analysis-only mode) |
| `saved_task_{name}_model/` | Additional fine-tuned models (multi-task experiments) |

## 📈 Interpreting Results

### 1D Alpha Sweep Interpretation

#### If Zero-Crossings Found (L(α) ≈ L(M_base) for α ≠ 0)

✓ Suggests special scaling factors exist (analogous to 180° rotations)
✓ Rich geometric structure in loss landscapes
✓ Potential applications in model merging and multi-task learning
✓ Indicates non-monotonic loss behavior along task vector direction

#### If No Zero-Crossings Found

✓ Loss is monotonic along task vector direction
✓ Task vectors lack rotation-like symmetry properties
✓ Optimal scaling is straightforward (minimum of monotonic curve)
✓ Still provides insights into parameter space geometry and optimal α values

### Riemannian Geometry Interpretation (v0.10.0+)

#### If Geodesics Deviate from Straight Lines

✓ **Curved geometry detected** → parameter space is non-Euclidean
✓ Task vector paths naturally curve through parameter space
✓ Riemannian norm `||T||_g` differs significantly from Euclidean `||T||`
✓ Supports "anthill hypothesis" of branching geometric structure

#### Curvature Analysis (v0.13.0)

- **Positive curvature** (K > 0): Sphere-like, geodesics converge
- **Negative curvature** (K < 0): Hyperbolic, geodesics diverge
- **Zero curvature** (K ≈ 0): Locally flat, Euclidean approximation valid

#### Symmetry Detection (v0.14.0)

- **Rotation symmetries**: Loss invariant to orthogonal transformations
- **Permutation symmetries**: Neuron ordering doesn't affect loss
- **Scaling symmetries**: Layer-wise rescaling preserves loss
- **Quotient space**: Working in canonical form removes redundancy

### Composition Analysis Interpretation (v0.12.0+)

#### Task Interaction Patterns

- **Additive** (R² > 0.95): Tasks compose independently, 1D properties predict 2D optimal
- **Synergistic** (R² 0.7-0.95): Tasks amplify each other, 2D optimal better than predicted
- **Interference** (R² < 0.7): Tasks interfere, requires careful balancing

#### Practical Implications

- **High independence**: Use simple weighted combination of task vectors
- **Moderate interaction**: Use 2D optimization to find optimal composition
- **Strong interference**: May need 3D analysis or sequential fine-tuning

## 🛠️ Use Cases

| Application                | Description                                                          |
| :------------------------- | :------------------------------------------------------------------- |
| **Model Merging**          | Identify optimal α for combining base and fine-tuned models          |
| **Task Vector Arithmetic** | Understand valid scaling ranges for task vector composition          |
| **Multi-task Learning**    | Balance general knowledge vs task-specific performance               |
| **Parameter Space Study**  | Visualize loss landscape geometry and functional structure           |
| **Transfer Learning**      | Guide fine-tuning strategies with landscape insights                 |
| **Geometric Structure Discovery** (v0.10.0+) | Detect curvature and symmetries in neural network parameter space |
| **Curvature-Aware Navigation** (v0.13.0) | Use geodesics for optimal parameter space traversal |
| **Symmetry Reduction** (v0.14.0) | Work in quotient spaces to remove parameter redundancy |
| **Three-Task Interaction** (v0.11.0) | Understand complex multi-task compositions in 3D |
| **Composition Prediction** (v0.12.0) | Predict multi-task optimal points from single-task properties |

## ⚙️ Configuration

### Configuration File (config.yaml)

The primary way to configure SITV is through `config.yaml` in the project root. All settings have sensible defaults optimized for modern GPUs.

**Example config.yaml:**

```yaml
# Reproducibility Configuration (v0.8.0+)
reproducibility:
  seed: 42  # Random seed for reproducible experiments (Python, NumPy, PyTorch)
  # Set to null to use non-deterministic behavior

# Model Configuration
model:
  name: "google/gemma-3-4b-it"
  device: null  # auto-detect (cuda/mps/cpu)

# Task Configuration
task:
  name: "sentiment_positive"  # Options: sentiment_positive, sentiment_negative, instruction_following, qa_factual

# Evaluation Configuration (v0.9.0+: batching and mixed precision)
evaluation:
  general_dataset: "combined"  # Options: mixed_domain, wikitext, coding, common_knowledge, combined
  batch_size: 32  # Number of texts per forward pass (higher = faster but more memory)
  enable_mixed_precision: true  # Use FP16/BF16 for 1.5-2x speedup
  max_length: 1024  # Maximum sequence length for evaluation

# Fine-Tuning Configuration
fine_tuning:
  num_epochs: 4
  learning_rate: 1.0e-4
  batch_size: 32
  max_length: 1024
  data_repetition_factor: 100  # Multiply training examples (30 × 100 = 3000)

# Alpha Sweep Configuration
alpha_sweep:
  alpha_min: -0.5
  alpha_max: 1.5
  num_samples: 100
  enable_squaring_test: true
  threshold: 0.20
  sampling_strategy: "uniform"  # Options: uniform, adaptive, bayesian

# 2D Composition Configuration (v0.12.0+: automatic analysis)
composition_2d:
  enable: true
  alpha_min: -2.0
  alpha_max: 2.0
  beta_min: -2.0
  beta_max: 2.0
  num_samples_per_dim: 30
  enable_analysis: true  # Auto-run composition analysis

# 3D Composition Configuration (v0.11.0+)
composition_3d:
  enable: false  # WARNING: computationally expensive!
  task_1: "sentiment_negative"
  task_2: "sentiment_positive"
  task_3: "instruction_following"
  alpha_min: -1.0
  alpha_max: 1.0
  beta_min: -1.0
  beta_max: 1.0
  gamma_min: -1.0
  gamma_max: 1.0
  num_samples_per_dim: 10  # 10³ = 1,000 evaluations

# Riemannian Geometry Configuration (v0.10.0+)
geometry:
  enabled: false  # Set true to enable geometric analysis
  metric_type: "fisher_diagonal"  # Options: euclidean, fisher_diagonal, fisher_kfac, fisher_full

  geodesic_integration:
    enabled: false
    num_steps: 20
    recompute_metric_every: 5  # Detect curvature

  curvature_analysis:  # v0.13.0+
    enabled: false
    compute_sectional: true

  symmetry_analysis:  # v0.14.0+
    enabled: false
    detect_rotations: true
    detect_permutations: true
    quotient_space: true
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

**v0.14.1**: The project uses a modular package architecture with **53 modules** organized in **9 layers** (**14,742 lines of code**), including comprehensive Riemannian geometry support.

```
SITV/
├── main.py                      # Thin entry point (44 lines)
├── sitv/                        # Main package (53 modules, 14,742 lines)
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
│   │   ├── composition_3d.py    # 3D composition experiment (v0.11.0+)
│   │   ├── orchestrator.py      # ExperimentOrchestrator
│   │   └── sampling/            # Sampling strategies (uniform, adaptive, bayesian)
│   ├── geometry/                # Riemannian geometry on parameter manifolds (v0.10.0+)
│   │   ├── config.py            # Geometry configuration
│   │   ├── metric.py            # Fisher Information Matrix computation
│   │   ├── geodesic.py          # Geodesic integration (Runge-Kutta)
│   │   ├── task_vector.py       # Geodesic task vector operations
│   │   ├── curvature.py         # Curvature analysis (v0.13.0)
│   │   └── symmetry.py          # Symmetry detection (v0.14.0)
│   ├── analysis/                # Results analysis
│   │   ├── analyzer.py          # Zero-crossing detection, min loss finding
│   │   ├── composition_analyzer.py  # Composition analysis (v0.12.0)
│   │   └── gradient/            # Gradient analysis and critical point detection
│   ├── reporting/               # Report generation
│   │   ├── markdown.py          # Markdown report generator
│   │   └── comparison_report.py # Multi-task comparison reports (v0.12.0)
│   ├── visualization/           # Plotting and visualization
│   │   └── plotter.py           # 1D/2D/3D plotting utilities
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
├── tests/                       # Test suite (17 test files, 57,000+ lines of tests)
│   ├── conftest.py              # Pytest fixtures
│   ├── test_data_models.py      # Data model tests
│   ├── test_device.py           # Device management tests
│   ├── test_model_management.py # Model service tests
│   ├── test_task_vector.py      # Task vector tests
│   ├── test_config.py           # Configuration tests
│   ├── test_data_loader.py      # Data loading tests
│   ├── test_analyzer.py         # Analysis tests
│   ├── test_markdown_reporter.py # Report generation tests
│   ├── test_sampling_strategies.py # Sampling strategy tests
│   ├── test_file_manager.py     # File I/O tests
│   └── geometry/                # Geometry module tests (v0.10.0+)
│       ├── test_fisher_metric.py # Fisher metric tests (10,273 lines)
│       ├── test_geodesic.py     # Geodesic integration tests (10,036 lines)
│       ├── test_curvature.py    # Curvature analysis tests (16,822 lines)
│       └── test_symmetry.py     # Symmetry detection tests (19,581 lines)
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
| **Plotly**      | 5.0.0+   | Interactive 3D visualizations (v0.11.0+) |
| **Accelerate**  | 0.20.0+  | Multi-GPU and mixed precision training |
| **SciPy**       | 1.11.0+  | Gradient analysis, smoothing, Riemannian geometry (v0.6.0+) |
| **PyYAML**      | 6.0+     | Configuration file parsing             |

### Optional Dependencies

- **scikit-learn** 1.0+ - Bayesian optimization sampling (install with `pip install -e ".[bayesian]"`, v0.6.0+)

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
