# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SITV (Self-Inverse Task Vectors) is a research project exploring loss landscape geometry of neural network task vectors. It maps `L(M_base + αT)` to find optimal scaling factors and investigate whether self-inverse properties from rotation groups emerge in parameter space.

**Core concept**: For a task vector `T = M_finetuned - M_base`, sweep α ∈ [-3.0, 3.0] and plot `L(M_base + αT)` to discover zero-crossings, optimal scaling, and landscape shape.

## Recent Features (v0.10.0 → v0.12.0)

### v0.12.0 (2025-11-01) - Composition Analysis
- **Automatic Composition Analysis**: Detects task interaction patterns after 2D sweeps
  - Analyzes task independence with R² score and interaction RMS metrics
  - Predicts optimal 2D composition from 1D task properties
  - Auto-runs when `composition_2d.enable_analysis: true`
- **Comparison Report Generator**: Multi-task comparison reports (`sitv/reporting/comparison_report.py`)
- **Enhanced Markdown Reports**: Comprehensive composition analysis sections with predictions

### v0.11.0 (2025-11-01) - 3D Composition & Multi-Task Comparison
- **3D Task Vector Composition**: Three-task interaction analysis `L(M_base + αT1 + βT2 + γT3)`
  - Configurable grid sampling (5³ to 20³ evaluations)
  - Interactive 3D plots (HTML via plotly) with rotation and zoom
  - 2D cross-sectional slices showing loss at different γ values
- **Multi-Task Comparison Plotting**: 5 layout modes (overlaid, side-by-side, grid, publication, heatmap)
- **Composition Analysis Module**: `CompositionAnalyzer` for 2D/3D composition analysis

### v0.10.0 (2025-10-31) - Riemannian Geometry
- **Riemannian Geometry Framework**: Proper geometry on parameter manifolds
  - Fisher Information Matrix computation (diagonal, KFAC, full)
  - Geodesic integration via Runge-Kutta 4
  - Replaces Euclidean `M(α) = M_base + αT` with geodesic `exp_M(α·T)`
- **Enhanced Numerical Stability**: Perplexity overflow protection, finite value validation
- **Geometry Reporting**: Geodesic vs Euclidean path comparison, curvature analysis

## Commands

### Development

```bash
# Install for development
pip install -e ".[dev]"

# Install with Bayesian optimization support
pip install -e ".[bayesian]"
```

### Running Experiments

```bash
# Full experiment with default config
python main.py

# Use custom config file
python main.py --config my_config.yaml

# Analysis-only mode (skip fine-tuning, use saved models)
python main.py --analysis-only

# Combine custom config with analysis-only
python main.py --config my_config.yaml --analysis-only
```

### Testing and Quality

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/test_data_models.py

# Run with coverage
pytest --cov=sitv

# Lint code
ruff check .

# Format code
ruff format .

# Type checking
mypy main.py sitv/

# Run all quality checks (recommended before commits)
ruff check . && ruff format --check . && mypy main.py sitv/ && pytest
```

## Architecture

### Modular Package Structure (v0.12.0)

The project uses a **8-layer service-oriented architecture** with 51 modules:

```
sitv/
├── data/          # Data models and task definitions
│   ├── models.py  # AlphaSweepResult, ExperimentMetrics, etc.
│   ├── tasks.py   # Predefined task definitions
│   └── loader.py  # Dataset loader for text files
├── core/          # Core services
│   ├── device.py        # Hardware detection (CUDA/MPS/CPU)
│   ├── task_vector.py   # Task vector operations (T = M_ft - M_base)
│   ├── evaluation.py    # Model evaluation and perplexity
│   ├── validation.py    # Input validation
│   └── error_handling.py # Error handling and retry logic
├── models/        # Model management
│   ├── loader.py       # Model and tokenizer operations
│   └── fine_tuner.py   # Fine-tuning service
├── experiments/   # Experiment orchestration
│   ├── base.py           # Abstract Experiment base class
│   ├── config.py         # Configuration classes (loads config.yaml)
│   ├── orchestrator.py   # ExperimentOrchestrator (main workflow)
│   ├── alpha_sweep.py    # 1D alpha sweep experiment
│   ├── composition_2d.py # 2D composition experiment
│   ├── composition_3d.py # 3D composition experiment (three-task interactions)
│   └── sampling/         # Sampling strategies (uniform, adaptive, bayesian)
├── geometry/      # Riemannian geometry on parameter manifolds
│   ├── config.py        # Geometry configuration
│   ├── metric.py        # Fisher Information Matrix computation
│   ├── geodesic.py      # Geodesic integration (Runge-Kutta)
│   └── task_vector.py   # Geodesic task vector operations
├── analysis/      # Results analysis
│   ├── analyzer.py              # Zero-crossing detection, min loss finding
│   ├── composition_analyzer.py  # Composition analysis and interaction detection
│   └── gradient/                # Gradient analysis and critical point detection
├── reporting/     # Report generation
│   ├── markdown.py          # Markdown report generator
│   └── comparison_report.py # Multi-task comparison reports
├── visualization/ # Plotting
│   └── plotter.py  # 1D and 2D plotting utilities
├── io/            # File I/O
│   ├── file_manager.py # JSON and figure saving
│   └── paths.py        # Path management
├── utils/         # Utilities
│   ├── console.py  # Console output formatting
│   ├── progress.py # Progress tracking with ETA
│   └── timing.py   # Timing utilities
└── cli/           # Command-line interface
    └── args_parser.py # Argument parsing
```

### Execution Flow

1. **Entry Point** (`main.py` - 44 lines)
   - Parses CLI args
   - Creates `ExperimentConfig` from args + config.yaml
   - Instantiates `ExperimentOrchestrator`
   - Runs experiment

2. **Orchestrator** (`sitv/experiments/orchestrator.py`)
   - Coordinates entire workflow
   - Loads/fine-tunes models via `ModelService` and `FineTuner`
   - Computes task vectors via `TaskVectorService` (Euclidean or geodesic)
   - Runs experiments (`AlphaSweepExperiment`, `Composition2DExperiment`, `Composition3DExperiment`)
   - Analyzes results via `ResultAnalyzer` and `CompositionAnalyzer`
   - Generates reports via `MarkdownReportGenerator` and `ComparisonReportGenerator`
   - Saves outputs via `FileManager`

3. **Experiments** (inherit from `sitv/experiments/base.py`)
   - **AlphaSweepExperiment**: 1D sweep of `L(M_base + αT)`
   - **Composition2DExperiment**: 2D sweep of `L(M_base + αT1 + βT2)` with optional composition analysis
   - **Composition3DExperiment**: 3D sweep of `L(M_base + αT1 + βT2 + γT3)` (three-task interactions)
   - All use **in-place parameter modification** for 10-100x speedup

4. **Sampling Strategies** (`sitv/experiments/sampling/`)
   - **Uniform**: 100% of samples uniformly distributed
   - **Adaptive**: 40-60% of samples, multi-resolution (coarse then refine)
   - **Bayesian**: 10-20% of samples, Gaussian Process-based (requires `[bayesian]` install)

### Key Design Patterns

- **Service-oriented architecture**: Core functionality split into services (ModelService, TaskVectorService, EvaluationService, etc.)
- **Configuration as code**: All experiment parameters in `config.yaml` (single source of truth)
- **In-place parameter modification**: Reuses single model instance, modifying parameters in-place for memory efficiency
- **Orchestrator pattern**: `ExperimentOrchestrator` coordinates workflow
- **Strategy pattern**: Pluggable sampling strategies (uniform/adaptive/bayesian)
- **Validation at boundaries**: Input validation in `core/validation.py` before processing

## Configuration Philosophy

**CRITICAL**: All experiment settings are configured via `config.yaml`. The CLI has been **intentionally simplified** to avoid configuration drift. Only two CLI options exist:
- `--config <path>`: Specify custom config file
- `--analysis-only`: Skip fine-tuning, use saved models

To change experiment parameters (alpha range, sample count, batch size, etc.), **edit config.yaml**, do NOT add CLI flags.

### Important config.yaml Sections

```yaml
model:
  name: "google/gemma-3-4b-it"  # HuggingFace model ID
  device: null  # null = auto-detect (cuda/mps/cpu)

task:
  name: "sentiment_positive"  # Options: sentiment_{positive,negative}, instruction_following, qa_factual

evaluation:
  general_dataset: "combined"  # Options: mixed_domain, wikitext, coding, common_knowledge, combined
  batch_size: 32               # Number of texts per forward pass (higher = faster but more memory)
  enable_mixed_precision: true # Use FP16/BF16 for 1.5-2x speedup
  max_length: 1024             # Maximum sequence length for evaluation

fine_tuning:
  num_epochs: 2
  learning_rate: 5.0e-5
  batch_size: 16
  max_length: 512
  data_repetition_factor: 100  # Multiply training examples (30 × 100 = 3000)

alpha_sweep:
  alpha_min: -3.0
  alpha_max: 3.0
  num_samples: 150
  enable_squaring_test: true  # Test M(2α) for self-inverse properties
  sampling_strategy: "uniform"  # Options: uniform, adaptive, bayesian

composition_2d:
  enable: false         # Set true to run 2D composition experiment
  num_samples_per_dim: 30  # 30×30 = 900 evaluations
  enable_analysis: true # Auto-run composition analysis after 2D sweep

composition_3d:
  enable: false         # Set true to run 3D composition experiment
  task_1: "sentiment_negative"      # First task vector
  task_2: "sentiment_positive"      # Second task vector
  task_3: "instruction_following"   # Third task vector
  alpha_min: -1.0
  alpha_max: 1.0
  beta_min: -1.0
  beta_max: 1.0
  gamma_min: -1.0
  gamma_max: 1.0
  num_samples_per_dim: 10  # 10×10×10 = 1,000 evaluations (scales cubically!)

geometry:
  enabled: false        # Enable Riemannian geometry features (experimental)
  metric_type: "fisher_diagonal"  # Options: euclidean | fisher_diagonal | fisher_kfac | fisher_full
  cache_metric: true    # Cache Fisher matrix for reuse

  geodesic_integration:
    enabled: false      # Use geodesic interpolation instead of straight lines
    num_steps: 100      # Runge-Kutta integration steps
    tolerance: 1.0e-6   # Integration error tolerance
```

### Analysis-Only Mode

After running a full experiment, models are saved to `outputs/saved_{base,finetuned}_model/`. Use `--analysis-only` to:
- Skip time-consuming fine-tuning
- Re-run alpha sweep with different parameters
- Generate new visualizations
- Test different sampling strategies

**Workflow**:
1. Run full experiment once: `python main.py`
2. Edit `config.yaml` (change alpha range, num_samples, etc.)
3. Re-analyze: `python main.py --analysis-only`

## Data Organization

```
data/
├── general/       # General evaluation datasets (broad language modeling)
│   ├── mixed_domain.txt
│   ├── wikitext.txt
│   ├── coding.txt
│   ├── common_knowledge.txt
│   └── (combined = all above)
├── tasks/         # Task training datasets
│   ├── sentiment_positive.txt
│   ├── sentiment_negative.txt
│   ├── instruction_following.txt
│   └── qa_factual.txt
└── eval/          # Task evaluation datasets
    ├── sentiment_positive_eval.txt
    └── ...
```

**Datasets are plain text files**: One example per line, used directly by `sitv/data/loader.py`.

## Output Files

All outputs go to `outputs/` directory (gitignored):

### 1D Alpha Sweep Outputs
| File | Description |
|------|-------------|
| `loss_landscape_sweep.png` | 2x2 visualization grid (general/task loss, both linear and log scale) |
| `loss_landscape_results.json` | All α values and metrics |
| `experiment_report.md` | Markdown report for LLM analysis |
| `experiment_metrics.json` | Timing and performance metrics |

### 2D Composition Outputs
| File | Description |
|------|-------------|
| `loss_landscape_2d_sweep.png` | 2D heatmap of loss landscape |
| `loss_landscape_2d_results.json` | All (α, β) values and losses |
| `composition_analysis_*.png` | Composition analysis visualizations (if enable_analysis: true) |
| `analysis_results.json` | Interaction detection and predictions (if enable_analysis: true) |

### 3D Composition Outputs
| File | Description |
|------|-------------|
| `loss_landscape_3d_*.png` | 3D surface plots and projections |
| `loss_landscape_3d_*.html` | Interactive 3D plots (Plotly) |
| `loss_landscape_3d_results.json` | All (α, β, γ) values and losses |

### Multi-Task Comparison
| File | Description |
|------|-------------|
| `comparison_report.md` | Multi-task comparison report |
| `task_comparison_*.png` | Side-by-side task comparison plots |

### Saved Models
| File | Description |
|------|-------------|
| `saved_base_model/` | Saved base model (for --analysis-only) |
| `saved_finetuned_model/` | Saved fine-tuned model (for --analysis-only) |
| `saved_task_{name}_model/` | Additional fine-tuned models (for multi-task experiments) |

## Code Quality Standards

- **Python 3.12+** required (modern type hints)
- **Line length**: 100 characters (ruff configured)
- **Type hints**: Preferred but not required (`mypy` configured with `disallow_untyped_defs = false`)
- **Import order**: Enforced by ruff (isort)
- **Testing**: Use pytest, fixtures in `tests/conftest.py`

## Performance Considerations

### Hardware Detection

Device selection is automatic via `sitv/core/device.py`:
- **CUDA**: NVIDIA GPUs (recommended for large models)
- **MPS**: Apple Silicon (M1/M2/M3)
- **CPU**: Fallback

### Memory Optimization

**In-place parameter modification** is the key optimization:
- Single model instance reused across all α values
- Parameters modified in-place: `param.data.copy_(base_param + alpha * task_vector_param)`
- Achieves **10-100x speedup** vs. reloading model for each α
- See `sitv/experiments/alpha_sweep.py` method `_apply_alpha_scaling()`

### Sampling Strategy Trade-offs

| Strategy | Sample % | Speedup | Use Case |
|----------|----------|---------|----------|
| Uniform | 100% | 1x | Full landscape, research quality |
| Adaptive | 40-60% | 2-3x | Good balance, default for development |
| Bayesian | 10-20% | 5-10x | Find optimal α quickly |

## Common Development Tasks

### Adding a New Task

1. Create training and eval text files in `data/tasks/` and `data/eval/`
2. Add task definition to `sitv/data/tasks.py` (see `PREDEFINED_TASKS` dict)
3. Update `config.yaml` task.name to new task

### Adding a New Sampling Strategy

1. Create new sampler in `sitv/experiments/sampling/`
2. Inherit from `BaseSampler` (see `uniform_sampler.py`, `adaptive_sampler.py`)
3. Implement `sample()` method
4. Register in `sitv/experiments/sampling/__init__.py`
5. Add strategy name to config.yaml options

### Adding a New Experiment Type

1. Create new experiment in `sitv/experiments/`
2. Inherit from `Experiment` base class (`sitv/experiments/base.py`)
3. Implement `run()` method
4. Register in `ExperimentOrchestrator.run()`
5. Add configuration section to `config.yaml`

## Troubleshooting

### Out of Memory Errors

- Reduce `batch_size` in config.yaml
- Reduce `max_length` in config.yaml
- Use smaller model (e.g., "google/gemma-2-2b-it")
- Reduce `num_samples` in alpha_sweep

### Slow Experiments

- Use `sampling_strategy: "adaptive"` or `"bayesian"`
- Reduce alpha range to `[0, 3]` or `[-0.5, 3]` (saves 40-50%)
- Set `composition_2d.enable: false` (skips 2D sweep)
- Use `--analysis-only` after initial run

### Fine-tuning Not Converging

- Increase `num_epochs` in config.yaml
- Increase `data_repetition_factor` (default 100)
- Adjust `learning_rate` (default 5e-5)

## Testing Philosophy

The project has comprehensive test coverage with **14 test modules** (15 files including `conftest.py`):

### Core Tests
- `test_data_models.py`: AlphaSweepResult, ExperimentMetrics, data models
- `test_device.py`: Hardware detection (CUDA/MPS/CPU)
- `test_model_management.py`: Model loading/saving
- `test_task_vector.py`: Task vector computation
- `test_config.py`: Configuration loading and validation
- `test_data_loader.py`: Dataset loading

### Analysis & Reporting Tests
- `test_analyzer.py`: Zero-crossing detection, min loss finding
- `test_markdown_reporter.py`: Markdown report generation
- `test_sampling_strategies.py`: Uniform/adaptive/Bayesian sampling

### I/O Tests
- `test_file_manager.py`: JSON and figure saving

### Geometry Tests (Riemannian features)
- `tests/geometry/test_fisher_metric.py`: Fisher Information Matrix computation
- `tests/geometry/test_geodesic.py`: Geodesic integration

**Integration tests** run via `main.py` with small configs (few samples, small models).

## Additional Documentation

For detailed technical documentation on specific features:

- **[RIEMANNIAN_GEOMETRY_IMPLEMENTATION.md](RIEMANNIAN_GEOMETRY_IMPLEMENTATION.md)**: Complete technical specification of the Riemannian geometry module, Fisher Information Matrix computation, geodesic integration, and implementation details.

- **[IMPLEMENTATION_COMPLETE.md](IMPLEMENTATION_COMPLETE.md)**: Implementation status tracker for all project features and phases.

- **[CHANGELOG.md](CHANGELOG.md)**: Complete version history with detailed change notes for each release.

- **[README.md](README.md)**: Project overview, quick start guide, and research context.
