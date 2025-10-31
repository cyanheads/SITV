# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SITV (Self-Inverse Task Vectors) is a research project exploring loss landscape geometry of neural network task vectors. It maps `L(M_base + αT)` to find optimal scaling factors and investigate whether self-inverse properties from rotation groups emerge in parameter space.

**Core concept**: For a task vector `T = M_finetuned - M_base`, sweep α ∈ [-3.0, 3.0] and plot `L(M_base + αT)` to discover zero-crossings, optimal scaling, and landscape shape.

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

### Modular Package Structure (v0.3.0+)

The project uses a **7-layer service-oriented architecture** with 30+ modules:

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
│   └── sampling/         # Sampling strategies (uniform, adaptive, bayesian)
├── analysis/      # Results analysis
│   ├── analyzer.py  # Zero-crossing detection, min loss finding
│   └── gradient/    # Gradient analysis and critical point detection
├── reporting/     # Report generation
│   └── markdown.py # Markdown report generator
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
   - Computes task vectors via `TaskVectorService`
   - Runs experiments (`AlphaSweepExperiment`, `Composition2DExperiment`)
   - Analyzes results via `ResultAnalyzer`
   - Generates reports via `MarkdownReportGenerator`
   - Saves outputs via `FileManager`

3. **Experiments** (inherit from `sitv/experiments/base.py`)
   - **AlphaSweepExperiment**: 1D sweep of `L(M_base + αT)`
   - **Composition2DExperiment**: 2D sweep of `L(M_base + αT1 + βT2)`
   - Both use **in-place parameter modification** for 10-100x speedup

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
  enable: false  # Set true to run 2D composition experiment
  num_samples_per_dim: 30  # 30×30 = 900 evaluations
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

| File | Description |
|------|-------------|
| `loss_landscape_sweep.png` | 2x2 visualization grid |
| `loss_landscape_results.json` | All α values and metrics |
| `experiment_report.md` | Markdown report for LLM analysis |
| `experiment_metrics.json` | Timing and performance metrics |
| `saved_base_model/` | Saved base model (for --analysis-only) |
| `saved_finetuned_model/` | Saved fine-tuned model (for --analysis-only) |

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

Tests focus on core components and data models:
- `test_data_models.py`: AlphaSweepResult, ExperimentMetrics
- `test_device.py`: Hardware detection
- `test_model_management.py`: Model loading/saving
- `test_task_vector.py`: Task vector computation
- `test_markdown_reporter.py`: Report generation

**Integration tests** run via `main.py` with small configs (few samples, small models).
