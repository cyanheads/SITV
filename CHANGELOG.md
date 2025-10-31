# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2025-10-31

### Added

- **2D Composition Reporting** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added comprehensive 2D composition analysis section to markdown reports
  - Implemented `_create_2d_composition_section()` method with 154 lines of detailed analysis
  - Added optimal composition point detection and reporting
  - Added worst-case composition analysis
  - Added axis analysis for pure α and β scaling
  - Added loss landscape statistics (mean, std dev, range)
  - Included interpretation guidance for additive/synergistic/interference effects
  - Enhanced `generate()` method to accept optional `results_2d` parameter

- **Project Documentation** ([CLAUDE.md](CLAUDE.md)):
  - Added comprehensive project documentation for AI assistants (Claude Code)
  - Documented 7-layer architecture with 30+ modules
  - Added development commands, testing, and quality checks
  - Documented configuration philosophy and analysis-only mode
  - Included troubleshooting guides and common development tasks

### Changed

- **Orchestrator Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added `results_2d` instance variable to store 2D composition results
  - Updated report generation to pass 2D results to markdown generator
  - Enhanced data flow between 2D experiments and reporting layer

- **Configuration Tuning** ([config.yaml](config.yaml)):
  - Changed `alpha_min` from -0.5 to -3.0 for full symmetry analysis
  - Reduced `num_samples` from 200 to 100 for faster iteration
  - Increased `threshold` from 0.1 to 0.15 for zero-crossing detection
  - Reduced `num_samples_per_dim` from 60 to 20 (20×20 = 400 evaluations instead of 3600)

### Fixed

- **2D Composition Device Loading** ([sitv/experiments/composition_2d.py](sitv/experiments/composition_2d.py)):
  - Fixed device mismatch issues by pre-loading task vectors to device before composition loop
  - Added explicit device pre-loading for both task vectors
  - Ensures task vectors are on correct device (CPU/CUDA/MPS) for performance and correctness

### Technical Details

- **Reporting**: 2D composition reports now include comprehensive analysis of task vector interactions
- **Configuration**: Optimized for faster iteration while maintaining research quality
- **Device Handling**: Improved device management in 2D experiments prevents cross-device errors

## [0.6.0] - 2025-10-31

### Added

- **Gradient Analysis Module** (`sitv/analysis/gradient/`):
  - Added `NumericalGradientAnalyzer` for computing dL/dα derivatives
  - Added `CriticalPointFinder` for detecting minima, maxima, and inflection points
  - Implemented Gaussian smoothing for robust gradient estimation
  - Added curvature-based critical point classification
  - Exported gradient analysis tools through `sitv/analysis/__init__.py`

- **Error Handling Infrastructure** (`sitv/core/error_handling.py`):
  - Added `FailureTracker` for monitoring evaluation failures with configurable thresholds
  - Implemented `retry_on_cuda_oom` and `retry_on_evaluation_failure` decorators
  - Added `handle_evaluation_error` for graceful error recovery
  - Added `safe_cuda_cleanup` utility for memory management
  - Created custom exception classes: `EvaluationError`, `CUDAOutOfMemoryError`

- **Validation Infrastructure** (`sitv/core/validation.py`):
  - Added comprehensive input validation for alpha ranges, sample counts, and evaluation texts
  - Implemented `validate_alpha_sweep_config` for pre-flight configuration checks
  - Implemented `validate_2d_composition_config` for 2D experiment validation
  - Added `validate_task_vector` for parameter validation
  - Created `ValidationError` exception class for clear error reporting

- **Sampling Strategies** (`sitv/experiments/sampling/`):
  - Implemented `BaseSampler` abstract interface for sampling strategies
  - Added `UniformSampler` for evenly-spaced alpha values (original behavior)
  - Added `AdaptiveSampler` for multi-resolution sampling (40-60% speedup)
  - Added `BayesianSampler` for GP-based optimization (80-90% speedup)
  - Integrated sampling strategy configuration into `AlphaSweepConfig`

- **Configuration Enhancements** (`sitv/experiments/config.py`):
  - Added `SamplingConfig` dataclass for configuring sampling strategies
  - Added `GradientAnalysisConfig` dataclass for gradient analysis settings
  - Enhanced `_get()` helper with type preservation
  - Exported expanded configuration classes through experiments module

- **Dependencies** (`pyproject.toml`):
  - Added scipy>=1.11.0 for gradient analysis (Gaussian smoothing)
  - Added types-PyYAML>=6.0 to dev dependencies for type checking
  - Added optional [bayesian] extra with scikit-learn>=1.0 for Bayesian optimization

### Changed

- **Alpha Sweep Experiment** (`sitv/experiments/alpha_sweep.py`):
  - Integrated validation checks for all configuration parameters
  - Added retry logic for base loss evaluation
  - Added `FailureTracker` for robust error recovery during sweeps
  - Implemented safe evaluation with fallback results for failed alphas
  - Added performance optimization via `preload_task_vector_to_device`
  - Enhanced progress reporting with failure summaries
  - Improved resource cleanup with try-finally blocks

- **Base Experiment Class** (`sitv/experiments/base.py`):
  - Added `preload_task_vector_to_device` method for performance optimization
  - Optimized `apply_task_vector` to avoid repeated device transfers
  - Optimized `apply_2d_composition` to avoid repeated device transfers
  - Updated documentation to recommend pre-loading for tight loops

- **Configuration System** (`config.yaml`):
  - Adjusted fine-tuning parameters: learning_rate (5.0e-5 → 1.0e-4), batch_size (128 → 32), data_repetition_factor (25 → 100)
  - Changed alpha sweep defaults: alpha_min (-3.0 → -0.5), num_samples (300 → 200)
  - Added sampling_strategy configuration (uniform/adaptive/bayesian)
  - Added gradient analysis configuration section
  - Added extensive documentation about alpha range trade-offs
  - Added hardware optimization recommendations
  - Added sampling strategy speed comparison notes

- **Experiment Orchestrator** (`sitv/experiments/orchestrator.py`):
  - Enhanced `_print_header` with detailed configuration summary
  - Added fine-tuning, alpha sweep, and 2D composition parameter display
  - Improved startup visibility of experiment configuration

### Technical Details

- **Error Resilience**: Experiments can now recover from transient CUDA OOM and evaluation failures
- **Performance**: Task vector pre-loading significantly reduces device transfer overhead in tight loops
- **Sampling Efficiency**: Adaptive and Bayesian strategies reduce computation time by 40-90% while maintaining accuracy
- **Validation**: Pre-flight checks catch configuration errors before expensive computation begins
- **Gradient Analysis**: Numerical derivatives enable automatic detection of optimal alpha values
- **Type Safety**: Improved type hints and validation throughout codebase

## [0.5.5] - 2025-10-31

### Changed

- **CLI Interface Simplification** (`sitv/cli/args_parser.py`):
  - Removed ~150 lines of command-line arguments, keeping only `--config` and `--analysis-only`
  - Centralized all experiment configuration to `config.yaml` to eliminate configuration drift
  - Removed CLI arguments for: model selection, device, output directory, alpha sweep parameters, task selection, multi-task mode, 2D composition parameters, and fine-tuning hyperparameters
  - Updated documentation and examples to reflect YAML-first configuration approach
  - Simplified `parse_arguments()` function to minimal implementation

- **Configuration System** (`sitv/experiments/config.py`):
  - Refactored `from_args()` method to load configuration exclusively from YAML
  - Removed CLI override logic for all parameters except config file path and analysis-only flag
  - Enforced single source of truth for experiment configuration

- **Fine-Tuning Configurability** (`sitv/models/fine_tuner.py`):
  - Made `max_length`, `save_strategy`, and `logging_steps` configurable parameters instead of hardcoded values
  - Added parameters to `__init__()` method with proper documentation
  - Updated training arguments to use configurable values
  - Added `max_length` to console output during fine-tuning

- **Experiment Orchestration** (`sitv/experiments/orchestrator.py`):
  - Updated fine-tuner initialization to pass `max_length`, `save_strategy`, and `logging_steps` from config
  - Applied changes to both single-task and 2D composition experiment workflows

### Technical Details

- **Configuration Philosophy**: Shifted from CLI-first to YAML-first approach to prevent configuration inconsistencies
- **Minimal CLI**: Only essential flags (--config, --analysis-only) remain as command-line options
- **Flexibility**: Fine-tuning parameters now configurable via YAML without code changes
- **Backward Compatibility**: Existing config.yaml files work unchanged

## [0.5.4] - 2025-10-31

### Added

- **Per-Category Evaluation** (`sitv/core/evaluation.py`):
  - Added `evaluate_by_category()` method for domain-specific loss analysis
  - Enabled separate evaluation of coding, wikitext, mixed_domain, and common_knowledge domains
  - Returns dictionary mapping category names to their average losses

- **Category-Aware Dataset Loading** (`sitv/data/loader.py`):
  - Added `load_general_with_categories()` method returning both texts and category labels
  - Enhanced `load_general()` to support "combined" option that loads all general datasets (120 examples total)
  - Improved error handling with warnings for missing datasets
  - Updated type hints to modern Python style (`str | None`)

- **Category Breakdown Reporting** (`sitv/reporting/markdown.py`):
  - Added `_create_category_breakdown()` method for per-domain analysis in reports
  - Added "Task & Evaluation" section showing which general evaluation dataset was used
  - Included best α for each category with interpretations
  - Added baseline comparison table showing losses at α≈0 for each domain

- **Category Tracking in Data Models** (`sitv/data/models.py`):
  - Added `category_losses` field to `AlphaSweepResult` for storing per-domain losses
  - Added `general_eval_dataset` field to `ExperimentMetrics` for tracking which dataset was used

### Changed

- **Alpha Sweep Experiments** (`sitv/experiments/alpha_sweep.py`):
  - Updated to accept `general_eval_categories` parameter for category labels
  - Enhanced `evaluate_alpha()` to compute per-category losses when categories provided
  - Integrated category losses into result objects

- **Experiment Orchestration** (`sitv/experiments/orchestrator.py`):
  - Updated to use `load_general_with_categories()` instead of `load_general()`
  - Added general evaluation dataset name to console output and metrics
  - Passes category labels through to alpha sweep experiments

- **Configuration** (`config.yaml`):
  - Changed default `general_dataset` from "mixed_domain" to "combined" for comprehensive evaluation
  - Added documentation about "combined" option (evaluates all 120 examples across all domains)

- **Documentation** (`data/README.md`):
  - Added detailed "Dataset Options" section explaining all 5 choices
  - Enhanced configuration examples with all available options
  - Clarified that "combined" provides most comprehensive evaluation with 120 total examples

### Technical Details

- **Category Support**: All general datasets can now be evaluated separately to measure domain-specific effects
- **Combined Mode**: New "combined" option loads all general datasets together for comprehensive analysis
- **Report Integration**: Category breakdowns automatically appear in markdown reports when using combined datasets
- **Type Safety**: Improved type hints throughout data loading layer

## [0.5.3] - 2025-10-30

### Added

- **Data Loader Architecture** (`sitv/data/loader.py`):
  - Created `DatasetLoader` class for loading datasets from text files
  - Added support for three dataset categories: general evaluation, task training, and task evaluation
  - Implemented file validation, comment support, and UTF-8 encoding
  - Added `list_available()` and `verify_setup()` utilities for dataset management

- **Structured Data Directory** (`data/`):
  - Created `data/general/` for general evaluation datasets (mixed_domain, wikitext, coding, common_knowledge)
  - Created `data/tasks/` for task training data (sentiment_positive, sentiment_negative, instruction_following, qa_factual)
  - Created `data/eval/` for task-specific evaluation data
  - Moved all training and evaluation examples from hardcoded Python to external text files

- **Evaluation Configuration** (`sitv/experiments/config.py`):
  - Added `EvaluationConfig` class with `general_dataset` parameter
  - Integrated evaluation configuration into `ExperimentConfig`
  - Added configuration support via `config.yaml` for general evaluation dataset selection

### Changed

- **Task Definitions** (`sitv/data/tasks.py`):
  - Refactored from 250+ lines of hardcoded text to 85 lines using data loader
  - Replaced inline training/evaluation texts with file-based loading
  - Improved maintainability by separating data from code
  - Retained data repetition factor support for training data

- **Experiment Orchestration** (`sitv/experiments/orchestrator.py`):
  - Updated alpha sweep experiments to use configurable general evaluation datasets for L(α)
  - Updated 2D composition experiments to use configurable general evaluation datasets for L(α,β)
  - Separated general language modeling evaluation from task-specific performance evaluation
  - Task-specific evaluation now measures performance on the trained task only

- **Configuration** (`config.yaml`):
  - Added `evaluation.general_dataset` configuration option (default: "mixed_domain")
  - Added documentation explaining general vs task-specific evaluation
  - Extended task name options to include instruction_following and qa_factual

### Technical Details

- **Architecture**: Clean separation between data and code improves maintainability
- **Extensibility**: New tasks/datasets can be added by creating text files without code changes
- **Evaluation**: General evaluation (L(α)) now uses broad language modeling datasets instead of task-specific texts
- **Data Format**: One example per line, comment support with #, empty lines ignored, UTF-8 encoding

## [0.5.2] - 2025-10-30

### Changed

- **Configurable Training Data Repetition**:
  - Made training data repetition factor configurable instead of hardcoded (previously fixed at 3x)
  - Added `data_repetition_factor` parameter to `config.yaml` with default value of 100
  - Added `--data-repetition-factor` CLI argument for runtime override
  - Updated `get_predefined_tasks()` function to accept `data_repetition_factor` parameter
  - Applied configurable repetition to all task types (sentiment_positive, sentiment_negative, instruction_following, qa_factual)
  - Default configuration now produces 3,000 training examples per task (30 unique × 100 repetitions) instead of 90 (30 unique × 3)

### Technical Details

- **Configuration**: New `fine_tuning.data_repetition_factor` field in config.yaml
- **CLI Integration**: Added argument parsing for data repetition factor
- **Data Pipeline**: Updated `FineTuningConfig` class and `ExperimentOrchestrator` to pass repetition factor throughout the pipeline
- **Backward Compatible**: Default value of 100 maintains reasonable training dataset size while allowing experimentation

## [0.5.1] - 2025-10-30

### Changed

- **Enhanced Task Training Data** (`sitv/data/tasks.py`):
  - Replaced simple, generic training examples with detailed, realistic examples across all task types
  - Updated `sentiment_positive` examples from basic praise to detailed product/service reviews (15 unique examples)
  - Updated `sentiment_negative` examples from simple complaints to detailed negative experiences (30 unique examples)
  - Updated `instruction_following` examples from basic colors/animals to technical programming concepts (30 unique examples covering web frameworks, cloud platforms, ML concepts, etc.)
  - Updated `qa_factual` examples from simple trivia to detailed technical Q&A about science, programming, and technology (30 unique examples)
  - Enhanced evaluation texts across all tasks with similarly detailed and realistic examples

### Fixed

- **Documentation Formatting** (`README.md`):
  - Fixed line break in project description paragraph for better readability

### Technical Details

- **Training Data Quality**: Significantly improved realism and diversity of training examples to better represent real-world use cases
- **Example Count**: Each task now contains 30 unique, detailed examples (3x multiplication factor maintained for training texts)
- **Content Depth**: Examples now include technical terminology, realistic scenarios, and contextual detail appropriate for modern AI training

## [0.5.0] - 2025-10-30

### Added

- **Configuration File Support** (`config.yaml`):
  - Added YAML-based configuration system as single source of truth
  - Created default `config.yaml` with optimized settings for modern GPUs (H200 SXM specs)
  - Added `load_config_yaml()`, `reload_config()`, and `_get()` helper functions in `sitv/experiments/config.py`
  - Command-line arguments now override config.yaml values
  - Added `--config` CLI argument for custom config file paths

- **Complete 2D Composition Experiment** (`sitv/experiments/orchestrator.py`):
  - Implemented full 2D composition experiment functionality (previously placeholder)
  - Automated second task selection (sentiment_negative ↔ sentiment_positive)
  - Added fine-tuning on second task and second task vector computation
  - Generated 2D heatmap visualizations and JSON result export
  - Removed placeholder code and implemented end-to-end workflow for exploring L(M_base + α·T1 + β·T2)

- **Enhanced Reporting** (`sitv/reporting/markdown.py`):
  - Added training history section with step-by-step metrics table
  - Added alpha sweep details section with sample data points and key metrics
  - Enhanced configuration section with comprehensive model, training, and task vector details
  - Added safety guards for division by zero and None values in all metric calculations

### Changed

- **Default Configuration**:
  - Changed default model from `Qwen/Qwen2.5-0.5B` to `google/gemma-3-4b-it`
  - Optimized training hyperparameters for modern GPUs:
    - Epochs: 3 → 2
    - Learning rate: 1e-4 → 5e-5
    - Batch size: 4 → 16
    - Max length: 128 → 512 tokens
  - Increased alpha sweep resolution: 100 → 150 samples
  - Increased 2D composition resolution: 20×20 → 30×30 grid

- **Configuration System** (`sitv/experiments/config.py`):
  - Refactored all dataclass fields to use `field(default_factory=...)` with YAML config loading
  - Made `config.yaml` the primary configuration method
  - CLI arguments now serve as overrides rather than defaults
  - Updated docstrings to reflect config.yaml priority

- **Documentation** (`README.md`):
  - Added comprehensive "Configuration File (config.yaml)" section with example
  - Restructured configuration examples to emphasize YAML-first approach
  - Changed "Inspired by" to "Loosely inspired by" for the Eckmann & Tlusty (2025) paper
  - Removed inline Python configuration section (now outdated)
  - Added examples for custom config file usage and CLI overrides

### Technical Details

- **Configuration**: YAML-based with automatic loading from project root or custom path
- **2D Composition**: Full implementation with automatic second task fine-tuning and vector computation
- **Reporting**: Enhanced with training history (gradient norms, loss progression) and detailed alpha sweep metrics
- **Default Model**: Upgraded to gemma-3-4b-it for improved performance and modern architecture support

## [0.4.0] - 2025-10-30

### Added

- **Fine-Tuning Implementation** (`sitv/models/fine_tuner.py`):
  - Completed full fine-tuning implementation with `FineTuner` service
  - Added `TextDataset` class for tokenizing training data
  - Integrated HuggingFace Trainer with custom training configuration
  - Added comprehensive metrics collection (training loss, steps, duration)
  - Implemented `FineTuningProgressCallback` for detailed progress tracking with ETA
  - Support for gradient checkpointing and bfloat16 precision

- **Enhanced Reporting** (`sitv/reporting/markdown.py`):
  - Added statistical summary section with loss/functional return/task performance distributions
  - Added squaring test analysis section for [W(λ)]² = I analog investigation
  - Added theoretical connection section linking to Eckmann & Tlusty (2025) paper
  - Enhanced markdown formatting with proper tables and interpretation

- **Enhanced Visualization** (`sitv/visualization/plotter.py`):
  - Redesigned 1D sweep plots with 6-panel layout for squaring test mode
  - Added comparison plots for L(α) vs L(2α)
  - Added squaring functional return visualization
  - Enhanced plot annotations with highlighted zero-crossings and minima
  - Improved color schemes and marker styling for key features

- **Progress Tracking** (`sitv/utils/progress.py`):
  - Added `FineTuningProgressCallback` for real-time training progress
  - Implemented ETA calculation during fine-tuning
  - Added training history tracking for post-training analysis
  - Enhanced console output with epoch boundaries and step-level metrics

### Changed

- **Experiment Orchestration** (`sitv/experiments/orchestrator.py`):
  - Integrated FineTuner service into experiment workflow
  - Replaced placeholder fine-tuning logic with full implementation
  - Added model saving functionality after fine-tuning
  - Enhanced 2D composition documentation with example integration code
  - Updated metrics collection to include fine-tuning statistics

### Removed

- Removed TODO comments from fine-tuning, reporting, and visualization modules
- Removed placeholder implementations and NotImplementedError exceptions

### Fixed

- Fine-tuning now fully functional instead of raising NotImplementedError
- Models are properly saved after fine-tuning for future analysis

### Technical Details

- **Fine-Tuning**: Full integration with HuggingFace Transformers Trainer API
- **Metrics**: Comprehensive tracking of training metrics (loss, steps, duration, LR, etc.)
- **Visualization**: 2x3 grid layout for squaring test, 2x2 for standard mode
- **Progress**: Real-time ETA and step-level progress reporting during training

## [0.3.0] - 2025-10-30

### Changed

- **Major Architectural Refactoring**: Transformed monolithic 2,232-line `main.py` into modular package architecture
- Reduced `main.py` from 2,232 lines to 44 lines (98% reduction) - now serves as thin entry point
- Created `sitv/` package with 30+ modules organized in 7 architectural layers
- Reorganized codebase following Service Layer, Strategy, and Dependency Injection patterns
- Implemented SOLID principles throughout package design
- Improved separation of concerns with clear layer boundaries

### Added

- **Data Layer** (`sitv/data/`):
  - `models.py`: Core data structures (AlphaSweepResult, TaskDefinition, ExperimentMetrics, etc.)
  - `tasks.py`: Predefined task definitions
- **Core Services** (`sitv/core/`):
  - `device.py`: Hardware detection and device management
  - `task_vector.py`: Task vector computation and operations
  - `evaluation.py`: Model evaluation and perplexity calculation
- **Model Management** (`sitv/models/`):
  - `loader.py`: Model loading, saving, and parameter counting
  - `fine_tuner.py`: Placeholder for fine-tuning migration
- **Experiments** (`sitv/experiments/`):
  - `base.py`: Abstract Experiment base class with template method pattern
  - `config.py`: Configuration classes for all experiment types
  - `alpha_sweep.py`: 1D alpha sweep experiment implementation
  - `composition_2d.py`: 2D composition experiment implementation
  - `orchestrator.py`: ExperimentOrchestrator for workflow coordination
- **Analysis & Reporting** (`sitv/analysis/`, `sitv/reporting/`, `sitv/visualization/`):
  - `analyzer.py`: Results analysis, zero-crossing detection, min loss finding
  - `markdown.py`: Markdown report generation
  - `plotter.py`: Visualization and plotting utilities
- **Infrastructure** (`sitv/io/`, `sitv/utils/`):
  - `file_manager.py`: JSON and figure saving
  - `paths.py`: Path management utilities
  - `console.py`: Console output formatting
  - `progress.py`: Progress tracking with ETA calculation
  - `timing.py`: Timing utilities
- **CLI** (`sitv/cli/`):
  - `args_parser.py`: Centralized argument parsing
- **Test Suite** (`tests/`):
  - 22 passing tests covering data models, device management, model operations, and task vectors
  - Test fixtures for models, tokenizers, and task vectors
  - 100% test pass rate with pytest
- **Documentation**:
  - `ARCHITECTURE.md`: Comprehensive 6-layer architecture documentation
  - `REFACTORING_SUMMARY.md`: Complete refactoring metrics and summary
  - `archive/main_original.py`: Backup of original monolithic implementation
- `.gitignore`: Added `.venv/` pattern, removed `data/` from ignore list

### Fixed

- Device mismatch error in task vector computation (handles models on cuda/meta/cpu)
- AttributeError with offloaded model parameters during checkpoint saving

### Technical Details

- **Lines of Code**: 3,817 lines across 30+ modules (up from 2,232 monolithic)
- **Modules**: 30+ focused, single-responsibility modules
- **Test Coverage**: 22 tests (0 → 22)
- **Architecture Layers**: 7 distinct layers (Presentation → Data)
- **Design Patterns**: Service Layer, Strategy, Template Method, Facade, Dependency Injection, Repository
- **Maintainability**: Significantly improved with modular structure and clear separation of concerns

## [0.2.0] - 2025-10-30

### Changed

- Simplified experiment from dual-version testing to direct loss landscape sweep
- Refactored core question from "Do self-inverse λ values exist?" to "What does the loss landscape look like along the task vector direction?"
- Renamed parameter from λ (lambda) to α (alpha) throughout codebase for clarity
- Expanded sweep range from [-2.0, 2.0] to [-3.0, 3.0] with increased sampling (50 → 100 points)
- Optimized for 10-100x performance improvement via in-place parameter modification instead of model reloading
- Simplified data structure from `InverseResult` to `AlphaSweepResult` with focused metrics
- Updated visualization from 2x3 comparison grid to cleaner 2x2 layout emphasizing loss landscape
- Streamlined documentation to directly answer: "Does L(α) cross L(M_base) at any α ≠ 0?"

### Removed

- Dual-version testing framework (linear vs compositional task vector doubling)
- Geometric distance metrics (not relevant for functional loss analysis)
- Compositional application functions (`apply_task_vector_compositional`)
- Utility score calculations
- Complex comparison plots between linear and compositional versions

## [0.1.0] - 2025-10-30

### Added

- Initial implementation of self-inverse task vectors experiment
- Research code exploring functional return in neural network parameter space
- Inspired by "Walks in Rotation Spaces Return Home when Doubled and Scaled" (Eckmann & Tlusty, 2025)
- Support for both linear (2λT) and compositional (λT + λ²T) task vector transformations
- Comprehensive evaluation framework with geometric and functional metrics
- Fine-tuning pipeline for creating task vectors from base models
- Visualization tools for loss landscape analysis across λ values
- Support for CUDA, Apple Silicon (MPS), and CPU devices
- Sentiment analysis task implementation as proof-of-concept
- Detailed result tracking with JSON export functionality
- Project configuration with pyproject.toml (Python 3.12+)
- Comprehensive .gitignore and .gitattributes for repository management
