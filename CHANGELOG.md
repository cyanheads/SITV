# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
