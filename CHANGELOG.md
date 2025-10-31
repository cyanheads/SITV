# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
