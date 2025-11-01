# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.15.0] - 2025-11-01

### Added

- **T2 1D Sweep Data Support** ([sitv/analysis/composition_analyzer.py](sitv/analysis/composition_analyzer.py)):
  - Added `t2_1d_path` parameter to `CompositionAnalyzer.__init__()` for loading T2's 1D alpha sweep results
  - Implemented `_load_t2_1d()` method with auto-detection of conventional filenames
  - Added `--t2-1d` command-line argument to composition analysis script
  - Enhanced `run_full_analysis()` to display T2 properties when available
  - Improved "independent" prediction strategy using actual T2 optimal alpha when provided
  - Falls back to symmetric prediction when T2 data unavailable

- **Percentile Analysis Tables** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added comprehensive percentile tables to statistical summary (10th, 25th, 50th, 75th, 90th)
  - Shows alpha values at each loss percentile for pattern recognition
  - Added ΔL from base column showing deviation from base model loss
  - Extended percentile analysis to functional return and task performance distributions

- **Category Evolution Tracking** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added category loss evolution table showing how each domain changes across key alpha values
  - Samples key alpha values (-3.0, -2.0, -1.0, 0.0, +1.0, +2.0, +3.0) for consistent comparison
  - Marks optimal alpha in evolution table for easy identification
  - Enables pattern analysis across different evaluation domains

- **2D Axis Cross-Sections** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added detailed axis slice tables for 2D composition landscapes
  - Shows pure T1 effect along α-axis (β ≈ 0)
  - Shows pure T2 effect along β-axis (α ≈ 0)
  - Shows balanced composition along diagonal (α ≈ β)
  - Each slice includes loss, ΔL from base, and perplexity

- **Helper Functions** ([sitv/analysis/composition_analyzer.py](sitv/analysis/composition_analyzer.py)):
  - Added `_closest_index_to_zero()` for robust base loss calculation
  - Added `_robust_unique_sorted()` for handling floating-point grid jitter
  - Added `_polyfit_curvature()` for local curvature estimation via quadratic fit

### Changed

- **Enhanced Numerical Robustness** ([sitv/analysis/composition_analyzer.py](sitv/analysis/composition_analyzer.py)):
  - Improved `extract_1d_properties()` with better base loss calculation (finds alpha≈0 instead of first item)
  - Enhanced zero-crossing detection with linear interpolation and finite value checks
  - Improved curvature estimation using local quadratic fit instead of simple finite difference
  - Better NaN handling throughout 2D property extraction
  - Robust grid reconstruction with automatic rounding to handle floating-point jitter

- **Enhanced Visualizations** ([sitv/analysis/composition_analyzer.py](sitv/analysis/composition_analyzer.py)):
  - Improved color scaling using robust percentile-based limits (1st-99th percentile)
  - Added NaN masking for cleaner display of incomplete grids
  - Symmetric color limits for interaction heatmaps centered on zero
  - Better axis cross-section plotting with NaN-aware interpolation

- **Report Sampling Strategy** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Changed alpha sweep table to prioritize key alpha values (-3.0 to +3.0)
  - Added intelligent sampling combining key values, optimal points, and evenly-spaced samples
  - Increased display limits for better data visibility:
    - `MAX_ZERO_CROSSINGS_DISPLAY`: 3 → 8
    - `MAX_SAMPLE_DATA_POINTS`: 20 → 30
    - `MAX_SQUARING_RETURNS_DISPLAY`: 10 → 15
    - `TRAINING_STEP_INTERVAL`: 5 → 3
    - `MAX_TRAINING_MILESTONES`: 25 → 35
    - `GEODESIC_TABLE_SAMPLE_INTERVAL`: 5 → 3

- **Squaring Test Analysis** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Enhanced squaring return table with L(α) column for better comparison
  - Added pattern recognition guidance for |L(2α) - L_base| << |L(α) - L_base|
  - Improved interpretation of functional return behavior

- **Code Organization** ([sitv/analysis/composition_analyzer.py](sitv/analysis/composition_analyzer.py)):
  - Added `from __future__ import annotations` for improved type hint support
  - Reorganized imports following standard conventions
  - Added section comments for better code navigation (helpers, 1D, 2D, cross-sections, etc.)
  - Enhanced docstrings with parameter descriptions

### Technical Details

- **T2 Data Integration**: Enables asymmetric task pair analysis with actual optimal scaling for both tasks
- **Numerical Stability**: Robust handling of floating-point grids and NaN values throughout analysis pipeline
- **Visualization Quality**: Percentile-based color scaling prevents outliers from dominating color maps
- **Report Density**: Increased sample counts provide more comprehensive data for LLM pattern recognition
- **Key Alpha Values**: Standardized sampling at [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, +0.5, +1.0, +1.5, +2.0, +2.5, +3.0]

## [0.14.4] - 2025-11-01

### Fixed

- **Numerical Stability in Alpha Sweep** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Enhanced loss validation with `math.isfinite()` checks for `loss_alpha` and `task_eval_loss`
  - Changed fallback values from `0.0` to `float("inf")` for unavailable metrics (loss_2alpha, functional_return_2alpha, perplexities)
  - Improved perplexity overflow protection with combined `math.isfinite()` and threshold checks
  - Added explicit variable initialization to prevent UnboundLocalError on early failures
  - Enhanced error messages for conservative "unavailable" markers vs misleading zeros

- **Dtype Preservation in Symmetry Analysis** ([sitv/geometry/symmetry.py](sitv/geometry/symmetry.py)):
  - Fixed orthogonal rotation matrix dtype mismatch
  - Changed `Q.to(param.device)` to `Q.to(device=param.device, dtype=param.dtype)` to preserve parameter dtype during rotation

### Changed

- **Geodesic Configuration Alignment** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Unified geodesic gating logic with orchestrator pattern
  - Changed to require both `geometry.enabled` AND `geodesic_integration.enabled` for geodesic interpolation
  - Replaced upfront Christoffel computation with deferred on-demand approach
  - Improved configuration consistency across experiment modules

- **Progress Reporting Control** ([sitv/geometry/metric.py](sitv/geometry/metric.py)):
  - Added `show_progress: bool = True` parameter to all Fisher computation methods
  - Allows suppressing progress bars during nested Fisher computations (e.g., Christoffel symbol computation)
  - Applied to `compute_fisher_information_matrix()`, `_compute_diagonal_fisher()`, `_compute_kfac_fisher()`, `_compute_full_fisher()`
  - Christoffel computation now uses `show_progress=False` for cleaner nested output
  - Prevents cluttered console output during complex geometric computations

- **Code Quality Improvements** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Added `from __future__ import annotations` for improved type hint support
  - Added `import math` for explicit finite value checking
  - Enhanced code formatting with trailing commas for better diffs
  - Improved readability with consistent parameter formatting

### Technical Details

- **Error Handling**: Robust handling of non-finite loss values with appropriate fallback markers
- **Numerical Stability**: Enhanced overflow protection and validation throughout evaluation pipeline
- **Type Safety**: Improved dtype preservation in tensor operations
- **Configuration**: Unified geodesic configuration logic for consistency
- **User Experience**: Cleaner progress reporting for complex multi-level computations

## [0.14.3] - 2025-11-01

### Fixed

- **Critical: Base Model Contamination in Composition Experiments** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Fixed base model contamination bug where alpha sweep modified base_model in-place, causing subsequent 2D/3D composition experiments to use contaminated parameters (stuck at last alpha value)
  - Implemented pristine base model pattern: separate `base_for_eval` and `base_for_ft` instances prevent mutation
  - Updated `_run_2d_composition()` to load fresh base models before fine-tuning task 2 and before running composition experiment
  - Updated `_run_3d_composition()` to load pristine base for evaluation with separate copies for fine-tuning tasks 2 and 3
  - Modified `_fine_tune_and_compute_task_vector()` helper to never mutate caller-provided models
  - **Impact**: All 2D and 3D composition results prior to this fix were scientifically incorrect

- **Geometry Configuration Bug** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Fixed incorrect reference to `self.config.geometry.use_riemannian` (changed to `self.config.geometry.enabled`)
  - Fixed incorrect reference to `self.config.geometry.use_geodesics` (changed to `self.config.geometry.geodesic_integration.enabled`)
  - Fixed potential division by zero in Riemannian magnitude ratio calculation

### Changed

- **Code Quality and Type Safety** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added `from __future__ import annotations` for improved type hint support
  - Enhanced type annotations with explicit generic types (e.g., `dict[str, dict[str, torch.Tensor]]`)
  - Improved docstrings: removed verbose parameter documentation, added concise notes about key behaviors
  - Removed unused variables (`ft_metrics_2`, `metadata_2d`, `metadata_3d`)
  - Consistent code formatting (spacing, line breaks, import organization)

- **Reproducibility Improvements** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Enhanced `set_random_seed()` with `PYTHONHASHSEED` environment variable for complete Python hash reproducibility
  - Added conditional CUDA checks (`torch.cuda.is_available()`) before setting CUDA-specific seeds
  - Added conditional cuDNN checks to prevent errors on non-CUDA systems
  - Added comment about `torch.use_deterministic_algorithms()` for strict determinism

- **Memory Management** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added MPS (Apple Silicon) cache clearing support alongside existing CUDA cache clearing
  - Improved accelerator cache handling with proper error handling for unsupported platforms
  - Enhanced GPU memory cleanup after Fisher metric computation

- **Parameter Protection** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Changed Christoffel computation to use `.clone()` when passing base_params to prevent in-place mutation
  - Changed curvature analysis to use `.clone()` when extracting base parameters

- **Analysis-Only Mode Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added guards to skip 2D/3D composition experiments when `analysis_only=True` (these require fresh fine-tuning)
  - Added informative warning messages explaining why composition experiments are skipped

### Technical Details

- **Bug Severity**: Critical - all 2D/3D composition results before this fix used contaminated base models
- **Root Cause**: Alpha sweep modifies base_model in-place via `Experiment.apply_task_vector()`, leaving parameters at last α value
- **Solution**: Load pristine base models immediately before composition experiments and fine-tuning operations
- **Verification**: Task vector magnitudes should now be independent of alpha sweep execution order
- **Code Quality**: Improved type safety, documentation clarity, and code consistency throughout orchestrator

## [0.14.2] - 2025-11-01

### Added

- **Christoffel Computation Configuration** ([sitv/geometry/config.py](sitv/geometry/config.py), [config.yaml](config.yaml)):
  - Added `ChristoffelComputationConfig` dataclass for fine-grained control over curvature detection
  - Added `skip_vision_tower` flag to skip vision_tower parameters (not modified during text fine-tuning)
  - Added `skip_frozen` flag to filter parameters where requires_grad=False
  - Added `num_samples` parameter for Fisher computation sample count (reduced from 100 to 20)
  - Added `parameter_sample_fraction` for statistical sampling of parameters (1.0 = all, 0.1 = 10%)
  - Added `max_parameters` limit for quick testing (null = unlimited)
  - Documented 90% speedup achieved by skipping vision_tower (21 min → 2-3 min)

- **Enhanced Progress Reporting** ([sitv/core/task_vector.py](sitv/core/task_vector.py), [sitv/geometry/metric.py](sitv/geometry/metric.py)):
  - Added padded console output with flush for cleaner progress display
  - Added parameter name truncation for long names (60 char limit)
  - Added multi-level progress tracking (overall + current parameter)
  - Improved formatting to clear previous longer lines

### Changed

- **Comprehensive README Documentation** ([README.md](README.md)):
  - Updated project description with "anthill hypothesis" and experimental cartography framing
  - Added complete Riemannian geometry section documenting Fisher metrics, geodesics, and curvature analysis
  - Added multi-dimensional task vector composition section (2D and 3D)
  - Enhanced output files documentation with all experiment types
  - Added comprehensive interpretation guides for geometry and composition results
  - Updated project structure with current module counts (53 modules, 14,742 lines of code)
  - Added changelog reference link for version history

- **Christoffel Symbol Computation Optimization** ([sitv/geometry/metric.py](sitv/geometry/metric.py)):
  - Implemented parameter filtering logic reducing computation from ~21 min to 2-3 min (90% speedup)
  - Added statistics logging showing total, filtered, and processed parameter counts
  - Added support for parameter sampling (parameter_sample_fraction and max_parameters)
  - Enhanced progress display with overall progress and current parameter details
  - Added config parameter to `compute_christoffel_symbols_finite_diff()` method
  - Improved robustness with type checking for Fisher metric tensor values

- **Orchestrator Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Updated to use `christoffel_computation.num_samples` from config instead of hardcoded 100
  - Added type casting for Fisher metrics when computing Riemannian norm
  - Improved GPU memory management by handling metadata in Fisher metrics
  - Enhanced Christoffel computation to respect config-driven filtering and sampling

- **Type Safety Improvements** (5 files):
  - Enhanced type annotations in geometry modules for better static analysis
  - Fixed type handling for Fisher metrics containing metadata dictionaries
  - Improved type safety in symmetry analysis with explicit tensor typing
  - Added proper type casting for geodesic task vector operations
  - Fixed plotly import type annotations

- **Geometry Configuration** ([sitv/geometry/config.py](sitv/geometry/config.py)):
  - Integrated `ChristoffelComputationConfig` into `GeometryConfig` dataclass
  - Added comprehensive serialization for christoffel_computation settings
  - Enhanced configuration export with all filtering and sampling parameters

### Technical Details

- **Performance**: 90% speedup in Christoffel computation by skipping vision_tower parameters (21 min → 2-3 min)
- **Parameter Filtering**: Automatic detection of parameters not modified during text fine-tuning
- **Statistical Sampling**: Support for partial parameter sampling for quick curvature detection
- **Documentation**: Complete feature coverage in README for all versions through v0.14.1
- **Type Safety**: Improved static type checking throughout geometry module
- **User Experience**: Better progress reporting with clearer formatting and multi-level tracking

## [0.14.1] - 2025-11-01

### Added

- **Hyperparameter Analysis Reporting** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added `_create_hyperparameter_analysis()` method generating comprehensive hyperparameter sensitivity analysis
  - Implemented training convergence detection with loss stability metrics
  - Added hyperparameter sensitivity guidance (learning rate, epochs, batch size)
  - Created recommendations table for different training scenarios (quick test, standard, high quality, etc.)
  - Included convergence status indicators (✅ Converged / ⚠️ May need more epochs)
  - Added interpretation sections explaining convergence quality and recommendations

- **Enhanced Geodesic Integration Documentation** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Expanded geodesic integration section with detailed implementation documentation
  - Added algorithm description (RK4 integration of geodesic equation)
  - Included mathematical formulation of geodesic equation and Christoffel symbols
  - Added convergence criteria and computational cost analysis
  - Documented varying-metric geodesics for curvature detection
  - Added configuration reference with all geodesic integration parameters
  - Included theoretical background explaining geodesics vs Euclidean paths

### Changed

- **Project Documentation** ([CLAUDE.md](CLAUDE.md)):
  - Enhanced project overview with clearer description of research goals
  - Reframed SITV as "experimental cartography" of neural network internals
  - Added explicit "anthill hypothesis" describing branching structure theory
  - Clarified end goal: discovering shortcuts for LLM training via geometric understanding
  - Improved core methodology description with explicit 2D/3D composition mention

### Removed

- **Documentation Consolidation**:
  - Removed [RIEMANNIAN_GEOMETRY_IMPLEMENTATION.md](RIEMANNIAN_GEOMETRY_IMPLEMENTATION.md) (541 lines)
  - Content was comprehensive technical specification of Riemannian geometry module
  - Information now covered in CLAUDE.md and inline code documentation
  - Reduces documentation duplication and maintenance burden

### Technical Details

- **Reporting Enhancement**: Hyperparameter analysis helps users understand training convergence and optimize experiment parameters
- **Documentation Quality**: Enhanced geodesic integration documentation provides implementation-level details for researchers
- **Project Clarity**: Updated CLAUDE.md better communicates the branching structure hypothesis and research vision
- **Maintenance**: Consolidating Riemannian geometry documentation reduces duplicate information across files

## [0.14.0] - 2025-11-01

### Added

- **Symmetry Detection Module** ([sitv/geometry/symmetry.py](sitv/geometry/symmetry.py)):
  - Added `SymmetryAnalyzer` class for detecting parameter space symmetries
  - Implemented rotation symmetry detection via random orthogonal transformations
  - Implemented permutation symmetry detection with neuron reordering tests
  - Implemented scaling symmetry detection for layer-wise rescaling
  - Added quotient space projection to canonical parameter forms
  - Supports tolerance-based symmetry detection (configurable threshold)
  - Generates comprehensive symmetry analysis reports with violation tracking

- **Quotient Space Operations** ([sitv/geometry/symmetry.py](sitv/geometry/symmetry.py)):
  - Added `compute_canonical_representative()` for symmetry-invariant parameter forms
  - Added `project_to_quotient_space()` for working in reduced parameter space
  - Permutation quotient: sorts neurons by weight magnitude (decreasing order)
  - Scaling quotient: normalizes parameters to unit Frobenius norm
  - Addresses theoretical critique from arXiv:2506.13018 on parameter redundancy

- **Symmetry Configuration** ([sitv/geometry/config.py](sitv/geometry/config.py), [config.yaml](config.yaml)):
  - Updated `SymmetryAnalysisConfig` with full configuration options
  - Added `detect_rotations`, `detect_permutations`, `detect_scaling` flags
  - Added `quotient_space` flag for canonical form projection
  - Added `symmetry_tolerance` parameter for invariance threshold

- **Symmetry Tests** ([tests/geometry/test_symmetry.py](tests/geometry/test_symmetry.py)):
  - Created comprehensive test suite with 25+ tests
  - Tests for orthogonal matrix generation (QR decomposition)
  - Tests for all three symmetry types (rotation, permutation, scaling)
  - Tests for canonical form computation and idempotence
  - Tests for quotient space projection
  - Mock evaluator for isolated symmetry testing

### Changed

- **Geometry Module Exports** ([sitv/geometry/__init__.py](sitv/geometry/__init__.py)):
  - Exported `SymmetryAnalyzer` and `SymmetryAnalysisConfig`
  - Updated module docstring to describe symmetry detection capabilities

### Technical Details

- **Rotation Symmetry**: Uses QR decomposition to generate random orthogonal matrices, tests L(R·θ) ≈ L(θ)
- **Permutation Symmetry**: Randomly permutes neurons within layers, propagates to connected layers
- **Scaling Symmetry**: Tests layer-wise rescaling invariance L(λ·θ) ≈ L(θ)
- **Quotient Space**: Reduces parameter redundancy by working modulo symmetries
- **Performance**: ~30 evaluations for full symmetry analysis (10 tests × 3 types)

## [0.13.0] - 2025-11-01

### Added

- **Curvature Analysis Module** ([sitv/geometry/curvature.py](sitv/geometry/curvature.py)):
  - Added `CurvatureAnalyzer` class for computing Riemannian curvature on parameter manifolds
  - Implemented sectional curvature K(X,Y) computation via finite differences
  - Implemented Ricci curvature (trace of sectional curvatures)
  - Implemented scalar curvature (total curvature at a point)
  - Added curvature distribution estimation via random tangent sampling
  - Supports caching of curvature results for reuse
  - Provides human-readable interpretations (flat, positively curved, negatively curved)

- **Geometric Helper Methods** ([sitv/geometry/curvature.py](sitv/geometry/curvature.py)):
  - Added random tangent vector sampling on parameter manifolds
  - Added Gram-Schmidt orthogonalization in Riemannian metric
  - Added tangent vector normalization (unit length in metric)
  - Added Riemannian inner product computation (⟨v,w⟩_g)
  - Added Riemann tensor component approximation via parallel transport

- **Curvature Configuration** ([sitv/geometry/config.py](sitv/geometry/config.py), [config.yaml](config.yaml)):
  - Updated `CurvatureAnalysisConfig` with full implementation
  - Added `compute_sectional`, `compute_ricci`, `compute_scalar` flags
  - Added `num_tangent_samples` parameter for statistical curvature estimation
  - Default: sectional curvature only (fast), Ricci/scalar optional (expensive)

- **Curvature Tests** ([tests/geometry/test_curvature.py](tests/geometry/test_curvature.py)):
  - Created comprehensive test suite with 20+ tests
  - Tests for sectional curvature properties (symmetry, finite values, zero for parallel vectors)
  - Tests for Riemannian inner product (symmetry, positive-definiteness)
  - Tests for tangent vector operations (normalization, orthogonalization)
  - Tests for curvature distribution estimation
  - Tests for caching functionality
  - Parametrized tests for multiple sample counts

### Changed

- **Geometry Module Exports** ([sitv/geometry/__init__.py](sitv/geometry/__init__.py)):
  - Exported `CurvatureAnalyzer` and `CurvatureAnalysisConfig`
  - Updated module docstring to describe curvature analysis capabilities

### Technical Details

- **Sectional Curvature**: Measures curvature of 2D planes, K(X,Y) = R(X,Y,Y,X) / (||X||²||Y||² - ⟨X,Y⟩²)
- **Finite Differences**: Approximates Riemann tensor via metric variation along parallelograms
- **Sample-Based Estimation**: Uses 10-20 random tangent pairs for statistical curvature distribution
- **Computational Cost**: O(k·n) for k samples, sectional curvature is fast and informative
- **Interpretation**: Positive = sphere-like (geodesics converge), Negative = hyperbolic (geodesics diverge)

## [0.12.0] - 2025-11-01

### Added

- **Composition Analysis Module** ([sitv/reporting/comparison_report.py](sitv/reporting/comparison_report.py)):
  - Added `ComparisonReportGenerator` class for multi-task comparison reports
  - Refactored comparison report generation logic from scripts into reusable reporting module
  - Generates comparative metrics tables across multiple task vectors
  - Provides detailed per-task analysis with optimal scaling factors and zero-crossings
  - Analyzes variance in optimal α values and self-inverse properties across tasks
  - Exported through `sitv.reporting` module for consistent API

- **Automatic Composition Analysis** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added `_run_composition_analysis()` method to automatically analyze 2D composition results
  - Integrated `CompositionAnalyzer` to detect task interaction patterns after 2D sweeps
  - Added `composition_analysis` field to store analysis results
  - Runs automatically when `composition_2d.enable_analysis` is enabled
  - Detects interaction types (additive vs non-additive) and generates predictions
  - Saves analysis results and visualizations to output directory

- **Enhanced 2D Composition Reporting** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added `_create_composition_analysis_section()` method for comprehensive composition analysis in reports
  - Reports 1D task vector properties (optimal α, curvature, zero-crossings)
  - Reports 2D composition properties (optimal (α, β), loss at optimal)
  - Analyzes task independence with R² score and interaction RMS
  - Generates prediction tables comparing strategies (independent, weighted, balanced)
  - Provides practical implications based on interaction strength
  - References saved analysis files and visualizations

### Changed

- **Configuration Enhancement** ([config.yaml](config.yaml), [sitv/experiments/config.py](sitv/experiments/config.py)):
  - Added `enable_analysis` flag to composition_2d configuration (default: true)
  - Enables automatic composition analysis after 2D sweeps when enabled
  - Added `enable_analysis` field to `Composition2DConfig` dataclass

- **Report Generation Pipeline** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Updated `_generate_outputs()` to pass composition analysis results to markdown generator
  - Enhanced report generation to include composition analysis section when available
  - Reports now show predictability of 2D optimal composition from 1D properties

- **Comparison Script Refactoring** ([scripts/compare_tasks.py](scripts/compare_tasks.py)):
  - Removed ~110 lines of inline `generate_comparison_report()` function
  - Replaced with call to `ComparisonReportGenerator` class from `sitv.reporting`
  - Improved code organization and reusability
  - Maintained identical functionality with cleaner architecture

### Technical Details

- **Composition Analysis**: Predicts optimal 2D composition from 1D task properties using multiple strategies
- **Interaction Detection**: Quantifies task independence with R² score and interaction RMS metrics
- **Automation**: Analysis runs automatically after 2D sweeps when enabled via configuration
- **Modularity**: Comparison report generation moved from scripts to core reporting module
- **Breaking Changes**: None - composition analysis is opt-in via configuration flag

## [0.11.0] - 2025-11-01

### Added

- **3D Task Vector Composition** ([sitv/experiments/composition_3d.py](sitv/experiments/composition_3d.py)):
  - Implemented complete 3D composition experiment exploring `L(M_base + α·T1 + β·T2 + γ·T3)`
  - Added `Composition3DExperiment` class for three-task interaction analysis
  - Supports configurable grid sampling (5³ to 20³ evaluations)
  - Uses in-place parameter modification for memory efficiency
  - Integrated batched evaluation and mixed precision support

- **3D Composition Data Models** ([sitv/data/models.py](sitv/data/models.py)):
  - Added `ThreeDSweepResult` dataclass for 3D composition results
  - Added 3D composition metrics to `ExperimentMetrics`:
    - `enable_3d_composition`, `task_name_3d_1/2/3`
    - `task_vector_3d_1/2/3_magnitude`
  - Tracks full three-task configuration and vector magnitudes

- **3D Composition Configuration** ([sitv/experiments/config.py](sitv/experiments/config.py), [config.yaml](config.yaml)):
  - Added `Composition3DConfig` dataclass with alpha/beta/gamma ranges
  - Added `enable_3d_composition` flag to experiment configuration
  - Added comprehensive configuration section to config.yaml with:
    - Three task selection (task_1, task_2, task_3)
    - Independent scaling ranges for each dimension
    - Configurable grid resolution (default: 10×10×10 = 1,000 evaluations)
    - Timing estimates for quick/standard/high-res configurations

- **3D Visualization** ([sitv/visualization/plotter.py](sitv/visualization/plotter.py)):
  - Added `plot_3d_composition()` method generating interactive 3D plots and 2D slices
  - Interactive 3D scatter plot (HTML via plotly) with rotation and zoom
  - 2D cross-sectional slices showing loss at different γ values (6-panel PNG)
  - Automatic optimal point detection and marking
  - Base model reference point visualization

- **Multi-Task Comparison Plotting** ([sitv/visualization/plotter.py](sitv/visualization/plotter.py)):
  - Added `plot_multi_task_comparison()` with 5 layout modes:
    - `overlaid`: 2×2 grid with all tasks on same plots
    - `side_by_side`: 3×2 grid, one row per task
    - `grid`: 3×3 grid with loss, return, and squaring test
    - `publication`: Single panel with insets for publication quality
    - `heatmap`: Averaged loss landscape revealing universal structure
  - Added comprehensive statistics tables and comparative metrics
  - Color-coded task differentiation with consistent styling
  - Support for 800+ lines of advanced visualization code

- **3D Composition Orchestration** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added `_run_3d_composition()` method for three-task workflow
  - Added `_fine_tune_and_compute_task_vector()` helper for task vector generation
  - Automatic fine-tuning on second and third tasks
  - Reuses main task vector when matching 3D task_1
  - Generates 3D visualizations and saves results to JSON
  - Integrated into main experiment workflow

- **3D Composition Reporting** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added `_create_3d_composition_section()` method for comprehensive 3D analysis
  - Reports optimal composition point (α*, β*, γ*)
  - Reports worst-case composition and closest return to base
  - Includes loss landscape statistics (mean, std, range)
  - Provides interpretation guidance for three-way task interactions
  - Documents visualization file locations

- **Composition Analysis Module** ([sitv/analysis/composition_analyzer.py](sitv/analysis/composition_analyzer.py)):
  - Added `CompositionAnalyzer` class for analyzing multi-task compositions
  - Exported through `sitv.analysis` module
  - Provides specialized analysis for 2D and 3D composition results

### Changed

- **Base Experiment Enhancement** ([sitv/experiments/base.py](sitv/experiments/base.py)):
  - Added `apply_3d_composition()` method for three-task vector composition
  - Validates all three scaling factors (alpha, beta, gamma) for finiteness
  - Uses optimized floating-point accumulation order for numerical stability
  - Supports pre-loaded task vectors to device for performance

- **Experiment Orchestrator Workflow** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Enhanced configuration display to show 3D composition settings
  - Added Phase 5 for 3D composition experiments (after 2D composition)
  - Stores 3D results in `self.results_3d` for report generation
  - Passes 3D results to markdown report generator

- **Markdown Report Generation** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Updated `generate()` and `_build_report()` to accept `results_3d` parameter
  - Enhanced report structure to include 3D composition analysis when available
  - Maintains backward compatibility with 1D and 2D only experiments

### Technical Details

- **3D Composition**: Full implementation of three-task interaction analysis with volumetric loss landscapes
- **Computational Cost**: Scales cubically - 10³ = 1,000 evaluations typical, 20³ = 8,000 for high-resolution
- **Memory Efficiency**: In-place parameter modification handles 3D grids without memory explosion
- **Visualization**: Interactive plotly 3D plots enable rotation and exploration of loss landscape structure
- **Multi-Task Analysis**: Advanced comparison plotting reveals universal patterns across different tasks
- **Line Count**: Added 800+ lines of visualization code and 200+ lines for 3D experiment infrastructure
- **Breaking Changes**: None - 3D composition is opt-in via `composition_3d.enable` configuration

## [0.10.2] - 2025-10-31

### Fixed

- **Riemannian Geometry Device Management** ([sitv/geometry/metric.py](sitv/geometry/metric.py)):
  - Fixed device mismatch error when computing Riemannian norm with task vectors on CPU and Fisher metrics on CUDA
  - Added automatic device detection and tensor movement to ensure same-device operations
  - Handles all approximation types (diagonal, KFAC, full)

- **GPU Memory Management** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Fixed CUDA out of memory errors during alpha sweep experiments
  - Move Fisher metric to CPU after computing Riemannian norm to free ~77GB GPU memory
  - Clear CUDA cache after Fisher computation to release fragmented memory
  - Fisher automatically moved back to GPU during geodesic operations

- **Geodesic Integration Device Consistency** ([sitv/geometry/geodesic.py](sitv/geometry/geodesic.py)):
  - Fixed device mismatch errors in exponential_map when inputs are on different devices
  - Automatically move base_point, tangent_vector, and christoffel to integrator's device
  - Enables proper operation with CPU-stored Fisher/Christoffel symbols
  - Maintains performance by running integration on GPU when available

- **Empty Results Handling** ([sitv/analysis/analyzer.py](sitv/analysis/analyzer.py)):
  - Fixed IndexError when experiment aborts before completing any evaluations
  - Added check for empty results list with graceful degradation
  - Returns None values for all metrics when no results available
  - Displays clear warning message instead of crashing

- **Device Type Check** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Fixed AttributeError when checking device type for CUDA cache clearing
  - Changed `self.device.type` to `self.device` (stored as string, not torch.device object)

- **Type Safety** ([sitv/geometry/metric.py](sitv/geometry/metric.py), [sitv/analysis/gradient/critical_points.py](sitv/analysis/gradient/critical_points.py)):
  - Fixed mypy type checking errors
  - Changed `_compute_full_fisher` return type to allow metadata dict
  - Added explicit type annotation for `indices_set: set[int]` in zero-crossing detection

### Technical Details

- **Device Management**: Complete fix for multi-device scenarios (CPU task vectors, CUDA Fisher metrics)
- **Memory Efficiency**: Fisher metric moved to CPU when not in use, freeing majority of GPU memory
- **Robustness**: Graceful handling of experiment failures and aborts
- **Type Safety**: All mypy errors resolved without changing functionality
- **GPU Memory**: Prevents OOM on 80GB GPUs when model + Fisher consume entire memory

## [0.10.1] - 2025-10-31

### Added

- **2D Composition Task Tracking** ([sitv/data/models.py](sitv/data/models.py)):
  - Added `task_name_2` field to `ExperimentMetrics` dataclass
  - Tracks the name of the second task in 2D composition experiments
  - Previously hardcoded as "sentiment_negative", now dynamically captured from config

### Changed

- **Markdown Reporter Refactoring** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Extracted all magic numbers to named constants at module level
  - Added comprehensive configuration constants section with documentation
  - Created `TRAINING_STEP_INTERVAL`, `MAX_ZERO_CROSSINGS_DISPLAY`, `MAX_SAMPLE_DATA_POINTS`, and 10+ other constants
  - Added `CATEGORY_INTERPRETATIONS` dictionary for extensible category descriptions
  - Improved threshold constants for analysis: `AXIS_PROXIMITY_THRESHOLD`, `NORM_RATIO_SIGNIFICANT_HIGH/LOW`, `CURVATURE_POSITIVE/NEGATIVE_THRESHOLD`
  - Modernized type hints to Python 3.10+ syntax (`list[T]` instead of `List[T]`, `dict` instead of `Dict`, `| None` instead of `Optional`)
  - Fixed 2D composition report to use `task_name_2` from metrics instead of hardcoded string

- **Console Utilities Enhancement** ([sitv/utils/console.py](sitv/utils/console.py), [sitv/utils/__init__.py](sitv/utils/__init__.py)):
  - Added configuration constants: `BANNER_WIDTH`, `BANNER_CHAR`, `SECTION_CHAR`, `SUBSECTION_CHAR`, `PROGRESS_BAR_WIDTH`
  - Added `print_separator()` function for simple separator lines
  - Updated `print_banner()`, `print_section()`, and `print_progress()` to use default constants
  - Made width and character parameters optional with sensible defaults
  - Modernized type hints to Python 3.10+ syntax
  - Exported new constants and functions through `sitv.utils` module

- **Orchestrator Cleanup** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Replaced hardcoded banner/separator printing with centralized utility functions
  - Updated to use `print_banner()` and `print_separator()` from `sitv.utils`
  - Added `task_name_2` to metrics when running 2D composition experiments
  - Improved code consistency and reduced duplication across experiment phases

### Technical Details

- **Code Quality**: Eliminated magic numbers throughout reporting system, improved maintainability
- **Type Safety**: Modernized type hints to leverage Python 3.10+ features for cleaner code
- **Consistency**: Centralized console formatting ensures uniform output styling across all experiment phases
- **Extensibility**: Configuration constants make thresholds and display limits easily adjustable
- **2D Composition**: Task names now properly tracked in both metrics and reports for full experiment reproducibility

## [0.10.0] - 2025-10-31

### Added

- **Riemannian Geometry Module** ([sitv/geometry/](sitv/geometry/)):
  - Implemented complete Riemannian geometry framework for task vectors
  - Added `FisherMetricService` for computing Fisher Information Matrix (diagonal, KFAC, full)
  - Added `GeodesicIntegrator` using Runge-Kutta 4 for exponential map integration
  - Added `GeodesicTaskVectorService` for geodesic-based task vector operations
  - Added `GeometryConfig` with comprehensive configuration for metrics and integration
  - Supports multiple metric types: euclidean, fisher_diagonal, fisher_kfac, fisher_full
  - Includes Christoffel symbol computation for parallel transport

- **Geodesic Interpolation** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py), [sitv/experiments/base.py](sitv/experiments/base.py)):
  - Implemented geodesic task vector application via exponential map
  - Added `apply_geodesic_task_vector()` method using Riemannian geometry
  - Replaces Euclidean straight-line interpolation M(α) = M_base + α·T with proper geodesic exp_M(α·T)
  - Integrated geodesic support into alpha sweep experiment workflow
  - Added geometry service and Fisher metric support to experiment initialization

- **Enhanced Numerical Stability** (11 files):
  - Added perplexity overflow protection in `EvaluationService.compute_perplexity()` (prevents exp(88+) overflow)
  - Added finite value validation for alpha in task vector operations
  - Improved task vector magnitude computation using float64 accumulation
  - Enhanced zero-crossing detection with automatic deduplication
  - Improved curvature computation with proper finite difference for non-uniform grids
  - Added numerical stability to Bayesian sampler Expected Improvement calculation
  - Enhanced 2D composition with better floating-point accumulation order

- **Riemannian Metrics Tracking** ([sitv/data/models.py](sitv/data/models.py)):
  - Added `euclidean_distance`, `geodesic_distance`, `geometry_overhead_seconds` to `AlphaSweepResult`
  - Added geometry configuration fields to `ExperimentMetrics`:
    - `geometry_enabled`, `metric_type`, `fisher_computation_time`
    - `fisher_num_samples`, `fisher_condition_number`
    - `task_vector_magnitude_euclidean`, `task_vector_magnitude_riemannian`
    - `geodesic_integration_enabled`, `geodesic_num_steps`

- **Geometry Reporting** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added Riemannian geometry analysis section to markdown reports
  - Added geodesic vs Euclidean path comparison tables
  - Added Fisher metric configuration and computation time tracking
  - Added task vector norm ratio analysis (||T||_g / ||T||)
  - Included interpretation guidance for positive/negative curvature regions

### Changed

- **Configuration System** ([sitv/experiments/config.py](sitv/experiments/config.py), [config.yaml](config.yaml)):
  - Added `geometry` property to `ExperimentConfig` (lazy-loaded to avoid circular imports)
  - Added comprehensive geometry configuration section to config.yaml:
    - Metric type selection (euclidean, fisher_diagonal, fisher_kfac, fisher_full)
    - Fisher approximation parameters (sampling strategy, num_samples, eigenvalue floor)
    - Geodesic integration settings (RK4 steps, tolerance, step size control)
    - Placeholder sections for symmetry and curvature analysis (Phase 3/4)
  - Changed default task from sentiment_positive to sentiment_negative
  - Increased evaluation batch size from 8 to 32
  - Changed sampling strategy from adaptive back to uniform
  - Adjusted zero-crossing threshold from 0.15 to 0.20
  - Removed model options (Qwen) to focus on Gemma

- **Orchestrator Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added Riemannian geometry computation in task vector phase
  - Integrated Fisher metric computation and Riemannian norm calculation
  - Added geometry configuration display in experiment header
  - Enhanced task vector output to show both Euclidean and Riemannian magnitudes
  - Passes geometry service and Fisher metric to alpha sweep experiments
  - Added geometry metrics to experiment finalization

- **Gradient Analysis** ([sitv/analysis/gradient/critical_points.py](sitv/analysis/gradient/critical_points.py)):
  - Improved `_find_zero_crossing_indices()` with automatic deduplication
  - Uses set-based approach to efficiently handle overlapping sign changes and near-zero points
  - Enhanced documentation explaining detection criteria

- **Evaluation Service** ([sitv/core/evaluation.py](sitv/core/evaluation.py)):
  - Added overflow protection in `compute_perplexity()` method
  - Returns `float('inf')` for losses > 88.0 to prevent float32 overflow
  - Added input validation for non-finite loss values
  - Enhanced documentation with safety margin explanation

- **Task Vector Service** ([sitv/core/task_vector.py](sitv/core/task_vector.py)):
  - Enhanced `compute_magnitude()` with float64 precision for numerical stability
  - Added finite value validation in `apply()` method
  - Improved docstrings with validation and error handling details

- **Base Experiment** ([sitv/experiments/base.py](sitv/experiments/base.py)):
  - Enhanced `apply_2d_composition()` with better numerical stability
  - Added finite value validation for alpha and beta
  - Improved floating-point accumulation order for precision
  - Added comprehensive error messages for invalid scaling factors

- **Adaptive Sampler** ([sitv/experiments/sampling/adaptive_sampler.py](sitv/experiments/sampling/adaptive_sampler.py)):
  - Fixed `_find_high_curvature_regions()` with proper finite difference formula
  - Uses correct discrete derivative for non-uniform grids
  - Added epsilon protection against division by very small intervals
  - Enhanced numerical stability in second derivative computation

- **Bayesian Sampler** ([sitv/experiments/sampling/bayesian_sampler.py](sitv/experiments/sampling/bayesian_sampler.py)):
  - Improved `_expected_improvement()` numerical stability
  - Added epsilon to sigma to prevent division by zero
  - Enhanced robustness of Gaussian Process acquisition function

### Removed

- **Reorganized Findings** (5 files deleted):
  - Removed `findings/2025-10-31/ANALYSIS.md` (355 lines)
  - Removed `findings/2025-10-31/IF.md` (284 lines)
  - Removed `findings/2025-10-31/QA.md` (289 lines)
  - Removed `findings/2025-10-31/SN.md` (289 lines)
  - Removed `findings/2025-10-31/SP.md` (344 lines)
  - New organized structure: `findings/2025-10-31/QA/`, `findings/2025-10-31/SN/`, `findings/2025-10-31/SP/`

### Fixed

- Perplexity computation overflow for large loss values (> 88.0)
- Zero-crossing detection duplicate indices from overlapping criteria
- Curvature computation errors on non-uniform alpha grids
- Bayesian sampler division by zero in Expected Improvement
- Numerical precision loss in task vector magnitude computation
- Invalid alpha/beta handling in task vector operations

### Technical Details

- **Riemannian Geometry**: Full implementation using Fisher Information Matrix as metric tensor
- **Geodesic Integration**: RK4 solver with configurable step count (default: 100 steps)
- **Numerical Stability**: Float64 accumulation, overflow protection, epsilon guards throughout
- **Fisher Metric**: Diagonal, KFAC, and full matrix approximations supported
- **Performance**: Fisher computation cached for reuse, optional parallel transport
- **Experimental**: Symmetry and curvature analysis placeholder (Phase 3/4 not yet implemented)
- **Breaking Changes**: None - Riemannian features are opt-in via geometry.enabled configuration

## [0.9.0] - 2025-10-31

### Added

- **Batched Evaluation** ([sitv/core/evaluation.py](sitv/core/evaluation.py)):
  - Added `batch_size` parameter to EvaluationService (default: 8)
  - Implemented batched text processing in `evaluate()` method
  - Processes multiple texts per forward pass to reduce overhead
  - Configurable via `evaluation.batch_size` in config.yaml

- **Mixed Precision Support** ([sitv/core/evaluation.py](sitv/core/evaluation.py)):
  - Added `enable_mixed_precision` parameter (default: true)
  - Automatically selects BF16 on Ampere+ GPUs, FP16 on older CUDA/MPS
  - Uses `torch.autocast` for 1.5-2x evaluation speedup
  - Configurable via `evaluation.enable_mixed_precision` in config.yaml

- **Evaluation Configuration** ([sitv/experiments/config.py](sitv/experiments/config.py)):
  - Added `batch_size`, `enable_mixed_precision`, and `max_length` to EvaluationConfig
  - Defaults: batch_size=8, enable_mixed_precision=true, max_length=512
  - Configuration loaded from `evaluation` section in config.yaml

### Changed

- **EvaluationService Refactor** ([sitv/core/evaluation.py](sitv/core/evaluation.py)):
  - Updated type hints to modern Python 3.12+ syntax (list[str] instead of List[str], dict instead of Dict)
  - Modified loss accumulation to properly average across batches
  - Added `strict=True` to zip() calls for safer iteration
  - Enhanced docstrings with performance optimization details

- **Experiment Integration** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py), [sitv/experiments/composition_2d.py](sitv/experiments/composition_2d.py)):
  - Added evaluation performance parameters to both experiment classes
  - Passes `eval_batch_size`, `eval_enable_mixed_precision`, and `eval_max_length` to EvaluationService
  - Enables performance tuning through configuration

- **Orchestrator Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added "Evaluation Performance" section to configuration display
  - Shows batch size, mixed precision, and max length settings at startup
  - Passes evaluation config to AlphaSweepExperiment and Composition2DExperiment

- **Configuration Updates** ([config.yaml](config.yaml)):
  - Changed task from "qa_factual" to "sentiment_positive"
  - Increased alpha sweep samples from 75 to 100
  - Changed sampling strategy from "uniform" to "adaptive"
  - Enabled 2D composition experiment (enable: true)
  - Added model documentation with supported models (Gemma, Qwen)
  - Added NEW evaluation performance section with batch_size, enable_mixed_precision, max_length

### Technical Details

- **Performance**: Batched evaluation reduces forward pass overhead, mixed precision provides 1.5-2x speedup
- **Hardware Optimization**: Auto-detection selects optimal dtype (BF16 on Ampere+, FP16 on Pascal/MPS)
- **Type Safety**: Modernized type hints throughout evaluation module for Python 3.12+
- **Backward Compatibility**: Default settings provide optimal performance; can be disabled via config

## [0.8.1] - 2025-10-31

### Added

- **Experimental Findings** ([findings/2025-10-31/](findings/2025-10-31/)):
  - Added comprehensive experimental results for four task types with gemma-3-4b-it model
  - [SP.md](findings/2025-10-31/SP.md): Sentiment Positive task (343 lines) - fine-tuning with 4 epochs, learning rate 1e-4
  - [SN.md](findings/2025-10-31/SN.md): Sentiment Negative task (289 lines) - fine-tuning with 4 epochs, learning rate 1e-4
  - [IF.md](findings/2025-10-31/IF.md): Instruction Following task (284 lines) - fine-tuning with 4 epochs, learning rate 1e-4
  - [QA.md](findings/2025-10-31/QA.md): Question Answering Factual task (288 lines) - fine-tuning with 4 epochs, learning rate 1e-4
  - Each file contains full experimental logs including training progress, loss curves, alpha sweep results, and analysis
  - Total ~1200 lines of experimental documentation for reproducibility and analysis

## [0.8.0] - 2025-10-31

### Added

- **Reproducibility Support** ([config.yaml](config.yaml), [sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Added `reproducibility.seed` configuration parameter for deterministic experiments
  - Implemented `set_random_seed()` function to set seeds across Python, NumPy, and PyTorch
  - Added seed parameter to `ExperimentConfig`, `ExperimentOrchestrator`, and `FineTuner`
  - Enabled PyTorch deterministic mode (cudnn.deterministic=True, cudnn.benchmark=False)
  - Default seed value: 42 (configurable via config.yaml)

- **Sentiment Preference Evaluation** ([sitv/core/evaluation.py](sitv/core/evaluation.py), [sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Added `evaluate_sentiment_preference()` method to EvaluationService
  - Computes preference score as `negative_loss - positive_loss` (positive score indicates preference for positive sentiment)
  - Added `sentiment_preference` and `task_eval_loss_negative` fields to AlphaSweepResult
  - Orchestrator now automatically loads opposite sentiment texts for preference calculation
  - Integrated into alpha sweep experiment workflow for sentiment tasks

### Changed

- **Field Naming Clarity** (16 files):
  - Renamed `task_performance` → `task_eval_loss` throughout codebase for better clarity
  - Updated field name in `AlphaSweepResult` dataclass ([sitv/data/models.py](sitv/data/models.py))
  - Updated all references in analyzer, plotter, markdown reporter, and orchestrator
  - Updated all test files to use new field name
  - **Breaking change**: JSON files from previous versions will have `task_performance` instead of `task_eval_loss`

- **Enhanced Documentation** ([sitv/data/models.py](sitv/data/models.py)):
  - Improved docstring for AlphaSweepResult to clarify evaluation metrics
  - Documented that `loss` is evaluated on general dataset (broad language modeling)
  - Documented that `task_eval_loss` is evaluated on task-specific evaluation data
  - Added documentation for new sentiment preference fields

### Technical Details

- **Reproducibility**: Full deterministic behavior when seed is set (may impact performance slightly)
- **Sentiment Analysis**: Preference calculation enables quantitative measurement of directional task vector effects
- **Breaking Change**: Field rename affects saved JSON results and requires code updates for external analysis scripts
- **Backward Compatibility**: Set `reproducibility.seed: null` in config.yaml to disable deterministic mode

## [0.7.3] - 2025-10-31

### Added

- **MIT License** ([LICENSE](LICENSE)):
  - Added MIT License to the project (Copyright 2025, Casey Hand)
  - Ensures open-source licensing for distribution and modification

- **Test Suite Expansion** (5 new test files, 1,581 lines):
  - Added [tests/test_analyzer.py](tests/test_analyzer.py) for ResultAnalyzer testing (343 lines)
    - Tests for zero-crossing detection and min loss finding
    - Tests for squaring return point analysis
    - Edge case handling (empty results, monotonic loss, negative alphas)
  - Added [tests/test_config.py](tests/test_config.py) for configuration system testing (283 lines)
    - Tests for YAML config loading and reloading
    - Tests for ExperimentConfig, AlphaSweepConfig, FineTuningConfig classes
    - Tests for SamplingConfig and GradientAnalysisConfig
  - Added [tests/test_data_loader.py](tests/test_data_loader.py) for DatasetLoader testing (299 lines)
    - Tests for loading general, task, and eval datasets
    - Tests for comment handling, UTF-8 encoding, and error cases
    - Integration tests with real data directory
  - Added [tests/test_file_manager.py](tests/test_file_manager.py) for FileManager testing (288 lines)
    - Tests for JSON saving/loading (results, metrics, 2D results)
    - Tests for file existence checking and path resolution
    - Round-trip save/load verification
  - Added [tests/test_sampling_strategies.py](tests/test_sampling_strategies.py) for sampling strategies testing (368 lines)
    - Tests for UniformSampler, AdaptiveSampler, and BaseSampler
    - Tests for coarse sampling, refinement passes, and region detection
    - Integration tests comparing uniform vs adaptive strategies

- **Sampling Configuration Support** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Added `sampling_config` parameter to `AlphaSweepExperiment.__init__()`
  - Enhanced sampler initialization to pass configuration parameters
  - Adaptive sampler now receives `coarse_samples`, `refine_factor`, `curvature_threshold`
  - Bayesian sampler now receives `n_initial` and `acquisition` parameters

### Changed

- **Configuration Enhancement** ([config.yaml](config.yaml)):
  - Increased fine-tuning epochs from 6 to 8 for better convergence
  - Made adaptive sampling parameters explicit:
    - `adaptive_coarse_samples`: 60 (previously implicit default)
    - `adaptive_refine_factor`: 3 (previously implicit default)
    - `adaptive_curvature_threshold`: 0.5 (previously implicit default)
  - Increased 2D composition resolution from 20×20 to 30×30 (400 → 900 evaluations)
  - Improved configuration comments and organization

- **Test Suite Migration** (6 test files updated):
  - Updated [tests/conftest.py](tests/conftest.py):
    - Migrated imports from `main.py` to modular `sitv` packages
    - Updated `mock_task_definition` to match new TaskDefinition structure
    - Updated `mock_alpha_sweep_result` to return list of result objects instead of single object with lists
  - Updated [tests/test_data_models.py](tests/test_data_models.py): Imports from `sitv.data.models`
  - Updated [tests/test_device.py](tests/test_device.py): Imports from `sitv.core.device`
  - Updated [tests/test_model_management.py](tests/test_model_management.py): Imports from `sitv.models.loader`, uses `ModelService.check_saved_models_exist()`
  - Updated [tests/test_task_vector.py](tests/test_task_vector.py): Imports from `sitv.core.task_vector`, uses `TaskVectorService.compute()`

- **Documentation Updates** ([README.md](README.md)):
  - Updated version badge from 0.7.1 to 0.7.2
  - Updated package line count from 3,817 to 7,899 lines
  - Simplified test suite description (removed specific test count)
  - Removed references to ARCHITECTURE.md and REFACTORING_SUMMARY.md

- **Orchestrator Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Now passes `sampling_config` to `AlphaSweepExperiment` initialization
  - Enables full configuration of adaptive and Bayesian sampling strategies

### Fixed

- **macOS scipy Compatibility** ([pyproject.toml](pyproject.toml)):
  - Added pytest environment configuration for DYLD_LIBRARY_PATH
  - Fixes scipy linking issues on macOS with conda installations
  - Enables seamless test execution on macOS without manual environment setup

### Technical Details

- **Test Coverage**: Added 1,581 lines of comprehensive test coverage across 5 new test modules
- **Test Suite Modernization**: All tests now import from modular `sitv` package instead of monolithic `main.py`
- **Configuration Flexibility**: Sampling strategies now fully configurable via YAML
- **Platform Support**: Improved macOS compatibility for scientific computing libraries
- **Line Count Growth**: 3,817 → 7,899 lines (107% increase, primarily from test expansion)

## [0.7.2] - 2025-10-31

### Changed

- **Type Safety Improvements** (16 files):
  - Enhanced type annotations throughout codebase for better static type checking
  - Added explicit `Optional` type hints for nullable parameters and variables
  - Converted numpy/torch scalar returns to explicit Python `float()` types
  - Added type annotations for list and variable initializations
  - Added `type: ignore` comments for untyped third-party imports (scipy, sklearn, yaml)
  - Improved mypy compliance across analysis, core, experiments, and reporting modules

- **Configuration Path Fix** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Fixed sampling strategy configuration path from `sampling_strategy` to `sampling.strategy`
  - Updated to match nested configuration structure in config.yaml

### Technical Details

- **Type Coverage**: Improved type safety in critical modules including gradient analysis, task vectors, experiments, and samplers
- **Static Analysis**: Enhanced mypy compliance with specific type: ignore comments and proper type annotations
- **Code Quality**: Better IDE support and type checking for development workflow
- **Backward Compatibility**: No functional changes, purely type safety improvements

## [0.7.1] - 2025-10-31

### Added

- **Sampling Strategy Integration** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Integrated sampling strategies into alpha sweep experiment workflow
  - Added `sampling_strategy` parameter to `AlphaSweepExperiment.__init__()`
  - Implemented `_create_sampler()` factory method for instantiating samplers
  - Modified `run()` to use sampler-generated alpha values instead of uniform `np.linspace()`
  - Updated progress calculation to handle variable sample counts from adaptive/bayesian strategies
  - Added sampling strategy imports: `UniformSampler`, `AdaptiveSampler`, `BayesianSampler`

- **Sampling Strategy Tracking** ([sitv/data/models.py](sitv/data/models.py)):
  - Added `sampling_strategy` field to `ExperimentMetrics` dataclass
  - Defaults to "uniform" for backward compatibility
  - Tracks which sampling strategy was used in experiment results

### Changed

- **Orchestrator Enhancement** ([sitv/experiments/orchestrator.py](sitv/experiments/orchestrator.py)):
  - Wired `sampling_strategy` configuration from config to `AlphaSweepExperiment`
  - Added sampling strategy to metrics collection for result tracking

- **Report Generation** ([sitv/reporting/markdown.py](sitv/reporting/markdown.py)):
  - Added "Sampling Strategy" line to Alpha Sweep section
  - Displays capitalized strategy name (Uniform/Adaptive/Bayesian) in reports

- **Alpha Sweep Flexibility** ([sitv/experiments/alpha_sweep.py](sitv/experiments/alpha_sweep.py)):
  - Updated `_calculate_eta()` to accept `total_samples` parameter for flexibility
  - Updated `_print_summary()` to accept `total_samples` parameter
  - Changed internal variables from `self.num_samples` to dynamic `total_alphas` count

### Technical Details

- **Feature Completion**: Completes the sampling strategy infrastructure added in v0.6.0
- **Backward Compatibility**: Default "uniform" strategy maintains existing behavior
- **Configuration**: Sampling strategies now fully functional via `config.yaml` setting
- **Flexibility**: Adaptive and Bayesian strategies now usable for 40-90% speedup as originally designed

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
