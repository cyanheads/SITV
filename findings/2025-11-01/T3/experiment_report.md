# SITV Experiment Report

**Generated**: 2025-11-01 10:58:55
**Model**: google/gemma-3-4b-it
**Task**: sentiment_positive
**General Evaluation Dataset**: combined
**Device**: cuda

## Executive Summary

- **Best General α**: +1.1364 (Loss: 38.1775)
- **Best Task α**: +1.4596 (Loss: 37.2398)
- **Zero-Crossings**: 22 found
- **Total Duration**: 23.1 minutes

## Configuration

### Model
- **Model Name**: google/gemma-3-4b-it
- **Total Parameters**: 4,300,079,472
- **Device**: cuda

### Task & Evaluation
- **Training Task**: sentiment_positive
- **General Evaluation Dataset**: combined
  - *This dataset measures how the task vector affects general language modeling capability*

### Training
- **Training Examples**: 3000
- **Epochs**: 4
- **Learning Rate**: 1.00e-04
- **Training Steps**: 376
- **Final Training Loss**: 0.2341

### Task Vector
- **Euclidean Magnitude (||T||)**: 36.2709
- **Riemannian Magnitude (||T||_g)**: 0.1714
- **Norm Ratio (||T||_g / ||T||)**: 0.0047
- **Computation Time**: 31.39s

### Alpha Sweep
- **Alpha Range**: [-0.5, 1.5]
- **Samples**: 100
- **Sampling Strategy**: Uniform
- **Avg Time per Sample**: 0.68s

## Riemannian Geometry Analysis

### Metric Configuration
- **Metric Type**: fisher_diagonal
- **Fisher Computation Time**: 15.21s (21.9% of sweep time)
- **Fisher Samples**: 1,000

### Task Vector Norms
- **Euclidean Norm ||T||**: 36.2709
- **Riemannian Norm ||T||_g**: 0.1714
- **Ratio ||T||_g / ||T||**: 0.0047
  - *The Fisher metric shrinks the task vector (information-poor directions).*

### Geodesic Integration

#### Configuration
- **Enabled**: Yes
- **RK4 Steps per Evaluation**: 20
- **Integration Tolerance**: 1e-6 (from config.yaml)
- **Step Size Control**: Disabled (fixed step size for performance)
- **Overhead**: ~20x integration cost vs straight-line

#### Implementation Details

**Algorithm**: 4th-order Runge-Kutta (RK4) integration of the geodesic equation

The geodesic equation in Riemannian geometry is:

```
d²x^i/dt² + Γ^i_jk (dx^j/dt)(dx^k/dt) = 0
```

where Γ^i_jk are the Christoffel symbols computed from the Fisher metric.

**Integration Process**:
1. Start at base model parameters θ_base
2. Initial velocity v = T (task vector direction)
3. For each RK4 step:
   - Compute position: θ(t+dt) using RK4
   - Compute acceleration from Christoffel symbols Γ
   - Update velocity accounting for manifold curvature
4. Result: geodesic path exp_θ_base(α·v) instead of θ_base + α·v

**Convergence Criteria**:
- Tolerance: 1e-6 (maximum integration error per step)
- Max iterations: 1000 (safety limit)
- Early stopping if error < tolerance

**Code Implementation**: [`sitv/geometry/geodesic.py`](sitv/geometry/geodesic.py)

Key functions:
- `integrate_geodesic()`: Main RK4 integration loop
- `geodesic_rk4_step()`: Single RK4 step with Christoffel symbols
- `compute_christoffel_symbols()`: Finite difference computation of Γ^i_jk

#### Varying-Metric Geodesics (Curvature Detection)

**Metric Recomputation**:
- **Interval**: Every 5 RK4 steps
- **Total Recomputations**: 4 per α value
- **Finite Difference ε**: 1e-3 (for Christoffel symbol computation)

This approach recomputes the Fisher metric along the geodesic path, allowing detection
of true geometric curvature via Christoffel symbols. If Γ^i_jk ≠ 0, the space is curved!

**Computational Cost**:
- Each metric recomputation: ~2-3s (Fisher matrix + eigendecomposition)
- Per α evaluation: 10.0s (estimated)
- Total for 100 alphas: ~16.7 minutes

#### Configuration in config.yaml

All geodesic integration parameters are set in [`config.yaml`](config.yaml):

```yaml
geometry:
  geodesic_integration:
    enabled: true
    num_steps: 20                    # RK4 integration steps
    tolerance: 1.0e-6                # Integration error tolerance
    step_size_control: false         # Fixed vs adaptive step size
    max_iterations: 1000             # Safety limit
    recompute_metric_every: 5        # Recompute Fisher every N steps (0=never)
    metric_epsilon: 1.0e-3           # Finite difference ε for Christoffel
```

**To modify**: Edit these values in `config.yaml` and re-run the experiment.

#### Theoretical Background

Geodesic integration replaces Euclidean interpolation with proper Riemannian geometry:

- **Euclidean**: M(α) = M_base + α·T (straight line in parameter space)
- **Geodesic**: M(α) = exp_M_base(α·T) (shortest path on curved manifold)

The geodesic path follows the intrinsic geometry of the parameter manifold as
defined by the Fisher Information Matrix. This is the natural generalization of
"straight lines" to curved spaces.

**Key Insight**: If geodesic paths deviate significantly from Euclidean paths,
this indicates the parameter space has non-trivial curvature - the "anthill"
hypothesis rather than flat Euclidean space!

### 🔍 Curvature Detection (Varying-Metric Geodesics)

**Configuration**:
- **Metric Recomputation Interval**: Every 5 RK4 steps
- **Total Recomputations per α**: 4

**Christoffel Symbol Analysis**:
- **RMS Magnitude**: 0.000000
- **Curvature Detected**: ❌ No - Space appears flat

**Interpretation**:
The Christoffel symbols Γ are approximately zero (RMS = 0.000000).
This indicates the Fisher metric is approximately constant, meaning:
- Parameter space appears flat (Euclidean approximation valid)
- Geodesic paths are nearly identical to straight lines
- Riemannian geometry provides minimal benefit over Euclidean operations

This validates the simpler Euclidean approach: M(α) = M_base + α·T.

### Interpretation
The Riemannian norm accounts for the Fisher Information Matrix (local curvature of
parameter space). This reflects the "information geometry" of the parameter manifold,
where distances are measured according to the KL divergence between model distributions.

## Curvature Analysis

The parameter manifold's curvature reveals how the loss landscape curves in different
directions. Positive curvature (sphere-like) means geodesics converge, while negative
curvature (hyperbolic/saddle-like) means geodesics diverge.

### Sectional Curvature Distribution

| Metric | Value |
|--------|-------|
| Mean curvature | 0.000000 |
| Standard deviation | 0.000000 |
| Min curvature | 0.000000 |
| Max curvature | 0.000000 |
| Samples analyzed | 10 |

### Interpretation

**Nearly flat (Euclidean-like)**

The manifold is nearly flat in the base model region. This suggests that
Euclidean operations (simple addition of task vectors) provide a good approximation.
Geodesics and straight lines are nearly identical.


## Symmetry Analysis

Parameter space symmetries reveal redundancies in model representations. Detecting
these symmetries allows working in quotient space (parameters modulo symmetries),
providing a more principled geometric analysis.

### Detected Symmetries

#### Rotation Symmetry

**Status**: ✅ **Detected** (score: 1.00/1.00)

- Average loss deviation: 0.000000
- Number of tests: 10
- Parameter subspaces exhibit rotation invariance

#### Permutation Symmetry

**Status**: ✅ **Detected** (score: 1.00/1.00)

- Average loss deviation: 0.000000
- Number of tests: 10
- Neurons within layers can be reordered without affecting loss

#### Scaling Symmetry

**Status**: ❌ Not detected (score: 0.00/1.00)

### Summary

- **Total symmetries detected**: 2
- **Detected types**: rotation, permutation

**Implication**: The model exhibits parameter redundancy. Working in quotient space
(parameters modulo these symmetries) provides a more canonical representation and
addresses the theoretical critique from arXiv:2506.13018.



## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | 1.2m | 5.2% |
| Alpha Sweep | 1.2m | 5.0% |
| **Total** | **23.1m** | **100%** |

## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
| 10 | 0.11 | 1.4800 | 9.76e-05 | 6.25 |
| 30 | 0.32 | 0.2592 | 9.23e-05 | 2.06 |
| 60 | 0.64 | 0.2327 | 8.43e-05 | 1.30 |
| 90 | 0.96 | 0.2069 | 7.63e-05 | 1.28 |
| 120 | 1.28 | 0.2005 | 6.84e-05 | 1.25 |
| 150 | 1.60 | 0.1977 | 6.04e-05 | 0.95 |
| 180 | 1.91 | 0.1927 | 5.24e-05 | 0.85 |
| 210 | 2.23 | 0.1920 | 4.44e-05 | 0.79 |
| 240 | 2.55 | 0.1900 | 3.64e-05 | 0.97 |
| 270 | 2.87 | 0.1885 | 2.85e-05 | 0.94 |
| 300 | 3.19 | 0.1884 | 2.05e-05 | 1.09 |
| 330 | 3.51 | 0.1843 | 1.25e-05 | 0.77 |
| 360 | 3.83 | 0.1846 | 4.52e-06 | 1.15 |
| 376 | 4.00 | 0.0000 | 0.00e+00 | 0.00 |

### Training Summary
- **Initial Loss**: 1.4800
- **Final Loss**: 0.1846
- **Loss Improvement**: +1.2954
- **Mean Gradient Norm**: 1.23 (σ=0.93)
- **Total Training Steps**: 38


## Hyperparameter Analysis & Training Convergence

This section addresses hyperparameter sensitivity and provides recommendations for
different experimental scenarios.

### Hyperparameters Used

| Parameter | Value | Purpose |
|-----------|-------|---------|
| **Learning Rate** | 1.00e-04 | Controls optimization step size |
| **Epochs** | 4 | Number of complete passes through training data |
| **Batch Size** | 7 | Training examples per gradient update |
| **Training Examples** | 3000 | Total size of training dataset |
| **Max Sequence Length** | N/A | Maximum token length for training sequences |
| **Optimizer** | AdamW (default) | Adaptive learning rate with weight decay |

### Training Convergence Analysis

**Status**: ✅ **Converged**

- **Initial Loss**: 1.4800
- **Final Loss**: 0.1846
- **Total Reduction**: 1.2954 (87.5%)
- **Final Loss Stability**: Mean=0.1850, σ=0.0014
- **Mean Gradient Norm**: 1.235 (σ=0.928)

**Interpretation**: Training converged successfully. Loss stabilized and gradient norms
decreased, indicating the model reached a local minimum. The hyperparameters are well-tuned
for this task.

### Hyperparameter Sensitivity & Recommendations

#### Learning Rate Sensitivity
- **Current**: 1.00e-04
- **Too High**: Loss oscillates or diverges, gradients explode (norm > 10)
- **Too Low**: Loss decreases very slowly, requires many more epochs
- **Recommended Range**: 1e-5 to 5e-4 for fine-tuning pre-trained LLMs
- **For This Task**: Current value appears optimal

#### Epoch Sensitivity
- **Current**: 4 epochs
- **Signs of Underfitting**: Loss still decreasing at final epoch (need more)
- **Signs of Overfitting**: Task loss decreases but general loss increases
- **Recommended**: Current setting is good

#### Batch Size Impact
- **Current Setup**: 3000 examples over 376 steps
- **Larger Batch**: Faster training, more memory, more stable gradients
- **Smaller Batch**: Slower training, less memory, noisier gradients (can help escape local minima)
- **Memory Constraint**: Reduce if OOM errors occur

### Recommendations for Different Scenarios

| Scenario | Learning Rate | Epochs | Batch Size | Notes |
|----------|---------------|--------|------------|-------|
| **Quick Test** | 1e-4 | 2 | 16 | Fast iteration, may underfit |
| **Standard Training** | 5e-5 to 1e-4 | 3-5 | 16-32 | Balanced quality/speed |
| **High Quality** | 5e-5 | 6-10 | 32 | Best convergence, slower |
| **Large Models** | 1e-5 to 5e-5 | 4-8 | 8-16 | Lower LR for stability |
| **Small Datasets** | 1e-4 | 10-20 | 8 | More epochs to compensate |

### Configuration File Reference

These hyperparameters are set in [`config.yaml`](config.yaml):

```yaml
fine_tuning:
  num_epochs: 4
  learning_rate: 1.00e-04
  batch_size: [set in config]
  data_repetition_factor: [multiplication factor for dataset size]
```

**To modify**: Edit `config.yaml` and re-run the experiment.


## Results Summary

### Minimum General Loss
- **α**: +1.1364
- **Loss**: 38.1775
- **Δ from base**: -0.8898

### Zero-Crossings Found ★

1. α = -0.3788, |ΔL| = 0.006541
2. α = -0.3384, |ΔL| = 0.107336
3. α = -0.3182, |ΔL| = 0.107846
4. α = -0.2980, |ΔL| = 0.183133
5. α = -0.2778, |ΔL| = 0.114585
6. α = -0.2576, |ΔL| = 0.024059
7. α = -0.2374, |ΔL| = 0.033176
8. α = -0.2172, |ΔL| = 0.187107


## Alpha Sweep Details

**Base Model Loss**: L(M_base) = 39.0673

### Sample Data Points (Key Alphas + Strategic Samples)

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
| -0.500 | 39.4247 | 38.8497 | 0.357396 | 0.217564 | 38.5301 | 132408350158233728.00 | 74509817159153840.00 |
| -0.419 | 39.5502 | 38.9004 | 0.482975 | 0.166822 | 39.0288 | 150125225160552096.00 | 78388206210620816.00 |
| -0.338 | 39.1746 | 39.1303 | 0.107336 | 0.063036 | 38.7200 | 103113537944321024.00 | 98645302345249392.00 |
| -0.258 | 39.0432 | 38.7788 | 0.024059 | 0.288439 | 38.9165 | 90417333121462736.00 | 69411710720605872.00 |
| -0.177 | 39.0905 | 38.7545 | 0.023187 | 0.312806 | 39.0917 | 94791692698843568.00 | 67740812072372192.00 |
| -0.096 | 39.1514 | 38.8933 | 0.084171 | 0.173987 | 39.2796 | 100752430398228592.00 | 77828560575850592.00 |
| -0.035 | 38.9708 | 39.3667 | 0.096423 | 0.299469 | 38.9887 | 84105476905437664.00 | 124956293301319312.00 |
| -0.015 | 38.9733 | 38.8776 | 0.093933 | 0.189646 | 38.9352 | 84315180252025840.00 | 76619329509076800.00 |
| +0.066 | 39.1467 | 39.0282 | 0.079380 | 0.039117 | 38.6733 | 100270802932026064.00 | 89066038964460336.00 |
| +0.146 | 38.5875 | 38.4912 | 0.479778 | 0.576098 | 37.8698 | 57323892080670296.00 | 52060013483004976.00 |
| +0.227 | 38.3355 | 38.5513 | 0.731785 | 0.515979 | 38.0660 | 44554379848661520.00 | 55285810198859680.00 |
| +0.308 | 38.4534 | 38.4610 | 0.613826 | 0.606293 | 38.4689 | 50132451404800344.00 | 50511551338800032.00 |
| +0.389 | 38.5027 | 38.5020 | 0.564553 | 0.565248 | 37.7695 | 52664545434473128.00 | 52627914148132656.00 |
| +0.470 | 38.5757 | 38.3797 | 0.491584 | 0.687533 | 38.2881 | 56651106650933224.00 | 46570240134729928.00 |
| +0.551 | 38.6266 | 38.8516 | 0.440717 | 0.215710 | 38.1033 | 59607279910138320.00 | 74648120199660752.00 |
| +0.631 | 38.6542 | 38.7995 | 0.413085 | 0.267763 | 38.5574 | 61277339448642144.00 | 70861872638940216.00 |
| +0.712 | 38.7195 | 38.9938 | 0.347750 | 0.073496 | 38.6007 | 65414564789799704.00 | 86056046036664384.00 |
| +0.793 | 38.6236 | 38.7350 | 0.443715 | 0.332251 | 38.3763 | 59428854197809680.00 | 66436347324533064.00 |
| +0.874 | 38.6471 | 39.3663 | 0.420138 | 0.298996 | 37.9479 | 60846648079047200.00 | 124897168382251936.00 |
| +0.955 | 38.3227 | 40.1512 | 0.744537 | 1.083969 | 38.3857 | 43989817361782504.00 | 273817995592412608.00 |
| +1.035 | 38.2508 | 40.3773 | 0.816434 | 1.310062 | 37.6164 | 40938074613996456.00 | 343283600190827008.00 |
| +1.116 | 38.2365 | 41.2931 | 0.830754 | 2.225831 | 37.6843 | 40356055092186600.00 | 857761311871485184.00 |
| +1.136 | 38.1775 | 41.7232 | 0.889815 | 2.655908 | 37.3582 | 38041589273569056.00 | 1318701372918012928.00 |
| +1.197 | 38.4591 | 42.0461 | 0.608169 | 2.978861 | 37.3465 | 50416877063345464.00 | 1821391894547077376.00 |
| +1.278 | 38.5036 | 43.5001 | 0.563675 | 4.432858 | 37.3746 | 52710772549306040.00 | 7795900556174947328.00 |
| +1.359 | 39.0209 | 44.2817 | 0.046348 | 5.214393 | 37.9210 | 88424312956916864.00 | 17032667382450382848.00 |
| +1.439 | 38.6615 | 43.9643 | 0.405763 | 4.897011 | 37.3526 | 61727654409974592.00 | 12400667457207627776.00 |
| +1.460 | 38.6177 | 44.4002 | 0.449556 | 5.332888 | 37.2398 | 59082768307932592.00 | 19175391810604564480.00 |

### Key Metrics
- **Total Samples**: 100
- **Samples Displayed**: 28 (includes key α values: -0.5, +0.0, +0.5, +1.0, +1.5)
- **Alpha Range**: [-0.5, 1.5]
- **Alpha Step Size**: 0.0202
- **Total Sweep Time**: 1.2 minutes
- **Avg Time per Sample**: 0.68s


## Statistical Summary

### Loss Distribution Percentiles (Compact View)

High-density summary for LLM pattern recognition:

| Percentile | Loss | α at this Loss | ΔL from base |
|------------|------|----------------|--------------|
| 10th | 38.3426 | +0.7727 | -0.724681 |
| 25th | 38.4970 | +0.2475 | -0.570244 |
| 50th (Median) | 38.6578 | +0.6313 | -0.409424 |
| 75th | 38.9919 | +0.0253 | -0.075369 |
| 90th | 39.2454 | +1.3182 | +0.178139 |

**Base Model Loss**: L(M_base) = 39.0673

### Loss Distribution (General)
- **Mean**: 38.7656
- **Std Dev**: 0.3542
- **Min**: 38.1775 (at α = +1.1364)
- **Max**: 39.7276 (at α = -0.4394)

### Functional Return Distribution
- **Mean**: 0.3966
- **Std Dev**: 0.2432
- **Percentiles**:
  - 10th: 0.083692
  - 25th: 0.166878
  - 50th (Median): 0.417032
  - 75th: 0.582171
  - 90th: 0.724681
- **Min**: 0.006541 (at α = -0.3788)
- **Max**: 0.8898 (at α = +1.1364)

### Task Performance Distribution
- **Mean**: 38.3388
- **Std Dev**: 0.5639
- **Percentiles**:
  - 10th: 37.6125
  - 25th: 37.9412
  - 50th (Median): 38.3330
  - 75th: 38.7524
  - 90th: 39.0004
- **Min**: 37.2398 (at α = +1.4596)
- **Max**: 39.9674 (at α = -0.4596)

## Per-Category Loss Analysis

**Dataset**: combined

This breakdown shows how the task vector (sentiment_positive) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
| coding | +1.0960 | 37.6603 | How well does the model handle programming/technical content? |
| common_knowledge | +1.1364 | 37.8561 | How well does the model handle everyday general knowledge? |
| mixed_domain | +1.0354 | 38.0803 | How well does the model handle diverse multi-domain content? |
| wikitext | +0.6717 | 37.5625 | How well does the model handle factual/encyclopedic content? |

### Baseline Comparison (α ≈ 0.005)

Loss by category before applying task vector:

| Category | Loss |
|----------|------|
| coding | 39.4054 |
| common_knowledge | 38.9914 |
| mixed_domain | 39.1074 |
| wikitext | 38.7526 |

**Insight**: Lower values indicate better performance in that domain.

### Category Loss Evolution Across Key Alpha Values

How does each category's loss change as α varies?

| α | coding | common_knowledge | mixed_domain | wikitext |
|---|--- | --- | --- | ---|
| -0.50 | 38.8791 | 39.0099 | 39.4869 | 39.4164 |
| +0.01 | 39.4054 | 38.9914 | 39.1074 | 38.7526 |
| +0.49 | 38.9816 | 38.5771 | 39.5863 | 38.8052 |
| +0.99 | 37.8661 | 37.9622 | 38.7517 | 38.4197 |
| +1.14 (opt) | 37.9380 | 37.8561 | 38.2169 | 38.1087 |
| +1.50 | 38.7640 | 38.5024 | 38.3315 | 38.9375 |

**Pattern Analysis**: Look for categories that improve/degrade at different rates as α changes.

## Squaring Test Analysis: [W(λ)]² = I Analog

This experiment tests whether neural loss landscapes exhibit rotation-like symmetry properties
analogous to the [W(λ)]² = I property in rotation groups (Eckmann & Tlusty, 2025).

We evaluate both L(α) and L(2α) at each α value to identify "squaring return points" where
doubling the task vector scaling returns the loss to approximately the base model level.

### Squaring Return Points Found ★

Found 15 α value(s) where L(2α) ≈ L(M_base):

#### Detailed Comparison Table

| # | α | L(α) | L(2α) | |L(α) - L_base| | |L(2α) - L_base| |
|---|---|------|-------|----------------|-----------------|
| 1 | -0.4798 | 39.4786 | 39.1096 | 0.411301 | 0.042286 |
| 2 | -0.4192 | 39.5502 | 38.9004 | 0.482975 | 0.166822 |
| 3 | -0.3788 | 39.0738 | 38.9093 | 0.006541 | 0.157968 |
| 4 | -0.3384 | 39.1746 | 39.1303 | 0.107336 | 0.063036 |
| 5 | -0.3182 | 38.9594 | 38.9643 | 0.107846 | 0.103001 |
| 6 | -0.1970 | 39.1621 | 39.1907 | 0.094844 | 0.123465 |
| 7 | -0.1566 | 38.9890 | 38.9502 | 0.078289 | 0.117070 |
| 8 | +0.2475 | 38.5000 | 39.0042 | 0.567249 | 0.063108 |
| 9 | +0.2879 | 38.6233 | 39.2103 | 0.443923 | 0.143028 |
| 10 | +0.6717 | 38.2320 | 39.2206 | 0.835315 | 0.153309 |
| 11 | +0.6919 | 38.4763 | 38.9739 | 0.590995 | 0.093420 |
| 12 | +0.7121 | 38.7195 | 38.9938 | 0.347750 | 0.073496 |
| 13 | +0.7525 | 38.8401 | 38.8889 | 0.227187 | 0.178332 |
| 14 | +0.7727 | 38.3434 | 38.8785 | 0.723891 | 0.188736 |
| 15 | +0.8131 | 38.6010 | 39.0502 | 0.466226 | 0.017083 |

**Interpretation**: These α values suggest special scaling factors where doubling the task
vector exhibits self-inverse-like properties. This suggests neural loss landscapes MAY exhibit
rotation-like symmetry!

**Connection to paper**: Analogous to the 180° rotations in SO(3) that square to identity,
these α values represent "functional return" points under doubling in the neural loss landscape.

**Pattern to watch**: If |L(2α) - L_base| << |L(α) - L_base|, this indicates strong "squaring return" behavior.




## 2D Task Vector Composition Analysis

This experiment explores the loss landscape under composition of two task vectors:
**L(M_base + α·T1 + β·T2)**

### Experiment Setup

- **First Task Vector (T1)**: sentiment_positive
  - Magnitude: ||T1|| = 36.2709
- **Second Task Vector (T2)**: sentiment_negative
  - Magnitude: ||T2|| = 38.2143
- **Grid Configuration**: 30×30 = 900 evaluations
- **α range**: [-2.0, 2.0]
- **β range**: [-2.0, 2.0]
- **Base Model Loss**: L(M_base) = 3.9862

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.0690, β = +0.0690)
- **Loss**: 2.0350
- **Improvement over base**: +1.9512
- **Perplexity**: 7.65

#### Maximum Loss (Worst Composition)
- **Location**: (α = +2.0000, β = +2.0000)
- **Loss**: 41.7603
- **Degradation from base**: +37.7740
- **Perplexity**: 1368530438033294848.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = -0.6207, β = +0.6207)
- **Loss**: 4.0011
- **|L - L_base|**: 0.014892

### Loss Landscape Statistics

- **Mean Loss**: 20.7993
- **Std Dev**: 11.1947
- **Loss Range**: [2.0350, 41.7603]
- **Mean Functional Return**: 16.9132

### 2D Landscape Cross-Sections (Raw Data Samples)

These tables show loss values along key axes and diagonal:


#### Along Diagonal (α ≈ β): Balanced Composition

| α | β | Loss | ΔL from base | PPL |
|---|---|------|--------------|-----|
| -2.000 | -2.000 | 34.4456 | 30.459333 | 910992356909582.25 |
| -1.724 | -1.724 | 32.8469 | 28.860681 | 184174305456874.66 |
| -1.448 | -1.448 | 36.6822 | 32.696021 | 8528960251085671.00 |
| -1.172 | -1.172 | 35.6908 | 31.704614 | 3164706910352768.50 |
| -0.897 | -0.897 | 32.8452 | 28.859007 | 183866275958933.12 |
| -0.621 | -0.621 | 28.1840 | 24.197817 | 1738493976975.27 |
| -0.345 | -0.345 | 17.8005 | 13.814304 | 53786156.75 |
| -0.069 | -0.069 | 6.3475 | 2.361231 | 571.04 |
| +0.207 | +0.207 | 2.3396 | 1.646623 | 10.38 |
| +0.483 | +0.483 | 4.1102 | 0.123940 | 60.96 |
| +0.759 | +0.759 | 6.8653 | 2.879079 | 958.44 |
| +1.034 | +1.034 | 10.3654 | 6.379152 | 31741.33 |

### Interpretation

The 2D composition experiment reveals how two task vectors interact when combined.
The loss landscape shows whether:

1. **Additive effects**: Task vectors combine linearly (smooth gradients)
2. **Synergistic effects**: Certain combinations perform better than either task alone
3. **Interference effects**: Task vectors cancel or degrade performance when combined
4. **Rotation-like patterns**: Circular or symmetric patterns suggesting geometric structure

**Visualization**: See `loss_landscape_2d.png` for the complete heatmap showing L(α,β) across
the entire grid. The heatmap uses color intensity to show loss values, with the base model
marked at the origin (α=0, β=0).

**Data**: Full numerical results available in `loss_landscape_2d_results.json`


## 3D Task Vector Composition Analysis

This experiment explores the loss landscape under composition of three task vectors:
**L(M_base + α·T1 + β·T2 + γ·T3)**

### Experiment Setup

- **First Task Vector (T1)**: sentiment_negative
  - Magnitude: ||T1|| = 38.2143
- **Second Task Vector (T2)**: sentiment_positive
  - Magnitude: ||T2|| = 36.2709
- **Third Task Vector (T3)**: instruction_following
  - Magnitude: ||T3|| = 33.0267
- **Grid Configuration**: 10×10×10 = 1000 evaluations
- **α range**: [-1.0, 1.0]
- **β range**: [-1.0, 1.0]
- **γ range**: [-1.0, 1.0]
- **Base Model Loss**: L(M_base) = 3.9862

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.1111, β = +0.1111, γ = -0.1111)
- **Loss**: 2.0981
- **Improvement over base**: +1.8882
- **Perplexity**: 8.15

#### Maximum Loss (Worst Composition)
- **Location**: (α = -1.0000, β = -1.0000, γ = +0.5556)
- **Loss**: 39.4497
- **Degradation from base**: +35.4635
- **Perplexity**: 135761215696018864.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = +0.3333, β = +0.5556, γ = +0.1111)
- **Loss**: 3.9791
- **|L - L_base|**: 0.007102

### Loss Landscape Statistics

- **Mean Loss**: 13.7382
- **Std Dev**: 10.0893
- **Loss Range**: [2.0981, 39.4497]
- **Mean Functional Return**: 9.8581

### Interpretation

The 3D composition experiment reveals how three task vectors interact when combined simultaneously.
This higher-dimensional exploration allows us to investigate:

1. **Three-way interactions**: How tasks influence each other beyond pairwise combinations
2. **Optimal subspaces**: Whether optimal scaling occurs along specific 2D planes or 1D lines
3. **Symmetry patterns**: Whether the landscape exhibits rotational or other geometric symmetries
4. **Compositional principles**: Rules governing how multiple task vectors should be combined

### Visualization

- **Interactive 3D Plot**: See `loss_landscape_3d_interactive.html` for an interactive 3D scatter plot
  showing the complete loss landscape. Use your browser to rotate and explore the 3D structure.
- **2D Slices**: See `loss_landscape_3d_slices.png` for cross-sectional views at different γ values,
  showing how the loss landscape changes as the third task vector scaling varies.

**Data**: Full numerical results available in `loss_landscape_3d_results.json`

### Research Implications

Three-task composition provides insights into:
- **Multi-task learning**: How to optimally combine multiple capabilities
- **Model merging**: Strategies for merging models trained on different tasks
- **Task interference**: Understanding when tasks help or hinder each other
- **Dimensionality**: Whether task vector combinations live in lower-dimensional subspaces


## Connection to Theoretical Background

**Paper Reference**: Eckmann & Tlusty (2025), "Walks in Rotation Spaces Return Home when Doubled and Scaled"
(arXiv:2502.14367v3)

### Key Theorem

The paper proves that for rotation groups SO(3)/SU(2), almost any walk W can be scaled to reach
a 180° rotation R(n,π), which when squared returns to identity: R(n,π)² = I. This property is
abundant (density 2/π ≈ 64%).

### Our Experiment

**Rotation groups vs Task vectors:**
- **Rotation groups**: Multiplicative group with composition W₁ ∘ W₂
- **Task vectors**: Additive vector space with addition v₁ + v₂
- **Key difference**: Task vectors lack the group structure required for the theorem

**Question**: Do neural loss landscapes exhibit analogous "functional return" properties under scaling?

**Finding**: Yes - found both zero-crossings and squaring return points suggesting rotation-like symmetry

### Implications

The task vector approach explores whether the geometric properties proven for rotation groups
have analogs in the parameter space of neural networks. While task vectors lack the formal group
structure, identifying special scaling factors (zero-crossings, squaring returns) suggests that
loss landscapes may exhibit similar functional symmetries.

This connection opens interesting questions about the geometry of neural network parameter spaces
and whether principles from group theory and differential geometry can inform model merging and
task composition strategies.

## Practical Recommendations

1. **For General Knowledge**: Use α = +1.1364
2. **For Task Performance**: Use α = +1.4596
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*