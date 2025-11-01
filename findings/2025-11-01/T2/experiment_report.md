# SITV Experiment Report

**Generated**: 2025-11-01 08:10:05
**Model**: google/gemma-3-4b-it
**Task**: sentiment_positive
**General Evaluation Dataset**: combined
**Device**: cuda

## Executive Summary

- **Best General α**: +0.3889 (Loss: 2.0355)
- **Best Task α**: +0.5505 (Loss: 2.9260)
- **Zero-Crossings**: 9 found
- **Total Duration**: 35.2 minutes

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
- **Computation Time**: 30.97s

### Alpha Sweep
- **Alpha Range**: [-0.5, 1.5]
- **Samples**: 100
- **Sampling Strategy**: Uniform
- **Avg Time per Sample**: 11.00s

## Riemannian Geometry Analysis

### Metric Configuration
- **Metric Type**: fisher_diagonal
- **Fisher Computation Time**: 15.23s (1.4% of sweep time)
- **Fisher Samples**: 1,000

### Task Vector Norms
- **Euclidean Norm ||T||**: 36.2709
- **Riemannian Norm ||T||_g**: 0.1714
- **Ratio ||T||_g / ||T||**: 0.0047
  - *The Fisher metric shrinks the task vector (information-poor directions).*

### Geodesic Integration
- **Enabled**: Yes
- **RK4 Steps per Evaluation**: 100
- **Overhead**: ~100x integration cost vs straight-line

**Note**: Geodesic interpolation uses Runge-Kutta 4 to integrate the geodesic equation
along the Riemannian manifold defined by the Fisher metric. This replaces Euclidean
straight-line interpolation M(α) = M_base + α·T with proper geodesic exp_M(α·T).

### Interpretation
The Riemannian norm accounts for the Fisher Information Matrix (local curvature of
parameter space). This reflects the "information geometry" of the parameter manifold,
where distances are measured according to the KL divergence between model distributions.

## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | 1.2m | 3.3% |
| Alpha Sweep | 18.4m | 52.3% |
| **Total** | **35.2m** | **100%** |

## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
| 10 | 0.11 | 1.4800 | 9.76e-05 | 6.25 |
| 20 | 0.21 | 0.2730 | 9.49e-05 | 2.78 |
| 30 | 0.32 | 0.2592 | 9.23e-05 | 2.06 |
| 40 | 0.43 | 0.2312 | 8.96e-05 | 1.17 |
| 50 | 0.53 | 0.2352 | 8.70e-05 | 1.45 |
| 60 | 0.64 | 0.2327 | 8.43e-05 | 1.30 |
| 70 | 0.74 | 0.2078 | 8.16e-05 | 1.59 |
| 80 | 0.85 | 0.1979 | 7.90e-05 | 1.07 |
| 90 | 0.96 | 0.2069 | 7.63e-05 | 1.28 |
| 100 | 1.06 | 0.2169 | 7.37e-05 | 1.22 |
| 180 | 1.91 | 0.1927 | 5.24e-05 | 0.85 |
| 190 | 2.02 | 0.1907 | 4.97e-05 | 1.17 |
| 200 | 2.13 | 0.1934 | 4.71e-05 | 0.86 |
| 210 | 2.23 | 0.1920 | 4.44e-05 | 0.79 |
| 220 | 2.34 | 0.1911 | 4.18e-05 | 0.83 |
| 290 | 3.09 | 0.1867 | 2.31e-05 | 0.71 |
| 300 | 3.19 | 0.1884 | 2.05e-05 | 1.09 |
| 310 | 3.30 | 0.1850 | 1.78e-05 | 0.68 |
| 320 | 3.40 | 0.1872 | 1.52e-05 | 0.93 |
| 330 | 3.51 | 0.1843 | 1.25e-05 | 0.77 |
| 340 | 3.62 | 0.1864 | 9.84e-06 | 0.59 |
| 350 | 3.72 | 0.1826 | 7.18e-06 | 0.67 |
| 360 | 3.83 | 0.1846 | 4.52e-06 | 1.15 |
| 370 | 3.94 | 0.1846 | 1.86e-06 | 0.89 |
| 376 | 4.00 | 0.0000 | 0.00e+00 | 0.00 |

### Training Summary
- **Initial Loss**: 1.4800
- **Final Loss**: 0.1846
- **Loss Improvement**: +1.2954
- **Mean Gradient Norm**: 1.23 (σ=0.93)
- **Total Training Steps**: 38


## Results Summary

### Minimum General Loss
- **α**: +0.3889
- **Loss**: 2.0355
- **Δ from base**: -1.9507

### Zero-Crossings Found ★

1. α = -0.3788, |ΔL| = 0.150925
2. α = -0.3586, |ΔL| = 0.059837
3. α = -0.3384, |ΔL| = 0.109027


## Alpha Sweep Details

**Base Model Loss**: L(M_base) = 3.9862

### Sample Data Points

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
| -0.500 | 6.4039 | 13.8982 | 2.417649 | 9.911982 | 15.6625 | 604.18 | 1086209.85 |
| -0.399 | 4.6473 | 10.4253 | 0.661115 | 6.439076 | 9.7740 | 104.31 | 33701.56 |
| -0.298 | 3.7335 | 8.0532 | 0.252745 | 4.066983 | 6.7187 | 41.82 | 3143.86 |
| -0.197 | 4.6243 | 4.5509 | 0.638092 | 0.564716 | 7.4177 | 101.93 | 94.72 |
| -0.096 | 4.5281 | 4.8225 | 0.541893 | 0.836250 | 6.6518 | 92.58 | 124.27 |
| +0.005 | 3.9083 | 3.8629 | 0.077942 | 0.123334 | 5.7442 | 49.81 | 47.60 |
| +0.106 | 3.2275 | 3.0388 | 0.758682 | 0.947373 | 4.8239 | 25.22 | 20.88 |
| +0.207 | 3.1468 | 2.0544 | 0.839451 | 1.931794 | 4.5039 | 23.26 | 7.80 |
| +0.308 | 2.3144 | 2.2500 | 1.671824 | 1.736182 | 3.3671 | 10.12 | 9.49 |
| +0.409 | 2.0511 | 2.6724 | 1.935095 | 1.313848 | 3.0419 | 7.78 | 14.47 |
| +0.510 | 2.1034 | 3.3345 | 1.882775 | 0.651739 | 2.9333 | 8.19 | 28.06 |
| +0.611 | 2.2409 | 4.3116 | 1.745281 | 0.325376 | 2.9425 | 9.40 | 74.56 |
| +0.712 | 2.4245 | 5.6434 | 1.561680 | 1.657205 | 3.0418 | 11.30 | 282.43 |
| +0.813 | 2.6645 | 7.4394 | 1.321696 | 3.453191 | 3.2521 | 14.36 | 1701.75 |
| +0.914 | 2.9699 | 9.4633 | 1.016314 | 5.477073 | 3.5955 | 19.49 | 12878.26 |
| +1.015 | 3.3265 | 12.2698 | 0.659759 | 8.283605 | 4.0331 | 27.84 | 213166.37 |
| +1.116 | 3.8098 | 15.8087 | 0.176425 | 11.822521 | 4.6322 | 45.14 | 7339227.23 |
| +1.217 | 4.3035 | 20.1282 | 0.317285 | 16.141951 | 5.2483 | 73.96 | 551512029.97 |
| +1.318 | 4.9090 | 24.3219 | 0.922736 | 20.335647 | 6.0143 | 135.50 | 36547177112.68 |
| +1.419 | 5.6352 | 29.7276 | 1.649006 | 25.741391 | 6.9602 | 280.12 | 8138391755673.79 |
| +1.500 | 6.2769 | 33.9979 | 2.290683 | 30.011682 | 7.7673 | 532.14 | 582240651217943.25 |

### Key Metrics
- **Total Samples**: 100
- **Alpha Range**: [-0.5, 1.5]
- **Alpha Step Size**: 0.0202
- **Total Sweep Time**: 18.4 minutes
- **Avg Time per Sample**: 11.00s


## Statistical Summary

### Loss Distribution (General)
- **Mean**: 3.6408
- **Std Dev**: 1.1906
- **Min**: 2.0355 (at α = +0.3889)
- **Max**: 6.4039 (at α = -0.5000)

### Functional Return Distribution
- **Mean**: 1.0656
- **Std Dev**: 0.6335
- **Min**: 0.000291 (at α = +1.1566)
- **Max**: 2.4176 (at α = -0.5000)

### Task Performance Distribution
- **Mean**: 5.3455
- **Std Dev**: 2.5022
- **Min**: 2.9260 (at α = +0.5505)
- **Max**: 15.6625 (at α = -0.5000)

## Per-Category Loss Analysis

**Dataset**: combined

This breakdown shows how the task vector (sentiment_positive) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
| coding | +0.3889 | 2.5036 | How well does the model handle programming/technical content? |
| common_knowledge | +0.3687 | 1.7679 | How well does the model handle everyday general knowledge? |
| mixed_domain | +0.3889 | 2.3138 | How well does the model handle diverse multi-domain content? |
| wikitext | +0.3687 | 1.6124 | How well does the model handle factual/encyclopedic content? |

### Baseline Comparison (α ≈ 0.005)

Loss by category before applying task vector:

| Category | Loss |
|----------|------|
| coding | 4.7302 |
| common_knowledge | 3.4380 |
| mixed_domain | 4.4622 |
| wikitext | 3.1029 |

**Insight**: Lower values indicate better performance in that domain.

## Squaring Test Analysis: [W(λ)]² = I Analog

This experiment tests whether neural loss landscapes exhibit rotation-like symmetry properties
analogous to the [W(λ)]² = I property in rotation groups (Eckmann & Tlusty, 2025).

We evaluate both L(α) and L(2α) at each α value to identify "squaring return points" where
doubling the task vector scaling returns the loss to approximately the base model level.

### Squaring Return Points Found ★

Found 4 α value(s) where L(2α) ≈ L(M_base):

| # | α | L(2α) | |L(2α) - L_base| |
|---|---|-------|----------------|
| 1 | -0.1768 | 4.0071 | 0.020854 |
| 2 | -0.1566 | 3.8925 | 0.093676 |
| 3 | +0.5707 | 3.9401 | 0.046131 |
| 4 | +0.5909 | 4.1212 | 0.135010 |

**Interpretation**: These α values suggest special scaling factors where doubling the task
vector exhibits self-inverse-like properties. This suggests neural loss landscapes MAY exhibit
rotation-like symmetry!

**Connection to paper**: Analogous to the 180° rotations in SO(3) that square to identity,
these α values represent "functional return" points under doubling in the neural loss landscape.




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
- **Base Model Loss**: L(M_base) = 6.8506

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.0690, β = -0.8966)
- **Loss**: 2.0201
- **Improvement over base**: +4.8305
- **Perplexity**: 7.54

#### Maximum Loss (Worst Composition)
- **Location**: (α = +2.0000, β = +2.0000)
- **Loss**: 62.4746
- **Degradation from base**: +55.6240
- **Perplexity**: 1356291842425400492944261120.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = -0.6207, β = -0.0690)
- **Loss**: 6.8178
- **|L - L_base|**: 0.032713

### Loss Landscape Statistics

- **Mean Loss**: 22.8396
- **Std Dev**: 12.7731
- **Loss Range**: [2.0201, 62.4746]
- **Mean Functional Return**: 16.6288

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
  - Magnitude: ||T1|| = 58.7221
- **Second Task Vector (T2)**: sentiment_positive
  - Magnitude: ||T2|| = 69.6515
- **Third Task Vector (T3)**: instruction_following
  - Magnitude: ||T3|| = 78.6190
- **Grid Configuration**: 10×10×10 = 1000 evaluations
- **α range**: [-1.0, 1.0]
- **β range**: [-1.0, 1.0]
- **γ range**: [-1.0, 1.0]
- **Base Model Loss**: L(M_base) = 9.5865

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.1111, β = +0.1111, γ = -1.0000)
- **Loss**: 2.0530
- **Improvement over base**: +7.5335
- **Perplexity**: 7.79

#### Maximum Loss (Worst Composition)
- **Location**: (α = +1.0000, β = -0.5556, γ = -1.0000)
- **Loss**: 165.4428
- **Degradation from base**: +155.8564
- **Perplexity**: 709439529407609533029425293731163682764954856637587570464076833973338112.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = -0.5556, β = +0.5556, γ = -0.1111)
- **Loss**: 9.5862
- **|L - L_base|**: 0.000262

### Loss Landscape Statistics

- **Mean Loss**: 16.4786
- **Std Dev**: 17.6603
- **Loss Range**: [2.0530, 165.4428]
- **Mean Functional Return**: 8.3612

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

1. **For General Knowledge**: Use α = +0.3889
2. **For Task Performance**: Use α = +0.5505
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*