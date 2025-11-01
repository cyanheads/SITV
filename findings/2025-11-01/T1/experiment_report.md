# SITV Experiment Report

**Generated**: 2025-11-01 06:26:51
**Model**: google/gemma-3-4b-it
**Task**: sentiment_negative
**General Evaluation Dataset**: combined
**Device**: cuda

## Executive Summary

- **Best General α**: +0.4293 (Loss: 1.9924)
- **Best Task α**: +0.5505 (Loss: 3.3742)
- **Zero-Crossings**: 5 found
- **Total Duration**: 25.9 minutes

## Configuration

### Model
- **Model Name**: google/gemma-3-4b-it
- **Total Parameters**: 4,300,079,472
- **Device**: cuda

### Task & Evaluation
- **Training Task**: sentiment_negative
- **General Evaluation Dataset**: combined
  - *This dataset measures how the task vector affects general language modeling capability*

### Training
- **Training Examples**: 3000
- **Epochs**: 4
- **Learning Rate**: 1.00e-04
- **Training Steps**: 376
- **Final Training Loss**: 0.2529

### Task Vector
- **Euclidean Magnitude (||T||)**: 38.2143
- **Riemannian Magnitude (||T||_g)**: 0.1719
- **Norm Ratio (||T||_g / ||T||)**: 0.0045
- **Computation Time**: 28.90s

### Alpha Sweep
- **Alpha Range**: [-0.5, 1.5]
- **Samples**: 100
- **Sampling Strategy**: Uniform
- **Avg Time per Sample**: 10.65s

## Riemannian Geometry Analysis

### Metric Configuration
- **Metric Type**: fisher_diagonal
- **Fisher Computation Time**: 14.58s (1.4% of sweep time)
- **Fisher Samples**: 1,000

### Task Vector Norms
- **Euclidean Norm ||T||**: 38.2143
- **Riemannian Norm ||T||_g**: 0.1719
- **Ratio ||T||_g / ||T||**: 0.0045
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
| Fine-tuning | 1.2m | 4.6% |
| Alpha Sweep | 17.8m | 68.6% |
| **Total** | **25.9m** | **100%** |

## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
| 10 | 0.11 | 1.7215 | 9.76e-05 | 6.00 |
| 20 | 0.21 | 0.3493 | 9.49e-05 | 3.95 |
| 30 | 0.32 | 0.2863 | 9.23e-05 | 1.41 |
| 40 | 0.43 | 0.2605 | 8.96e-05 | 1.63 |
| 50 | 0.53 | 0.2280 | 8.70e-05 | 1.21 |
| 60 | 0.64 | 0.2232 | 8.43e-05 | 1.45 |
| 70 | 0.74 | 0.2151 | 8.16e-05 | 0.91 |
| 80 | 0.85 | 0.2114 | 7.90e-05 | 1.08 |
| 90 | 0.96 | 0.2156 | 7.63e-05 | 1.17 |
| 100 | 1.06 | 0.2189 | 7.37e-05 | 0.82 |
| 180 | 1.91 | 0.2041 | 5.24e-05 | 0.80 |
| 190 | 2.02 | 0.2031 | 4.97e-05 | 0.92 |
| 200 | 2.13 | 0.2089 | 4.71e-05 | 0.76 |
| 210 | 2.23 | 0.2028 | 4.44e-05 | 0.82 |
| 220 | 2.34 | 0.2022 | 4.18e-05 | 0.79 |
| 290 | 3.09 | 0.1980 | 2.31e-05 | 0.61 |
| 300 | 3.19 | 0.1982 | 2.05e-05 | 0.71 |
| 310 | 3.30 | 0.1946 | 1.78e-05 | 0.65 |
| 320 | 3.40 | 0.2001 | 1.52e-05 | 0.70 |
| 330 | 3.51 | 0.1970 | 1.25e-05 | 0.76 |
| 340 | 3.62 | 0.1963 | 9.84e-06 | 0.55 |
| 350 | 3.72 | 0.1962 | 7.18e-06 | 0.60 |
| 360 | 3.83 | 0.1948 | 4.52e-06 | 0.81 |
| 370 | 3.94 | 0.1970 | 1.86e-06 | 0.78 |
| 376 | 4.00 | 0.0000 | 0.00e+00 | 0.00 |

### Training Summary
- **Initial Loss**: 1.7215
- **Final Loss**: 0.1970
- **Loss Improvement**: +1.5245
- **Mean Gradient Norm**: 1.11 (σ=0.98)
- **Total Training Steps**: 38


## Results Summary

### Minimum General Loss
- **α**: +0.4293
- **Loss**: 1.9924
- **Δ from base**: -1.9938

### Zero-Crossings Found ★

1. α = -0.3788, |ΔL| = 0.013183
2. α = -0.1970, |ΔL| = 0.176155
3. α = +1.0960, |ΔL| = 0.166532


## Alpha Sweep Details

**Base Model Loss**: L(M_base) = 3.9862

### Sample Data Points

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
| -0.500 | 5.4086 | 12.4515 | 1.422363 | 8.465245 | 10.7654 | 223.32 | 255625.41 |
| -0.399 | 4.4199 | 9.2157 | 0.433709 | 5.229452 | 7.8972 | 83.09 | 10053.49 |
| -0.298 | 3.5056 | 6.3689 | 0.480655 | 2.382676 | 6.2958 | 33.30 | 583.41 |
| -0.197 | 4.1624 | 4.2788 | 0.176155 | 0.292546 | 6.9929 | 64.22 | 72.15 |
| -0.096 | 4.5497 | 4.4078 | 0.563477 | 0.421611 | 7.3731 | 94.60 | 82.09 |
| +0.005 | 3.9751 | 3.8380 | 0.011145 | 0.148266 | 6.5123 | 53.25 | 46.43 |
| +0.106 | 3.1294 | 2.9235 | 0.856838 | 1.062681 | 5.3992 | 22.86 | 18.61 |
| +0.207 | 2.9347 | 2.0240 | 1.051490 | 1.962210 | 4.9811 | 18.82 | 7.57 |
| +0.308 | 2.3633 | 2.1284 | 1.622896 | 1.857826 | 4.1616 | 10.63 | 8.40 |
| +0.409 | 2.0322 | 2.5081 | 1.954071 | 1.478090 | 3.6154 | 7.63 | 12.28 |
| +0.510 | 2.0348 | 3.3218 | 1.951445 | 0.664419 | 3.3786 | 7.65 | 27.71 |
| +0.611 | 2.1203 | 4.6925 | 1.865937 | 0.706242 | 3.4148 | 8.33 | 109.12 |
| +0.712 | 2.2653 | 6.4273 | 1.720901 | 2.441100 | 3.5463 | 9.63 | 618.52 |
| +0.813 | 2.4988 | 8.8625 | 1.487383 | 4.876237 | 3.7649 | 12.17 | 7061.83 |
| +0.914 | 2.8412 | 11.9889 | 1.144998 | 8.002688 | 4.0866 | 17.14 | 160960.00 |
| +1.015 | 3.3090 | 17.0322 | 0.677257 | 13.045991 | 4.4786 | 27.36 | 24945748.73 |
| +1.116 | 3.9752 | 22.6401 | 0.011016 | 18.653838 | 5.0019 | 53.26 | 6799131219.43 |
| +1.217 | 4.6764 | 27.2362 | 0.690217 | 23.249955 | 5.5920 | 107.39 | 673785791272.39 |
| +1.318 | 5.4818 | 29.8160 | 1.495583 | 25.829801 | 6.3640 | 240.28 | 8890673218277.20 |
| +1.419 | 6.4201 | 32.1217 | 2.433906 | 28.135522 | 7.5174 | 614.08 | 89185986930327.23 |
| +1.500 | 7.2595 | 33.6121 | 3.273231 | 29.625832 | 8.6611 | 1421.48 | 395849400065904.19 |

### Key Metrics
- **Total Samples**: 100
- **Alpha Range**: [-0.5, 1.5]
- **Alpha Step Size**: 0.0202
- **Total Sweep Time**: 17.8 minutes
- **Avg Time per Sample**: 10.65s


## Statistical Summary

### Loss Distribution (General)
- **Mean**: 3.6470
- **Std Dev**: 1.3450
- **Min**: 1.9924 (at α = +0.4293)
- **Max**: 7.2595 (at α = +1.5000)

### Functional Return Distribution
- **Mean**: 1.1874
- **Std Dev**: 0.7171
- **Min**: 0.011016 (at α = +1.1162)
- **Max**: 3.2732 (at α = +1.5000)

### Task Performance Distribution
- **Mean**: 5.5202
- **Std Dev**: 1.7717
- **Min**: 3.3742 (at α = +0.5505)
- **Max**: 10.7654 (at α = -0.5000)

## Per-Category Loss Analysis

**Dataset**: combined

This breakdown shows how the task vector (sentiment_negative) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
| coding | +0.4293 | 2.4635 | How well does the model handle programming/technical content? |
| common_knowledge | +0.4293 | 1.7409 | How well does the model handle everyday general knowledge? |
| mixed_domain | +0.4293 | 2.2506 | How well does the model handle diverse multi-domain content? |
| wikitext | +0.4293 | 1.5930 | How well does the model handle factual/encyclopedic content? |

### Baseline Comparison (α ≈ 0.005)

Loss by category before applying task vector:

| Category | Loss |
|----------|------|
| coding | 4.8010 |
| common_knowledge | 3.5001 |
| mixed_domain | 4.5368 |
| wikitext | 3.1640 |

**Insight**: Lower values indicate better performance in that domain.

## Squaring Test Analysis: [W(λ)]² = I Analog

This experiment tests whether neural loss landscapes exhibit rotation-like symmetry properties
analogous to the [W(λ)]² = I property in rotation groups (Eckmann & Tlusty, 2025).

We evaluate both L(α) and L(2α) at each α value to identify "squaring return points" where
doubling the task vector scaling returns the loss to approximately the base model level.

### Squaring Return Points Found ★

Found 2 α value(s) where L(2α) ≈ L(M_base):

| # | α | L(2α) | |L(2α) - L_base| |
|---|---|-------|----------------|
| 1 | +0.5505 | 3.8407 | 0.145474 |
| 2 | +0.5707 | 4.1597 | 0.173428 |

**Interpretation**: These α values suggest special scaling factors where doubling the task
vector exhibits self-inverse-like properties. This suggests neural loss landscapes MAY exhibit
rotation-like symmetry!

**Connection to paper**: Analogous to the 180° rotations in SO(3) that square to identity,
these α values represent "functional return" points under doubling in the neural loss landscape.




## 2D Task Vector Composition Analysis

This experiment explores the loss landscape under composition of two task vectors:
**L(M_base + α·T1 + β·T2)**

### Experiment Setup

- **First Task Vector (T1)**: sentiment_negative
  - Magnitude: ||T1|| = 38.2143
- **Second Task Vector (T2)**: sentiment_positive
  - Magnitude: ||T2|| = 36.2709
- **Grid Configuration**: 30×30 = 900 evaluations
- **α range**: [-2.0, 2.0]
- **β range**: [-2.0, 2.0]
- **Base Model Loss**: L(M_base) = 6.0645

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.0690, β = -0.8966)
- **Loss**: 1.9922
- **Improvement over base**: +4.0723
- **Perplexity**: 7.33

#### Maximum Loss (Worst Composition)
- **Location**: (α = +2.0000, β = +2.0000)
- **Loss**: 64.6606
- **Degradation from base**: +58.5961
- **Perplexity**: 12071284771338195110956367872.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = +0.8966, β = -1.4483)
- **Loss**: 6.0641
- **|L - L_base|**: 0.000392

### Loss Landscape Statistics

- **Mean Loss**: 22.5161
- **Std Dev**: 13.6724
- **Loss Range**: [1.9922, 64.6606]
- **Mean Functional Return**: 16.8952

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

1. **For General Knowledge**: Use α = +0.4293
2. **For Task Performance**: Use α = +0.5505
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*