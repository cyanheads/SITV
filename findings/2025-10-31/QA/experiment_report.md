# SITV Experiment Report

**Generated**: 2025-11-01 03:25:15
**Model**: google/gemma-3-4b-it
**Task**: instruction_following
**General Evaluation Dataset**: combined
**Device**: cuda

## Executive Summary

- **Best General α**: +0.0859 (Loss: 2.1882)
- **Best Task α**: +0.2677 (Loss: 1.4110)
- **Zero-Crossings**: 3 found
- **Total Duration**: 7.7 minutes

## Configuration

### Model
- **Model Name**: google/gemma-3-4b-it
- **Total Parameters**: 4,300,079,472
- **Device**: cuda

### Task & Evaluation
- **Training Task**: instruction_following
- **General Evaluation Dataset**: combined
  - *This dataset measures how the task vector affects general language modeling capability*

### Training
- **Training Examples**: 3000
- **Epochs**: 4
- **Learning Rate**: 1.00e-04
- **Training Steps**: 376
- **Final Training Loss**: 0.2106

### Task Vector
- **Magnitude (||T||)**: 33.0286
- **Computation Time**: 17.82s

### Alpha Sweep
- **Alpha Range**: [-0.5, 1.5]
- **Samples**: 100
- **Sampling Strategy**: Uniform
- **Avg Time per Sample**: 0.56s

## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | 1.2m | 15.5% |
| Alpha Sweep | 0.9m | 12.3% |
| **Total** | **7.7m** | **100%** |

## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
| 10 | 0.11 | 1.7229 | 9.76e-05 | 12.31 |
| 20 | 0.21 | 0.3064 | 9.49e-05 | 3.55 |
| 30 | 0.32 | 0.2184 | 9.23e-05 | 2.03 |
| 40 | 0.43 | 0.1990 | 8.96e-05 | 1.57 |
| 50 | 0.53 | 0.1843 | 8.70e-05 | 1.19 |
| 60 | 0.64 | 0.1765 | 8.43e-05 | 1.13 |
| 70 | 0.74 | 0.1713 | 8.16e-05 | 0.93 |
| 80 | 0.85 | 0.1715 | 7.90e-05 | 1.00 |
| 90 | 0.96 | 0.1766 | 7.63e-05 | 1.15 |
| 100 | 1.06 | 0.1779 | 7.37e-05 | 1.07 |
| 180 | 1.91 | 0.1610 | 5.24e-05 | 0.78 |
| 190 | 2.02 | 0.1582 | 4.97e-05 | 0.95 |
| 200 | 2.13 | 0.1683 | 4.71e-05 | 1.80 |
| 210 | 2.23 | 0.1636 | 4.44e-05 | 0.77 |
| 220 | 2.34 | 0.1649 | 4.18e-05 | 1.15 |
| 290 | 3.09 | 0.1537 | 2.31e-05 | 0.99 |
| 300 | 3.19 | 0.1547 | 2.05e-05 | 0.71 |
| 310 | 3.30 | 0.1517 | 1.78e-05 | 0.76 |
| 320 | 3.40 | 0.1546 | 1.52e-05 | 0.88 |
| 330 | 3.51 | 0.1542 | 1.25e-05 | 0.91 |
| 340 | 3.62 | 0.1548 | 9.84e-06 | 0.53 |
| 350 | 3.72 | 0.1539 | 7.18e-06 | 0.62 |
| 360 | 3.83 | 0.1503 | 4.52e-06 | 0.96 |
| 370 | 3.94 | 0.1535 | 1.86e-06 | 0.89 |
| 376 | 4.00 | 0.0000 | 0.00e+00 | 0.00 |

### Training Summary
- **Initial Loss**: 1.7229
- **Final Loss**: 0.1535
- **Loss Improvement**: +1.5694
- **Mean Gradient Norm**: 1.42 (σ=1.88)
- **Total Training Steps**: 38


## Results Summary

### Minimum General Loss
- **α**: +0.0859
- **Loss**: 2.1882
- **Δ from base**: -1.7980

### Zero-Crossings Found ★

1. α = +0.4091, |ΔL| = 0.144710
2. α = +0.4293, |ΔL| = 0.010785
3. α = +0.4495, |ΔL| = 0.196901


## Alpha Sweep Details

**Base Model Loss**: L(M_base) = 3.9862

### Sample Data Points

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
| -0.500 | 9.8139 | 23.9719 | 5.827644 | 19.985635 | 50.6567 | 18285.55 | 25754059140.97 |
| -0.399 | 8.4538 | 16.6390 | 4.467559 | 12.652749 | 50.1318 | 4692.79 | 16835019.58 |
| -0.298 | 7.0880 | 11.3642 | 3.101787 | 7.378019 | 50.1939 | 1197.52 | 86184.23 |
| -0.197 | 6.5535 | 8.4245 | 2.567326 | 4.438239 | 49.2951 | 701.73 | 4557.19 |
| -0.096 | 5.1856 | 6.5293 | 1.199366 | 2.543076 | 44.0877 | 178.68 | 684.92 |
| +0.005 | 3.1255 | 3.2907 | 0.860729 | 0.695523 | 7.9417 | 22.77 | 26.86 |
| +0.106 | 2.2375 | 2.5092 | 1.748728 | 1.476999 | 2.2212 | 9.37 | 12.30 |
| +0.207 | 2.4920 | 3.8799 | 1.494197 | 0.106322 | 1.5425 | 12.09 | 48.42 |
| +0.308 | 3.0988 | 5.7941 | 0.887426 | 1.807848 | 1.4225 | 22.17 | 328.35 |
| +0.409 | 3.8415 | 7.8209 | 0.144710 | 3.834640 | 1.5287 | 46.60 | 2492.06 |
| +0.510 | 4.7926 | 9.7205 | 0.806389 | 5.734293 | 1.6766 | 120.62 | 16655.84 |
| +0.611 | 5.7437 | 11.7309 | 1.757487 | 7.744669 | 1.8369 | 312.22 | 124354.50 |
| +0.712 | 6.7784 | 13.9341 | 2.792207 | 9.947911 | 2.0108 | 878.69 | 1125945.40 |
| +0.813 | 7.7662 | 16.3056 | 3.779935 | 12.319390 | 2.1849 | 2359.39 | 12062517.24 |
| +0.914 | 8.7576 | 18.7680 | 4.771377 | 14.781756 | 2.3843 | 6358.83 | 141523829.64 |
| +1.015 | 9.6608 | 21.1065 | 5.674593 | 17.120323 | 2.5761 | 15690.58 | 1467088467.07 |
| +1.116 | 10.6663 | 23.2206 | 6.680116 | 19.234425 | 2.8031 | 42887.64 | 12150649324.77 |
| +1.217 | 11.6735 | 24.8779 | 7.687322 | 20.891703 | 3.0510 | 117423.77 | 63730286745.72 |
| +1.318 | 12.7634 | 26.2332 | 8.777200 | 22.247024 | 3.3187 | 349207.96 | 247146438634.35 |
| +1.419 | 13.8740 | 27.1443 | 9.887812 | 23.158115 | 3.5991 | 1060270.44 | 614661683248.47 |
| +1.500 | 14.8136 | 27.8512 | 10.827392 | 23.864994 | 3.8439 | 2713132.94 | 1246320385068.07 |

### Key Metrics
- **Total Samples**: 100
- **Alpha Range**: [-0.5, 1.5]
- **Alpha Step Size**: 0.0202
- **Total Sweep Time**: 0.9 minutes
- **Avg Time per Sample**: 0.56s


## Statistical Summary

### Loss Distribution (General)
- **Mean**: 7.3392
- **Std Dev**: 3.5101
- **Min**: 2.1882 (at α = +0.0859)
- **Max**: 14.8136 (at α = +1.5000)

### Functional Return Distribution
- **Mean**: 3.8440
- **Std Dev**: 2.9643
- **Min**: 0.010785 (at α = +0.4293)
- **Max**: 10.8274 (at α = +1.5000)

### Task Performance Distribution
- **Mean**: 13.3689
- **Std Dev**: 19.4036
- **Min**: 1.4110 (at α = +0.2677)
- **Max**: 51.0892 (at α = -0.4798)

## Per-Category Loss Analysis

**Dataset**: combined

This breakdown shows how the task vector (instruction_following) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
| coding | +0.0859 | 2.6947 | How well does the model handle programming/technical content? |
| common_knowledge | +0.0859 | 1.8953 | How well does the model handle everyday general knowledge? |
| mixed_domain | +0.0859 | 2.5277 | How well does the model handle diverse multi-domain content? |
| wikitext | +0.0657 | 1.7114 | How well does the model handle factual/encyclopedic content? |

### Baseline Comparison (α ≈ 0.005)

Loss by category before applying task vector:

| Category | Loss |
|----------|------|
| coding | 3.8702 |
| common_knowledge | 2.7052 |
| mixed_domain | 3.6140 |
| wikitext | 2.4228 |

**Insight**: Lower values indicate better performance in that domain.

## Squaring Test Analysis: [W(λ)]² = I Analog

This experiment tests whether neural loss landscapes exhibit rotation-like symmetry properties
analogous to the [W(λ)]² = I property in rotation groups (Eckmann & Tlusty, 2025).

We evaluate both L(α) and L(2α) at each α value to identify "squaring return points" where
doubling the task vector scaling returns the loss to approximately the base model level.

### Squaring Return Points Found ★

Found 1 α value(s) where L(2α) ≈ L(M_base):

| # | α | L(2α) | |L(2α) - L_base| |
|---|---|-------|----------------|
| 1 | +0.2071 | 3.8799 | 0.106322 |

**Interpretation**: These α values suggest special scaling factors where doubling the task
vector exhibits self-inverse-like properties. This suggests neural loss landscapes MAY exhibit
rotation-like symmetry!

**Connection to paper**: Analogous to the 180° rotations in SO(3) that square to identity,
these α values represent "functional return" points under doubling in the neural loss landscape.


## 2D Task Vector Composition Analysis

This experiment explores the loss landscape under composition of two task vectors:
**L(M_base + α·T1 + β·T2)**

### Experiment Setup

- **First Task Vector (T1)**: instruction_following
  - Magnitude: ||T1|| = 33.0286
- **Second Task Vector (T2)**: sentiment_negative
  - Magnitude: ||T2|| = 41.0180
- **Grid Configuration**: 30×30 = 900 evaluations
- **α range**: [-2.0, 2.0]
- **β range**: [-2.0, 2.0]
- **Base Model Loss**: L(M_base) = 6.4499

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.0690, β = -0.8966)
- **Loss**: 2.0150
- **Improvement over base**: +4.4348
- **Perplexity**: 7.50

#### Maximum Loss (Worst Composition)
- **Location**: (α = -2.0000, β = -0.8966)
- **Loss**: 46.1963
- **Degradation from base**: +39.7465
- **Perplexity**: 115562446787491348480.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = -0.4828, β = -0.0690)
- **Loss**: 6.4645
- **|L - L_base|**: 0.014623

### Loss Landscape Statistics

- **Mean Loss**: 18.2830
- **Std Dev**: 9.3027
- **Loss Range**: [2.0150, 46.1963]
- **Mean Functional Return**: 12.2113

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

1. **For General Knowledge**: Use α = +0.0859
2. **For Task Performance**: Use α = +0.2677
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*