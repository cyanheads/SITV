# SITV Experiment Report

**Generated**: 2025-11-01 02:24:12
**Model**: google/gemma-3-4b-it
**Task**: sentiment_positive
**General Evaluation Dataset**: combined
**Device**: cuda

## Executive Summary

- **Best General α**: +0.1102 (Loss: 2.0781)
- **Best Task α**: +0.1780 (Loss: 2.8754)
- **Zero-Crossings**: 2 found
- **Total Duration**: 13.3 minutes

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
- **Final Training Loss**: 0.2347

### Task Vector
- **Magnitude (||T||)**: 35.0070
- **Computation Time**: 16.66s

### Alpha Sweep
- **Alpha Range**: [-0.5, 1.5]
- **Samples**: 100
- **Sampling Strategy**: Adaptive
- **Avg Time per Sample**: 1.68s

## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | 1.1m | 8.2% |
| Alpha Sweep | 1.7m | 12.8% |
| **Total** | **13.3m** | **100%** |

## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
| 10 | 0.11 | 1.4900 | 9.76e-05 | 3.11 |
| 20 | 0.21 | 0.3328 | 9.49e-05 | 2.00 |
| 30 | 0.32 | 0.2551 | 9.23e-05 | 1.52 |
| 40 | 0.43 | 0.2195 | 8.96e-05 | 1.06 |
| 50 | 0.53 | 0.2261 | 8.70e-05 | 1.12 |
| 60 | 0.64 | 0.2207 | 8.43e-05 | 1.14 |
| 70 | 0.74 | 0.2080 | 8.16e-05 | 1.01 |
| 80 | 0.85 | 0.1982 | 7.90e-05 | 0.95 |
| 90 | 0.96 | 0.2072 | 7.63e-05 | 1.30 |
| 100 | 1.06 | 0.2091 | 7.37e-05 | 1.03 |
| 180 | 1.91 | 0.1916 | 5.24e-05 | 0.84 |
| 190 | 2.02 | 0.1907 | 4.97e-05 | 1.13 |
| 200 | 2.13 | 0.1933 | 4.71e-05 | 0.79 |
| 210 | 2.23 | 0.1919 | 4.44e-05 | 0.84 |
| 220 | 2.34 | 0.1910 | 4.18e-05 | 0.81 |
| 290 | 3.09 | 0.1871 | 2.31e-05 | 0.66 |
| 300 | 3.19 | 0.1885 | 2.05e-05 | 0.98 |
| 310 | 3.30 | 0.1851 | 1.78e-05 | 0.68 |
| 320 | 3.40 | 0.1871 | 1.52e-05 | 0.97 |
| 330 | 3.51 | 0.1843 | 1.25e-05 | 0.74 |
| 340 | 3.62 | 0.1866 | 9.84e-06 | 0.59 |
| 350 | 3.72 | 0.1826 | 7.18e-06 | 0.64 |
| 360 | 3.83 | 0.1846 | 4.52e-06 | 1.17 |
| 370 | 3.94 | 0.1845 | 1.86e-06 | 0.82 |
| 376 | 4.00 | 0.0000 | 0.00e+00 | 0.00 |

### Training Summary
- **Initial Loss**: 1.4900
- **Final Loss**: 0.1845
- **Loss Improvement**: +1.3055
- **Mean Gradient Norm**: 1.06 (σ=0.43)
- **Total Training Steps**: 38


## Results Summary

### Minimum General Loss
- **α**: +0.1102
- **Loss**: 2.0781
- **Δ from base**: -1.8037

### Zero-Crossings Found ★

1. α = +0.6525, |ΔL| = 0.114406
2. α = +0.6864, |ΔL| = 0.069711


## Alpha Sweep Details

**Base Model Loss**: L(M_base) = 3.8818

### Sample Data Points

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
| -0.500 | 14.0477 | 27.9893 | 10.165889 | 24.107508 | 37.2322 | 1261342.05 | 1430873384276.39 |
| -0.398 | 11.9574 | 20.6746 | 8.075652 | 16.792784 | 34.3291 | 155974.73 | 952481711.38 |
| -0.297 | 10.1520 | 16.2150 | 6.270249 | 12.333178 | 29.0979 | 25643.53 | 11017279.75 |
| -0.195 | 8.1483 | 11.7944 | 4.266528 | 7.912635 | 22.8802 | 3457.59 | 132512.55 |
| -0.093 | 4.3699 | 7.9860 | 0.488107 | 4.104194 | 9.9280 | 79.04 | 2939.49 |
| +0.008 | 3.2625 | 3.1689 | 0.619249 | 0.712849 | 4.9398 | 26.12 | 23.78 |
| +0.110 | 2.0781 | 2.1583 | 1.803686 | 1.723473 | 3.0030 | 7.99 | 8.66 |
| +0.212 | 2.1425 | 2.7222 | 1.739328 | 1.159556 | 2.8872 | 8.52 | 15.21 |
| +0.314 | 2.3800 | 3.6345 | 1.501772 | 0.247290 | 3.0774 | 10.81 | 37.88 |
| +0.415 | 2.6877 | 4.8390 | 1.194078 | 0.957169 | 3.4571 | 14.70 | 126.34 |
| +0.517 | 3.1082 | 6.3244 | 0.773632 | 2.442570 | 4.0166 | 22.38 | 558.00 |
| +0.619 | 3.5785 | 8.2223 | 0.303300 | 4.340506 | 4.6681 | 35.82 | 3723.07 |
| +0.720 | 4.1471 | 10.5955 | 0.265268 | 6.713742 | 5.4953 | 63.25 | 39956.22 |
| +0.822 | 4.7853 | 13.5467 | 0.903544 | 9.664914 | 6.4928 | 119.74 | 764297.09 |
| +0.924 | 5.5025 | 17.0549 | 1.620729 | 13.173064 | 7.6140 | 245.31 | 25517159.48 |
| +1.025 | 6.2476 | 21.1863 | 2.365776 | 17.304543 | 8.7489 | 516.76 | 1588952111.19 |
| +1.127 | 7.1403 | 25.7131 | 3.258453 | 21.831339 | 10.1104 | 1261.74 | 146917445829.13 |
| +1.229 | 8.1386 | 30.6864 | 4.256773 | 26.804643 | 11.5168 | 3424.02 | 21230095986116.12 |
| +1.331 | 9.2563 | 37.1806 | 5.374513 | 33.298772 | 13.0117 | 10470.43 | 14038356276355572.00 |
| +1.432 | 10.4848 | 44.8050 | 6.603030 | 40.923183 | 14.5042 | 35768.67 | 28744574867326816256.00 |
| +1.500 | 11.3952 | 51.0006 | 7.513414 | 47.118832 | 15.5662 | 88895.01 | 14102364412756086489088.00 |

### Key Metrics
- **Total Samples**: 60
- **Alpha Range**: [-0.5, 1.5]
- **Alpha Step Size**: 0.0202
- **Total Sweep Time**: 1.7 minutes
- **Avg Time per Sample**: 1.68s


## Statistical Summary

### Loss Distribution (General)
- **Mean**: 6.1537
- **Std Dev**: 3.3555
- **Min**: 2.0781 (at α = +0.1102)
- **Max**: 14.0477 (at α = -0.5000)

### Functional Return Distribution
- **Mean**: 3.0733
- **Std Dev**: 2.6411
- **Min**: 0.069711 (at α = +0.6864)
- **Max**: 10.1659 (at α = -0.5000)

### Task Performance Distribution
- **Mean**: 11.5132
- **Std Dev**: 9.7580
- **Min**: 2.8754 (at α = +0.1780)
- **Max**: 37.2322 (at α = -0.5000)

## Per-Category Loss Analysis

**Dataset**: combined

This breakdown shows how the task vector (sentiment_positive) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
| coding | +0.1780 | 2.5304 | How well does the model handle programming/technical content? |
| common_knowledge | +0.1102 | 1.8104 | How well does the model handle everyday general knowledge? |
| mixed_domain | +0.1102 | 2.3419 | How well does the model handle diverse multi-domain content? |
| wikitext | +0.0763 | 1.6170 | How well does the model handle factual/encyclopedic content? |

### Baseline Comparison (α ≈ 0.008)

Loss by category before applying task vector:

| Category | Loss |
|----------|------|
| coding | 3.9347 |
| common_knowledge | 2.8142 |
| mixed_domain | 3.7334 |
| wikitext | 2.5474 |

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
| 1 | +0.3475 | 3.9951 | 0.113323 |

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
  - Magnitude: ||T1|| = 35.0070
- **Second Task Vector (T2)**: sentiment_negative
  - Magnitude: ||T2|| = 41.0180
- **Grid Configuration**: 30×30 = 900 evaluations
- **α range**: [-2.0, 2.0]
- **β range**: [-2.0, 2.0]
- **Base Model Loss**: L(M_base) = 6.5198

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.0690, β = -0.8966)
- **Loss**: 2.0255
- **Improvement over base**: +4.4943
- **Perplexity**: 7.58

#### Maximum Loss (Worst Composition)
- **Location**: (α = -2.0000, β = -1.1724)
- **Loss**: 50.6199
- **Degradation from base**: +44.1001
- **Perplexity**: 9637119013814074867712.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = +0.7586, β = -1.7241)
- **Loss**: 6.5234
- **|L - L_base|**: 0.003527

### Loss Landscape Statistics

- **Mean Loss**: 19.3754
- **Std Dev**: 10.9523
- **Loss Range**: [2.0255, 50.6199]
- **Mean Functional Return**: 13.3776

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

1. **For General Knowledge**: Use α = +0.1102
2. **For Task Performance**: Use α = +0.1780
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*