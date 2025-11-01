# SITV Experiment Report

**Generated**: 2025-11-01 03:13:28
**Model**: google/gemma-3-4b-it
**Task**: sentiment_negative
**General Evaluation Dataset**: combined
**Device**: cuda

## Executive Summary

- **Best General α**: +0.1869 (Loss: 2.0047)
- **Best Task α**: +0.1869 (Loss: 3.3859)
- **Zero-Crossings**: 3 found
- **Total Duration**: 7.8 minutes

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
- **Final Training Loss**: 0.2539

### Task Vector
- **Magnitude (||T||)**: 41.0180
- **Computation Time**: 17.00s

### Alpha Sweep
- **Alpha Range**: [-0.5, 1.5]
- **Samples**: 100
- **Sampling Strategy**: Uniform
- **Avg Time per Sample**: 0.63s

## Timing Breakdown

| Phase | Duration | Percentage |
|-------|----------|------------|
| Fine-tuning | 1.1m | 13.8% |
| Alpha Sweep | 1.1m | 13.7% |
| **Total** | **7.8m** | **100%** |

## Training History

### Training Progress (Key Steps)

| Step | Epoch | Loss | Learning Rate | Grad Norm |
|------|-------|------|---------------|-----------|
| 10 | 0.11 | 1.7328 | 9.76e-05 | 7.12 |
| 20 | 0.21 | 0.3314 | 9.49e-05 | 2.17 |
| 30 | 0.32 | 0.2498 | 9.23e-05 | 8.00 |
| 40 | 0.43 | 0.2409 | 8.96e-05 | 3.38 |
| 50 | 0.53 | 0.2631 | 8.70e-05 | 2.05 |
| 60 | 0.64 | 0.2530 | 8.43e-05 | 1.45 |
| 70 | 0.74 | 0.2503 | 8.16e-05 | 0.72 |
| 80 | 0.85 | 0.2146 | 7.90e-05 | 1.12 |
| 90 | 0.96 | 0.2122 | 7.63e-05 | 1.07 |
| 100 | 1.06 | 0.2181 | 7.37e-05 | 0.80 |
| 180 | 1.91 | 0.2031 | 5.24e-05 | 0.86 |
| 190 | 2.02 | 0.2035 | 4.97e-05 | 0.89 |
| 200 | 2.13 | 0.2093 | 4.71e-05 | 0.75 |
| 210 | 2.23 | 0.2032 | 4.44e-05 | 0.80 |
| 220 | 2.34 | 0.2020 | 4.18e-05 | 0.77 |
| 290 | 3.09 | 0.1978 | 2.31e-05 | 0.62 |
| 300 | 3.19 | 0.1982 | 2.05e-05 | 0.68 |
| 310 | 3.30 | 0.1947 | 1.78e-05 | 0.64 |
| 320 | 3.40 | 0.1997 | 1.52e-05 | 0.70 |
| 330 | 3.51 | 0.1970 | 1.25e-05 | 0.75 |
| 340 | 3.62 | 0.1961 | 9.84e-06 | 0.55 |
| 350 | 3.72 | 0.1963 | 7.18e-06 | 0.60 |
| 360 | 3.83 | 0.1948 | 4.52e-06 | 0.80 |
| 370 | 3.94 | 0.1969 | 1.86e-06 | 0.77 |
| 376 | 4.00 | 0.0000 | 0.00e+00 | 0.00 |

### Training Summary
- **Initial Loss**: 1.7328
- **Final Loss**: 0.1969
- **Loss Improvement**: +1.5359
- **Mean Gradient Norm**: 1.32 (σ=1.59)
- **Total Training Steps**: 38


## Results Summary

### Minimum General Loss
- **α**: +0.1869
- **Loss**: 2.0047
- **Δ from base**: -1.9815

### Zero-Crossings Found ★

1. α = +0.6515, |ΔL| = 0.189773
2. α = +0.6717, |ΔL| = 0.073509
3. α = +0.6919, |ΔL| = 0.063107


## Alpha Sweep Details

**Base Model Loss**: L(M_base) = 3.9862

### Sample Data Points

| α | L(α) | L(2α) | |ΔL| | |ΔL(2α)| | Task L(α) | PPL(α) | PPL(2α) |
|---|------|-------|--------|------------|-----------|--------|---------|
| -0.500 | 14.3098 | 29.3121 | 10.323535 | 25.325887 | 29.8484 | 1639264.13 | 5371400996503.84 |
| -0.399 | 11.8996 | 24.0826 | 7.913422 | 20.096374 | 28.5937 | 147214.40 | 28769941168.13 |
| -0.298 | 9.5050 | 17.0321 | 5.518776 | 13.045926 | 25.5151 | 13426.68 | 24944126.30 |
| -0.197 | 6.6506 | 11.8009 | 2.664396 | 7.814635 | 17.3482 | 773.26 | 133366.78 |
| -0.096 | 4.2678 | 6.5358 | 0.281600 | 2.549563 | 8.2448 | 71.37 | 689.38 |
| +0.005 | 3.1149 | 3.1214 | 0.871373 | 0.864794 | 5.4793 | 22.53 | 22.68 |
| +0.106 | 2.1666 | 2.0339 | 1.819669 | 1.952320 | 3.7522 | 8.73 | 7.64 |
| +0.207 | 2.0269 | 2.5528 | 1.959305 | 1.433455 | 3.4012 | 7.59 | 12.84 |
| +0.308 | 2.2232 | 3.5688 | 1.762973 | 0.417466 | 3.6573 | 9.24 | 35.47 |
| +0.409 | 2.5358 | 4.9639 | 1.450401 | 0.977723 | 4.0636 | 12.63 | 143.16 |
| +0.510 | 2.9985 | 6.5948 | 0.987717 | 2.608546 | 4.5559 | 20.06 | 731.26 |
| +0.611 | 3.5363 | 8.4595 | 0.449971 | 4.473282 | 5.0248 | 34.34 | 4719.72 |
| +0.712 | 4.1914 | 10.4257 | 0.205181 | 6.439450 | 5.5841 | 66.12 | 33714.15 |
| +0.813 | 4.9256 | 12.5128 | 0.939360 | 8.526611 | 6.1927 | 137.77 | 271803.38 |
| +0.914 | 5.7467 | 14.8182 | 1.760478 | 10.832011 | 6.8834 | 313.16 | 2725694.82 |
| +1.015 | 6.5423 | 17.2594 | 2.556066 | 13.273186 | 7.5129 | 693.87 | 31308786.42 |
| +1.116 | 7.4640 | 19.8227 | 3.477741 | 15.836509 | 8.2319 | 1744.05 | 406352586.77 |
| +1.217 | 8.4099 | 22.3176 | 4.423636 | 18.331394 | 8.9436 | 4491.13 | 4925132599.24 |
| +1.318 | 9.3953 | 24.6254 | 5.409040 | 20.639195 | 9.6866 | 12031.26 | 49508865039.54 |
| +1.419 | 10.3717 | 26.7874 | 6.385454 | 22.801175 | 10.4433 | 31942.01 | 430149334440.52 |
| +1.500 | 11.1864 | 28.3887 | 7.200172 | 24.402502 | 11.1074 | 72142.26 | 2133372721719.68 |

### Key Metrics
- **Total Samples**: 100
- **Alpha Range**: [-0.5, 1.5]
- **Alpha Step Size**: 0.0202
- **Total Sweep Time**: 1.1 minutes
- **Avg Time per Sample**: 0.63s


## Statistical Summary

### Loss Distribution (General)
- **Mean**: 6.0533
- **Std Dev**: 3.3227
- **Min**: 2.0047 (at α = +0.1869)
- **Max**: 14.3098 (at α = -0.5000)

### Functional Return Distribution
- **Mean**: 2.9797
- **Std Dev**: 2.5367
- **Min**: 0.063107 (at α = +0.6919)
- **Max**: 10.3235 (at α = -0.5000)

### Task Performance Distribution
- **Mean**: 9.7940
- **Std Dev**: 7.6385
- **Min**: 3.3859 (at α = +0.1869)
- **Max**: 30.0605 (at α = -0.4798)

## Per-Category Loss Analysis

**Dataset**: combined

This breakdown shows how the task vector (sentiment_negative) affects each evaluation domain separately.

### Best α for Each Category

| Category | Best α | Loss at Best α | Interpretation |
|----------|--------|----------------|----------------|
| coding | +0.1869 | 2.5024 | How well does the model handle programming/technical content? |
| common_knowledge | +0.1869 | 1.7448 | How well does the model handle everyday general knowledge? |
| mixed_domain | +0.1869 | 2.2225 | How well does the model handle diverse multi-domain content? |
| wikitext | +0.1667 | 1.6160 | How well does the model handle factual/encyclopedic content? |

### Baseline Comparison (α ≈ 0.005)

Loss by category before applying task vector:

| Category | Loss |
|----------|------|
| coding | 3.8502 |
| common_knowledge | 2.7013 |
| mixed_domain | 3.5997 |
| wikitext | 2.4193 |

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
| 1 | +0.3283 | 3.8314 | 0.154865 |
| 2 | +0.3485 | 4.0812 | 0.095025 |

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
  - Magnitude: ||T1|| = 41.0180
- **Second Task Vector (T2)**: sentiment_negative
  - Magnitude: ||T2|| = 35.0070
- **Grid Configuration**: 30×30 = 900 evaluations
- **α range**: [-2.0, 2.0]
- **β range**: [-2.0, 2.0]
- **Base Model Loss**: L(M_base) = 6.0168

### Key Findings

#### Minimum Loss (Optimal Composition)
- **Location**: (α = +0.0690, β = -0.8966)
- **Loss**: 2.0263
- **Improvement over base**: +3.9905
- **Perplexity**: 7.59

#### Maximum Loss (Worst Composition)
- **Location**: (α = +2.0000, β = +2.0000)
- **Loss**: 61.1634
- **Degradation from base**: +55.1466
- **Perplexity**: 365522941622989315045851136.00

#### Closest Return to Base (Functional Return)
- **Location**: (α = -0.6207, β = -0.2069)
- **Loss**: 6.0314
- **|L - L_base|**: 0.014636

### Loss Landscape Statistics

- **Mean Loss**: 21.6223
- **Std Dev**: 13.3087
- **Loss Range**: [2.0263, 61.1634]
- **Mean Functional Return**: 16.0190

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

1. **For General Knowledge**: Use α = +0.1869
2. **For Task Performance**: Use α = +0.1869
3. **For Model Merging**: Consider alpha values near zero-crossings

---
*Generated by SITV - Self-Inverse Task Vectors*