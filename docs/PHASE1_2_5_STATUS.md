# Phase 1, 2, 4, 5 Implementation Status

## Summary

Implemented differential/flow features extending Tier-0 k-NN geometry.

## Phase 1: Flow Primitives (IMPLEMENTED)

| Feature | Status | Correlation w/ boundary | Notes |
|---------|--------|-------------------------|-------|
| `local_gradient_magnitude` | **RECOMMENDED** | r=+0.196 | Least redundant, +8.68% on borderline |
| `gradient_direction_consistency` | Implemented | - | Full version (slower) |
| `pressure_differential` | **REDUNDANT** | r=+0.192 | r=0.932 with ridge_proximity - drop |
| `d_eff` (effective dimension) | Implemented | - | Used by other features |

### Key Finding
- Phase 1 provides **+8.68% improvement on borderline cases** (hypothesis supported)
- `pressure_differential` highly redundant with Tier-0 - use `local_gradient_magnitude` only

### Recommended Usage
```python
from mirrorfield.geometry import compute_gradient_magnitude_only

# Optimized defaults: k=75, c=1.2
g_mag, meta = compute_gradient_magnitude_only(queries, reference)
```

## Phase 2: Weather Features (IMPLEMENTED)

| Feature | Formula | Measures |
|---------|---------|----------|
| `turbulence_index` | 1 - weight uniformity | Local mixing/disorder |
| `thermal_gradient` | g_mag at boundaries only | Boundary-focused gradient |
| `vorticity` | Position dispersion proxy | Rotational tendency |

### Usage
```python
from mirrorfield.geometry import compute_phase2_features

features, meta = compute_phase2_features(
    queries, reference, g_mag, g_dir, ridge_proximity,
    k=50, include_topology=True
)
```

## Phase 5: Topology-Lite (IMPLEMENTED)

| Feature | Formula | Measures |
|---------|---------|----------|
| `d_eff` | PCA components to 90% energy | Effective local dimension |
| `spectral_entropy` | -sum(p_i log p_i) | Eigenvalue spread |
| `participation_ratio` | (sum lambda)^2 / sum(lambda^2) | Dimension participation |

Reuses SVD from Phase 1 - minimal additional compute.

## Files

- `mirrorfield/geometry/phase1_flow_features.py` - Phase 1 implementation
- `mirrorfield/geometry/phase2_weather_features.py` - Phase 2 + 5 implementation
- `experiments/phase1_flow_evaluation.py` - Evaluation script

## Evaluation Results (Phase 1)

```
BORDERLINE ZONE: +8.68% R^2 improvement (75% win rate)
UNSAFE ZONE:     +2.9% R^2 improvement (80% win rate)
SAFE ZONE:       -6.0% R^2 (expected - features help uncertain regions)
GLOBAL:          +0.69% R^2 (p=0.14, marginal)
```

## Phase 4: State Mapping (IMPLEMENTED)

Maps geometric signatures to interpretable cognitive states:

| State | Geometric Signature | Boundary Distance |
|-------|---------------------|-------------------|
| `uncertain` | Dispersed, boundary-adjacent | +0.435 (highest risk) |
| `novel_territory` | Sparse, flat region | +0.272 |
| `constraint_pressure` | Multi-basin tension | +0.111 |
| `confident` | Tight clustering, strong pull | -0.114 |
| `coherent` | Consistent flow toward attractor | -0.294 |
| `searching` | Inconsistent, turbulent flow | -0.452 |

### Usage
```python
from mirrorfield.geometry import compute_state_scores, get_state_summary

feature_dict = {
    'g_mag': g_mag, 'turbulence': turbulence,
    'knn_std_distance': tier0[:, 1], 'ridge_proximity': tier0[:, 5], ...
}
labels, scores, names = compute_state_scores(feature_dict)
summary = get_state_summary(labels, scores, names)
```

## Phase 2+5 Evaluation Results

```
CORRELATIONS WITH BOUNDARY DISTANCE:
  participation_ratio  : r=-0.394 (r=-0.529 on borderline!)
  spectral_entropy     : r=-0.384 (r=-0.496 on borderline)
  d_eff                : r=-0.312
  thermal_gradient     : r=+0.226 (r=+0.372 on borderline)
  turbulence_index     : r=-0.178

INCREMENTAL UTILITY:
  Tier-0 only:       R2=0.7703
  + Phase 1 (g_mag): R2=0.7730 (+0.35%)
  + Phase 2+5:       R2=0.7942 (+3.10%)
```

## Next Steps

1. ~~Run evaluation on Phase 2+5 features~~ DONE
2. Prune to features meeting acceptance criteria (ΔAUC >= 0.5%, stable across k)
3. Implement Phase 4 state mapping on retained features
4. If trajectory logging available: add Phase 3

## Hyperparameter Recommendations

From sensitivity analysis:
- k=75 performs better than k=50 for gradient magnitude
- Bandwidth multiplier c=1.2 slightly better than c=1.0
- Higher k values correlate with stronger signals

## Future Research: Recursive Self-Learning with Geometric Feedback

Not currently pursued, but worth revisiting. The idea: use geometric signals
(PR, SE, d_eff) as a feedback/regularization signal during recursive
self-learning, so the model learns to widen the narrow passages rather than
just detecting them after the fact.

**Potential approach:**
- PR/SE are SVD-based and differentiable — could serve as a regularization
  term or reward signal during training
- Novelty map signatures could guide curriculum (more training cycles on
  `terra_incognita` and `decision_boundary` regions)
- The v1.9 reasoning telemetry (PR tracking through steps) is a first step
  toward closing this loop

**Risks to consider:**
- Goodhart's Law: optimizing directly for high PR may inflate dimensionality
  without improving robustness — the metric stops being diagnostic once it
  becomes the optimization target
- Distribution shift: each self-learning iteration changes the embedding
  space, so geometric features from iteration N may not generalize to N+1
- Constrained geometry near boundaries may be structurally necessary for
  class separation — widening passages could hurt accuracy
