# Geometric Safety Features for AI Embedding Spaces

**Topology-based detection of behavioral instability, backdoor poisoning, and reasoning dynamics in AI models**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18290279.svg)](https://doi.org/10.5281/zenodo.18290279)
[![License](https://img.shields.io/badge/license-MIT-blue)]()
[![Python](https://img.shields.io/badge/python-3.9+-blue)]()
[![Version](https://img.shields.io/badge/version-1.9.0-green)]()

---

## Key Discoveries

### 1. Constrained dimensionality predicts behavioral instability (v1.5)

Unsafe regions in AI embedding spaces are characterized by **low participation ratio** and **low spectral entropy** — "narrow passages" where variance is squeezed into few dimensions.

| Feature | Global r | Borderline r | Interpretation |
|---------|----------|--------------|----------------|
| `participation_ratio` | **-0.394** | **-0.529** | LOW = HIGH RISK |
| `spectral_entropy` | **-0.384** | **-0.496** | LOW = HIGH RISK |
| `knn_std_distance` | +0.286 | +0.399 | Tier-0 baseline |

### 2. Same features detect backdoor poisoning (v1.6)

Topology features that predict instability also identify poisoned training samples:

| Poisoning Strategy | Best Feature | ROC-AUC |
|-------------------|-------------|---------|
| Cluster | `participation_ratio` | **0.947** |
| Boundary | `d_eff` | **0.785** |
| Random | `local_gradient_magnitude` | 0.653 |

### 3. Poisoning creates "condensate-like" geometry (v1.7)

Physics-inspired analysis reveals poisoned data collapses to uniform geometry:

| Dataset | G Ratio | CV | Interpretation |
|---------|---------|-----|---------------|
| Clean | 0.505 | 0.326 | Natural variation |
| Cluster poison | **0.950** | **0.033** | Uniform constraint |

### 4. Geometric telemetry tracks reasoning dynamics (v1.9)

Participation ratio captures cognitive expansion and constraint during AI reasoning:

```
TROUGH (constraint) -> EXPANSION (breakthrough) -> DROP (new challenge) -> PEAK (resolution)
```

---

## Quick Start

```bash
pip install -r requirements.txt
```

### Safety Diagnostics
```python
from mirrorfield.geometry import compute_safety_diagnostics

# Compute all features + state mapping
diag = compute_safety_diagnostics(query_embeddings, reference_embeddings)

# Flag high-risk samples
high_risk = diag.get_high_risk_mask(threshold=0.25)
print(f"High risk: {high_risk.sum()}/{len(query_embeddings)}")

# Get all 14 features as matrix
features = diag.get_all_features()  # (N, 14)
```

### Real-Time Agent Monitoring (v1.8+)
```python
from mirrorfield.api import GeometricStateMonitor

monitor = GeometricStateMonitor(reference_embeddings)

# Query a single embedding
report = monitor.get_state(embedding)
print(report.predicted_state)  # 'coherent', 'uncertain', etc.
print(report.flags)            # ['low_pr', 'high_g_ratio'], etc.

# Decision helpers
if monitor.should_pause(report):
    agent.reflect()

advice = monitor.get_modulation_advice(report)
# {'action': 'reflect', 'confidence_adjustment': -0.2, 'suggestions': [...]}
```

---

## Features (14 total)

### Tier-0: k-NN Geometry
| Feature | Measures |
|---------|----------|
| `knn_mean_distance` | Average neighbor distance |
| `knn_std_distance` | Neighborhood dispersion |
| `knn_max_distance` | Furthest neighbor (sparsity) |
| `local_curvature` | Manifold anisotropy |
| `ridge_proximity` | Coefficient of variation |

### Phase 1: Flow
| Feature | Measures |
|---------|----------|
| `local_gradient_magnitude` | Density gradient strength |

### Phase 2: Weather Metaphors
| Feature | Measures |
|---------|----------|
| `turbulence_index` | Local mixing/disorder |
| `thermal_gradient` | Boundary-focused gradient |
| `vorticity` | Rotational tendency |

### Phase 5: Topology (Strongest Signals)
| Feature | Measures |
|---------|----------|
| `participation_ratio` | Effective dimensionality (PR = (Σλ)²/Σλ²) |
| `spectral_entropy` | Eigenvalue distribution spread |
| `d_eff` | Dimensions for 90% variance |

### Phase 4: Cognitive States
| State | Meaning | Risk |
|-------|---------|------|
| `uncertain` | Dispersed, boundary-adjacent | HIGH |
| `novel_territory` | Sparse, unfamiliar | HIGH |
| `constraint_pressure` | Multi-basin tension | MEDIUM |
| `searching` | Turbulent exploration | LOW |
| `confident` | Tight clustering | LOW |
| `coherent` | Stable attractor flow | LOW |

---

## The "Narrow Passage" Interpretation

Why does **low** dimensionality predict **high** risk?

Near decision boundaries, the embedding manifold is geometrically constrained. The boundary "squeezes" nearby representations, concentrating variance along boundary-parallel directions. Points in these narrow passages can absorb perturbations in some directions but are sensitive in others — small changes in the constrained directions cause behavioral flips.

```
HIGH participation_ratio → isotropic neighborhood → robust → SAFE
LOW participation_ratio  → anisotropic neighborhood → fragile → RISKY
```

---

## Repository Structure

```
geometric_safety_features/
├── mirrorfield/
│   ├── geometry/
│   │   ├── bundle.py                     # GeometryBundle API (Tier-0)
│   │   ├── features.py                   # 7 k-NN features
│   │   ├── advanced_features.py          # Atmospheric metrics, conformal abstention
│   │   ├── phase1_flow_features.py       # Gradient magnitude
│   │   ├── phase2_weather_features.py    # Weather + topology features
│   │   ├── phase3_trajectory_features.py # For generative models
│   │   ├── phase4_state_mapping.py       # Cognitive state classification
│   │   └── unified_pipeline.py           # compute_safety_diagnostics()
│   └── api/
│       └── state_monitor.py              # GeometricStateMonitor (v1.8)
├── experiments/
│   ├── track1_poison/                    # Backdoor detection (v1.6)
│   │   ├── create_poisoned_dataset.py    # Label-flip backdoor injection
│   │   ├── train_classifiers.py          # Clean vs poisoned models
│   │   ├── compute_geometry.py           # Geometric feature analysis
│   │   ├── detect_poison.py              # Detection with AUC evaluation
│   │   └── TRACK1_REPORT.md
│   ├── track2_physics/                   # Physics analogies (v1.7)
│   │   ├── lambda_run.py                 # G ratio uniformity analysis
│   │   ├── decoherence_chamber.py        # Multi-scale entropy evolution
│   │   └── TRACK2_REPORT.md
│   ├── track3_api/                       # State API demos (v1.8)
│   │   ├── live_detection_demo.py        # Real-time agent simulation
│   │   ├── parallax_integration.py       # Multi-AI pipeline integration
│   │   └── TRACK3_REPORT.md
│   ├── equalizer_test/                   # Reasoning telemetry (v1.9)
│   │   ├── reasoning_trace_logger.py     # Trace with geometric monitoring
│   │   ├── build_reference_corpus.py     # Reference corpus builder
│   │   ├── visualize_trace.py            # Trajectory visualization
│   │   └── EQUALIZER_TEST_REPORT.md
│   ├── boundary_sliced_evaluation.py     # Main evaluation
│   └── key_findings_analysis.py          # Discovery analysis
├── examples/
│   └── detection_rule_demo.py            # Simple detection rule
├── paper/
│   └── PAPER_DRAFT.md                    # Paper draft
└── docs/
    └── PHASE1_2_5_STATUS.md              # Implementation status
```

---

## Evaluation Results

### Borderline Amplification
Features become MORE predictive in boundary regions:

| Feature | Global r | Borderline r | Amplification |
|---------|----------|--------------|---------------|
| `participation_ratio` | -0.394 | -0.529 | **1.34x** |
| `ridge_proximity` | +0.215 | +0.361 | **1.68x** |
| `thermal_gradient` | +0.226 | +0.372 | **1.65x** |

### Incremental R² Improvement
```
Embeddings only:      R² = 0.741 (global), 0.233 (borderline)
+ Tier-0 features:    R² = 0.770 (+3.9%)
+ Phase 5 topology:   R² = 0.794 (+3.1% additional)
```

### Poison Detection (v1.6)
```
Cluster poisoning:   AUC = 0.947 (participation_ratio)
Boundary poisoning:  AUC = 0.785 (d_eff)
Random poisoning:    AUC = 0.653 (local_gradient_magnitude)
```

### Reasoning Telemetry (v1.9)
```
PR tracks cognitive dynamics:
  TROUGH (PR=8.01) at constraint -> PEAK (PR=9.82) at insight
  Insight signature: TROUGH -> EXPANSION -> DROP -> PEAK
```

---

## Citation

```bibtex
@software{coghlan2026geometric,
  author = {Coghlan, Dillan John},
  title = {Geometric Safety Features for AI Embedding Spaces},
  year = {2026},
  url = {https://github.com/DillanJC/geometric_safety_features},
  doi = {10.5281/zenodo.18290279}
}
```

---

## Requirements

- Python 3.9+
- NumPy, SciPy, scikit-learn
- Optional: PyTorch (GPU acceleration), FAISS (large-scale k-NN)

```bash
pip install numpy scipy scikit-learn matplotlib
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Experiment Reports

Detailed reports with full results for each track:

- [Track 1: Poison Detection](experiments/track1_poison/TRACK1_REPORT.md) — Geometric backdoor detection (AUC = 0.947)
- [Track 2: Physics Analogies](experiments/track2_physics/TRACK2_REPORT.md) — Lambda Run and Decoherence Chamber
- [Track 3: State API](experiments/track3_api/TRACK3_REPORT.md) — Real-time GeometricStateMonitor
- [Equalizer Test](experiments/equalizer_test/EQUALIZER_TEST_REPORT.md) — Reasoning telemetry proof-of-concept

---

## Acknowledgments

Conceptual development with DeepSeek. Literature grounding by Kosmos/EdisonScientific. Synthesis and implementation with Claude.
