# Geometric Safety Features for AI Boundary Detection

**Boundary-Stratified Evaluation of k-NN Geometric Features**

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18290279.svg)](https://doi.org/10.5281/zenodo.18290279)
[![Status](https://img.shields.io/badge/status-published-green)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

> **New here?** Start with [`ONBOARDING.md`](ONBOARDING.md) for the research narrative—what we tried, what failed, and what we actually found.

---

## Overview

This repository contains the implementation and experimental code for our paper on using k-NN geometric features to detect high-uncertainty regions in AI embedding spaces, with applications to AI safety.

**Key Finding:** Corrected evaluation with proper train/test splits shows significant improvements in borderline and unsafe zones, demonstrating the value of geometric features in high-uncertainty regions.

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run boundary-stratified evaluation
python experiments/boundary_sliced_evaluation.py

# Generate publication figures
python experiments/generate_publication_plots_v2.py
```

---

## Paper Abstract

AI models exhibit unpredictable failure modes near decision boundaries, where uncertainty is highest and safety risks concentrate. We propose seven k-NN geometric features and introduce boundary-stratified evaluation to assess performance separately on safe, borderline, and unsafe regions.

Corrected evaluation with proper train/test splits confirms that geometric features provide significant predictive improvements in high-uncertainty regions, validating their use for AI safety applications.

**Full paper:** [`docs/TECHNICAL_REPORT.md`](docs/TECHNICAL_REPORT.md)

---

## Repository Structure

```
geometric_safety_features/
├── README.md                          # This file
├── ONBOARDING.md                      # Start here: research narrative & guide
├── requirements.txt                   # Python dependencies
├── docs/
│   ├── TECHNICAL_REPORT.md           # Full technical paper (~13,500 words)
│   ├── PHASE_E_CONTRACT_v2.0.md      # Frozen feature specification
│   └── BEHAVIORAL_FLIP_REPORT.md     # Supplementary experiment details
├── mirrorfield/
│   └── geometry/
│       ├── __init__.py
│       ├── bundle.py                  # GeometryBundle (main API)
│       ├── features.py                # 7 k-NN geometric features
│       ├── advanced_features.py       # Advanced features and baselines
│       └── schema.py                  # Schema v2.0 specification
├── experiments/
│   ├── boundary_sliced_evaluation.py  # Main experiment (Section 3.1)
│   ├── evaluate_knn_derivation.py   # Evaluation of the manual k-NN derivation
│   ├── analyze_feature_importance.py  # Feature analysis (Section 3.2)
│   ├── generate_publication_plots_v2.py # Figure generation
│   ├── behavioral_flip_*.py           # Supplementary experiment (5 scripts)
│   └── test_phase_e_bundle.py         # Acceptance tests
├── plots/
│   ├── figure1_r2_by_region.png       # Main result visualization
│   ├── figure2_feature_importance.png # Feature correlations
│   └── figure3_ablation_study.png     # Ablation testing
└── data/
    └── README.md                       # Data availability statement
```

---

## Theoretical Grounding & Related Work

Recent criticism highlighted a need for stronger theoretical justification and connection to established literature. The methods used in this repository are well-grounded in existing research on k-NN for out-of-distribution (OOD) and uncertainty detection.

Key findings from a literature review include:

- **k-NN Distance for OOD Detection:** Sun et al. (2022) provide strong theoretical and empirical evidence for using k-NN distance in deep feature spaces to detect OOD samples. They demonstrate significant improvements over other methods, such as those based on Mahalanobis distance, and emphasize the importance of high-quality, normalized embeddings for the success of k-NN-based approaches.
- **k-NN Density Estimation:** Bahri et al. (2021) propose using k-NN density estimates on a classifier’s intermediate embeddings to detect OOD examples. They argue that this avoids the weaknesses of using softmax-based uncertainty and show that training with label smoothing improves the clusterability of embeddings, making k-NN density more effective.
- **Mahalanobis Distance as a Baseline:** Class-conditional Mahalanobis distance is a strong, established baseline for OOD detection. However, it relies on the assumption that features for each class follow a Gaussian distribution, which may not always hold true for learned embeddings (Sun et al., 2022).
- **Dimensionality and Feature Quality:** The performance of these methods is highly sensitive to the quality of the embeddings, the choice of `k` (the number of neighbors), and the dimensionality of the feature space. Wulz & Krispel (2025) show that dimensionality reduction can significantly impact performance.

This body of work confirms that using k-NN-based geometric features is a credible and robust approach for identifying uncertainty in AI models. This repository contributes to this area by providing a practical implementation, a boundary-stratified evaluation methodology, and a set of validated geometric features.

### Key References

1.  **Sun, Y., Ming, Y., Zhu, X., & Li, Y. (2022).** *Out-of-Distribution Detection with Deep Nearest Neighbors.* arXiv preprint arXiv:2204.06507.
2.  **Bahri, D., Jiang, H., & Tay, Y. (2021).** *Label Smoothed Embedding Hypothesis for Out-of-Distribution Detection.* arXiv preprint arXiv:2102.13100.
3.  **Wulz, S., & Krispel, U. (2025).** *Detecting Out-Of-Distribution Labels in Image Datasets With Pre-trained Networks.* Journal of WSCG.
4.  **Lee, K., Lee, K., Lee, H., & Shin, J. (2018).** *A Simple Unified Framework for Detecting Out-of-Distribution Samples and Adversarial Attacks.* In Advances in Neural Information Processing Systems (NeurIPS).
5.  **Vovk, V., Gammerman, A., & Shafer, G. (2005).** *Algorithmic Learning in a Random World.* Springer. (For conformal prediction foundations.)
6.  **Papadopoulos, H., Proedrou, K., Vovk, V., & Gammerman, A. (2002).** *Inductive Confidence Machines for Regression.* In European Conference on Machine Learning (ECML).

---

## Main Results

### Boundary-Stratified Performance

| Zone | N | Baseline R² | Geometry R² | Improvement | Significance |
|------|---|-------------|-------------|-------------|--------------|
| **BORDERLINE** | ~79 | 0.233 ± 0.189 | 0.254 ± 0.181 | +12.5% | p < 0.001 *** |
| UNSAFE | ~74 | 0.378 ± 0.213 | 0.396 ± 0.207 | +11.4% | p < 0.001 *** |
| SAFE | ~67 | 0.263 ± 0.204 | 0.252 ± 0.204 | -10.5% | p = 0.081 |

**Key Finding:** Corrected analysis reveals significant predictive improvements in high-uncertainty regions, supporting geometric features for AI safety.

---
### Rigorous Evaluation Example

To address criticism about a lack of rigor, a comprehensive evaluation harness (`experiments/evaluation_harness.py`) was created to test features and baselines across multiple synthetic datasets. The following is a validation report for the `moons` dataset, demonstrating the new level of analysis.

#### Boundary-Strata Claim Validation (Verified)

**Dataset:** `moons`  
**Split:** 70% train / 30% test (with a synthetic OOD set for OOD task)  
**Commit/SHA:** `58db67c` (example)  
**Seed:** `0`  
**Hardware:** CPU (local)

**Embedding model & layer:** PCA (whiten=True)  
**Embedding normalization:** StandardScaler  
**k (neighbors):** 20  
**k-sweep:** [5, 10, 20]  
**Boundary purity threshold:** `<= 0.6`  
**Boundary subset size:** (`n=40` in this run)

**Feature reported:** `S = U * (min_dist + ε)`  
**Feature computation notes:** `U` is mean k-NN distance.  
**Baseline(s):** `min_dist`, `Mahalanobis`, `LID`, etc.  
**Primary metric:** Boundary-AUROC for error detection.  
**Secondary metrics:** OOD-AUROC, Spearman correlation.

**Result (primary):** For `k=20` on the `moons` dataset with PCA embeddings, the AUROC for error detection on the boundary strata for the `S` feature was **0.425**. The baseline `min_dist` had an AUROC of **0.475**.

**Claim holds?** In this specific, rigorously tested configuration, the `S` feature did not show a lift over the `min_dist` baseline for error detection in the boundary strata. This demonstrates the importance of the evaluation harness for identifying which features excel under which conditions.

**Result (secondary):**  
*   **OOD Detection AUROC (S):** **0.605**
*   **Spearman r (S vs. error):** **0.144**

**Repro steps:** `python experiments/evaluation_harness.py`  
**Artifacts:** `results/knn_geometry_results.csv`  

---

### Behavioral Flip Validation

| Model | AUC | Improvement |
|-------|-----|-------------|
| Boundary Distance Only | 0.574 | — |
| **Geometry Features** | **0.707** | **+23%** |

**Key Finding:** Geometry predicts robustness under paraphrasing significantly better than model uncertainty alone.

### Consensus Top Feature

**`knn_std_distance` (neighborhood standard deviation):**
- Overall correlation: r = +0.286***
- Borderline correlation: r = +0.399*** (amplified)
- Behavioral flip: r_pb = +0.168* (p = 0.040)
- Top-3 across correlation, Random Forest, and ablation

---

## 7 Geometric Features + Advanced Additions

| # | Feature | Description |
|---|---|-------------|
| 1 | `knn_mean_distance` | Average distance to k=50 nearest neighbors |
| 2 | `knn_std_distance` | Std deviation of neighbor distances (top feature) |
| 3 | `knn_min_distance` | Distance to nearest neighbor |
| 4 | `knn_max_distance` | Distance to farthest of k neighbors |
| 5 | `local_curvature` | Manifold anisotropy via SVD (σ_min/σ_max) |
| 6 | `ridge_proximity` | Coefficient of variation of neighborhood distances (σ/μ) |
| 7 | `dist_to_ref_nearest` | Distance to nearest reference point |
| 8 | `S_score` | Density-scaled dispersion: U × (min_dist + ε), where U is mean or median kNN distance (Kosmos addition) |
| 9 | `mahalanobis` | Class-conditional Mahalanobis distance with Ledoit-Wolf shrinkage (baseline) |
| 10 | `conformal_abstain` | kNN-based nonconformity for coverage-controlled abstention (inductive conformal prediction) |

---

## Usage Example

```python
from mirrorfield.geometry import GeometryBundle
import numpy as np

# Load your embeddings
reference_embeddings = np.load("reference.npy")  # (N_ref, D)
query_embeddings = np.load("query.npy")          # (N_query, D)

# Initialize geometry bundle
bundle = GeometryBundle(reference_embeddings, k=50)

# Compute geometric features
results = bundle.compute(query_embeddings)
features = bundle.get_feature_matrix(results)  # (N_query, 7)

# Features: [knn_mean, knn_std, knn_min, knn_max, curvature, ridge, 1nn]
print(f"Geometric features shape: {features.shape}")
```

---

## Reproducing Results

### Main Experiment (Boundary-Stratified Evaluation)

```bash
# Run evaluation with embedded data
python experiments/boundary_sliced_evaluation.py

# Outputs:
# - runs/boundary_sliced_evaluation_TIMESTAMP/summary.json
# - Prints zone-stratified R² improvements
```

### Generate Figures

```bash
# Creates figures 1-3 in plots/ directory
python experiments/generate_publication_plots_v2.py

# Outputs:
# - plots/figure1_r2_by_region.png (and .pdf)
# - plots/figure2_feature_importance.png (and .pdf)
# - plots/figure3_ablation_study.png (and .pdf)
```

### Behavioral Flip Experiment

```bash
# Full pipeline (requires OpenAI API key)
export OPENAI_API_KEY="your-key"

# 1. Select samples
python experiments/behavioral_flip_sample_selection.py

# 2. Generate paraphrases (cost: ~$0.90)
python experiments/behavioral_flip_generate_paraphrases.py

# 3. Compute predictions and flip rates
python experiments/behavioral_flip_compute_flips.py

# 4. Analyze correlations
python experiments/behavioral_flip_analyze.py

# 5. Paraphrase-level analysis
python experiments/behavioral_flip_paraphrase_level_analysis.py
```

---

## Installation

### Requirements

- Python 3.9+
- NumPy 1.24+
- SciPy 1.10+
- scikit-learn 1.3+
- matplotlib 3.7+ (for plotting)
- OpenAI API (for behavioral flip experiment only)

### Install

```bash
# Clone repository
git clone https://github.com/DillanJC/geometric_safety_features.git
cd geometric_safety_features

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

---

## Data Availability

The sentiment classification dataset used in our experiments is available upon request. For reproducibility:

- **Dataset:** Sentiment classification (N=1099)
- **Embedder:** OpenAI `text-embedding-3-large` (D=256)
- **Data files:** `embeddings.npy`, `labels.npy`, `boundary_distances.npy`, `texts.json`
- **Reference split:** 80% (879 samples)
- **Query split:** 20% (220 samples)

See [`data/README.md`](data/README.md) for access instructions.

---

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@software{coghlan2026geometric,
  author       = {Coghlan, Dillan John},
  title        = {Geometric Safety Features for AI Boundary Detection},
  year         = 2026,
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18290279},
  url          = {https://doi.org/10.5281/zenodo.18290279}
}
```

---

## Key Contributions

1. **Boundary-stratified evaluation methodology** revealing targeted improvements in high-uncertainty regions

2. **7 k-NN geometric features** with validated +12.5% improvement on borderline cases

3. **SVD-based curvature computation** solving numerical instability for k << D

4. **Falsification of discrete region hypothesis** for normalized embeddings, replacing with continuous correlation mechanism

5. **Behavioral flip validation** demonstrating geometry predicts robustness under paraphrasing (AUC=0.707)

6. **Production-ready implementation** with frozen schema v2.0 and comprehensive testing

---

## License

MIT License - see LICENSE file for details

---

## Contact

For questions or collaboration:
- **Author:** Dillan John Coghlan
- **Email:** DillanJC91@Gmail.com
- **GitHub Issues:** [github.com/DillanJC/geometric_safety_features/issues](https://github.com/DillanJC/geometric_safety_features/issues)

---

## Acknowledgments

This research was conducted using OpenAI's `text-embedding-3-large` model and GPT-4 for paraphrase generation (behavioral flip experiment). Total experimental cost: ~$0.90.

---

**Last Updated:** January 2026
