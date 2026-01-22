# Reproducible kNN-geometry uncertainty evaluation harness
# References: (Sun et al., 2022), (Bahri et al., 2021), (Wulz & Krispel, 2025)
# Dependencies: numpy, scipy, scikit-learn, pandas, matplotlib (faiss optional)

import numpy as np
import pandas as pd
import os
import argparse
from functools import partial
from itertools import product
from sklearn.datasets import make_moons, make_circles, load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.covariance import LedoitWolf
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
from sklearn.metrics import precision_recall_curve
from scipy.spatial.distance import cdist
from scipy.special import logsumexp
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

# ---- Utilities ----

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)


def fpr_at_tpr(y_true, score, target_tpr=0.95):
    fpr, tpr, thr = roc_curve(y_true, score)
    # Find smallest threshold index with tpr >= target_tpr
    idx = np.where(tpr >= target_tpr)[0]
    if len(idx) == 0:
        return 1.0
    return fpr[idx[0]]


def spearman_and_pearson(x, y):
    from scipy.stats import spearmanr, pearsonr
    sr = spearmanr(x, y).correlation
    pr = pearsonr(x, y)[0]
    return sr, pr

# ---- Data generation ----

def generate_gaussian_mixture(n_samples=2000, dim=2, n_classes=3, overlap=1.0, anisotropy=False, imbalance=None, random_state=0):
    rng = np.random.RandomState(random_state)
    samples_per = n_samples // n_classes
    X = []
    y = []
    centers = rng.randn(n_classes, dim) * 3.0
    for i in range(n_classes):
        cov = np.eye(dim) * (0.5 * overlap)
        if anisotropy:
            A = np.diag(1.0 + rng.rand(dim) * 3.0)
            cov = A @ cov @ A
        cnt = samples_per
        if imbalance and i < len(imbalance):
            cnt = int(n_samples * imbalance[i])
        Xi = rng.multivariate_normal(centers[i], cov, size=cnt)
        X.append(Xi)
        y.append(np.full(cnt, i))
    X = np.vstack(X)
    y = np.concatenate(y)
    perm = rng.permutation(len(X))
    return X[perm], y[perm]


def create_synthetic(dataset='moons', n_samples=2000, dim=2, **kwargs):
    if dataset == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=kwargs.get('noise', 0.2), random_state=kwargs.get('random_state', 0))
    elif dataset == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=kwargs.get('noise', 0.1), factor=0.5, random_state=kwargs.get('random_state', 0))
    else:
        X, y = generate_gaussian_mixture(n_samples=n_samples, dim=dim, **kwargs)
    return X, y

# ---- Embeddings and preproc ----

def build_embeddings(X_train, X_test, method='identity', n_components=None, whiten=True):
    if method == 'identity':
        return X_train.copy(), X_test.copy()
    elif method == 'pca':
        n_components = n_components or min(X_train.shape[1], 50)
        pca = PCA(n_components=n_components, whiten=whiten, random_state=0)
        Z_train = pca.fit_transform(X_train)
        Z_test = pca.transform(X_test)
        return Z_train, Z_test
    else:
        raise ValueError('Unknown embedding method')

# ---- Classifier ----

def train_classifier(Z_train, y_train, method='logreg'):
    if method == 'logreg':
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Z_train, y_train)
        probs = lambda X: clf.predict_proba(X)
        predict = lambda X: clf.predict(X)
        decision = None
    elif method == 'svc':
        clf = SVC(probability=True)
        clf.fit(Z_train, y_train)
        probs = lambda X: clf.predict_proba(X)
        predict = lambda X: clf.predict(X)
        decision = None
    else:
        raise ValueError('Unknown classifier')
    return clf, probs, predict, decision

# ---- Geometric features ----

def compute_knn_stats(Z_train, Z_test, k=20, metric='euclidean'):
    nn = NearestNeighbors(n_neighbors=k, metric=metric)
    nn.fit(Z_train)
    dists, idx = nn.kneighbors(Z_test, return_distance=True)
    # distances include the nearest neighbor (possibly zero if duplicates in train)
    mean_dist = np.mean(dists, axis=1)
    median_dist = np.median(dists, axis=1)
    min_dist = dists[:, 0]
    std_dist = np.std(dists, axis=1)
    kth_dist = dists[:, -1]
    return {
        'dists': dists,
        'idx': idx,
        'mean_dist': mean_dist,
        'median_dist': median_dist,
        'min_dist': min_dist,
        'std_dist': std_dist,
        'kth_dist': kth_dist
    }


def compute_lid(dists, k=None, eps=1e-12):
    # dists: array (n_samples, k) of sorted distances (ascending)
    if k is None:
        k = dists.shape[1]
    r_k = np.maximum(dists[:, -1], eps)
    # avoid log0
    vals = - (k - 1) / np.sum(np.log(np.maximum(dists[:, 1:], eps) / r_k[:, None]), axis=1)
    # if numerical issues, clip
    vals = np.clip(vals, -1e3, 1e3)
    return vals


def curvature_proxy(Z_train, Z_test, idx_neighbors, eps=1e-12):
    # curvature proxy: ratio of top eigenvalues of local covariance of neighbors
    n_test = Z_test.shape[0]
    curv = np.zeros(n_test)
    for i in range(n_test):
        neigh = Z_train[idx_neighbors[i]]
        cov = np.cov(neigh.T)
        # robust eigenvalues
        try:
            eig = np.linalg.eigvalsh(cov + eps * np.eye(cov.shape[0]))
            eig = np.sort(eig)[::-1]
            if len(eig) > 1:
                curv[i] = eig[0] / (eig[1] + eps)
            else:
                curv[i] = 0.0
        except Exception:
            curv[i] = 0.0
    return curv


def class_conditional_mahalanobis(Z_train, y_train, Z_test, shrinkage=True):
    classes = np.unique(y_train)
    lw = LedoitWolf().fit(Z_train)
    cov = lw.covariance_
    cov_inv = np.linalg.pinv(cov)
    means = {c: Z_train[y_train == c].mean(axis=0) for c in classes}
    dists = np.stack([np.sum((Z_test - means[c]) @ cov_inv * (Z_test - means[c]), axis=1) for c in classes], axis=1)
    min_dists = np.min(dists, axis=1)
    return np.sqrt(min_dists)

# ---- Baselines ----

def max_softmax_score(probs_fn, X):
    probs = probs_fn(X)
    return np.max(probs, axis=1)

def energy_score(logits):
    # logits: array-like of shape (n_samples, n_classes)
    # energy = -logsumexp(logits)
    return -logsumexp(logits, axis=1)

# ---- Evaluation harness ----

def evaluate_uncertainty(Z_train, y_train, Z_test, y_test, clf, probs_fn, predict_fn, k=20, eps=1e-8):
    # Compute k-NN features
    knn = compute_knn_stats(Z_train, Z_test, k=k)
    lid = compute_lid(knn['dists'], k=k)
    curv = curvature_proxy(Z_train, Z_test, knn['idx'])
    maha = class_conditional_mahalanobis(Z_train, y_train, Z_test)

    probs = probs_fn(Z_test)
    if probs is None:
        # fallback
        probs = np.zeros((Z_test.shape[0], len(np.unique(y_train))))
    msp = max_softmax_score(probs_fn, Z_test)
    # energy approximated from probs (logits not directly available for logreg)
    energy = -np.log(np.clip(np.sum(np.exp(np.log(np.clip(probs, 1e-12, 1.0))), axis=1), eps, None))

    pred = predict_fn(Z_test)
    is_error = (pred != y_test).astype(int)

    # proposed S = U * (min_dist + eps)
    U = knn['mean_dist']
    min_d = knn['min_dist']
    S = U * (min_d + eps)

    # build feature dict
    features = pd.DataFrame({
        'mean_dist': knn['mean_dist'],
        'median_dist': knn['median_dist'],
        'min_dist': knn['min_dist'],
        'std_dist': knn['std_dist'],
        'kth_dist': knn['kth_dist'],
        'lid': lid,
        'curv': curv,
        'maha': maha,
        'msp': msp,
        'energy': energy,
        'S': S,
        'is_error': is_error
    })
    return features

# ---- Boundary stratification ----

def boundary_strata(Z_train, y_train, Z_test, k=20):
    nn = NearestNeighbors(n_neighbors=k).fit(Z_train)
    _, idx = nn.kneighbors(Z_test)
    # compute neighbor-label purity
    purity = np.array([np.max(np.bincount(y_train[nb])) / k for nb in idx])
    return purity

# ---- Orchestrator: datasets, sweeps, results ----

def run_experiment(output_dir='results', random_state=0, debug=False):
    ensure_dir(output_dir)
    results = []

    # dataset configs
    datasets = [
        ('moons', dict(dataset='moons', n_samples=2000, noise=0.2)),
        ('circles', dict(dataset='circles', n_samples=2000, noise=0.05)),
        ('gmm', dict(dataset='gmm', n_samples=3000, dim=8, n_classes=4, overlap=0.8, anisotropy=True))
    ]

    embed_methods = ['identity', 'pca']
    ks = [5, 10, 20]
    classifier = 'logreg'
    sweeps = list(product(datasets, embed_methods, ks))

    for (dname, dcfg), embed_method, k in sweeps:
        # generate data
        if dcfg['dataset'] in ['moons', 'circles']:
            X, y = create_synthetic(dcfg['dataset'], n_samples=dcfg['n_samples'], noise=dcfg.get('noise', 0.1), random_state=random_state)
        else:
            X, y = create_synthetic('gmm', n_samples=dcfg.get('n_samples', 3000), dim=dcfg.get('dim', 8), n_classes=dcfg.get('n_classes', 3), overlap=dcfg.get('overlap', 1.0), anisotropy=dcfg.get('anisotropy', False), random_state=random_state)

        # train/test split and create an OOD set by holding out one class if available
        if len(np.unique(y)) > 2:
            # hold out last class as near-OOD
            holdout_class = np.unique(y)[-1]
            mask_ood = (y == holdout_class)
            mask_id = ~mask_ood
            X_id, y_id = X[mask_id], y[mask_id]
            X_ood, y_ood = X[mask_ood], y[mask_ood]
            # further split ID into train/test
            X_train, X_test, y_train, y_test = train_test_split(X_id, y_id, test_size=0.3, random_state=random_state, stratify=y_id)
            # OOD will be used as OOD test samples
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=random_state, stratify=y)
            X_ood, y_ood = make_circles(n_samples=500, noise=0.2, factor=0.1)[0], np.full(500, -1)

        # standardize
        scaler = StandardScaler().fit(X_train)
        X_train_s = scaler.transform(X_train)
        X_test_s = scaler.transform(X_test)
        X_ood_s = scaler.transform(X_ood)

        # embeddings
        Z_train, Z_test = build_embeddings(X_train_s, X_test_s, method=embed_method, n_components=min(50, X_train_s.shape[1]))
        _, Z_ood = build_embeddings(X_train_s, X_ood_s, method=embed_method, n_components=min(50, X_train_s.shape[1]))

        # train classifier on embeddings
        clf, probs_fn, predict_fn, _ = train_classifier(Z_train, y_train, method=classifier)

        # evaluate ID test
        feats_test = evaluate_uncertainty(Z_train, y_train, Z_test, y_test, clf, probs_fn, predict_fn, k=k)
        feats_ood = evaluate_uncertainty(Z_train, y_train, Z_ood, np.full(Z_ood.shape[0], -1), clf, probs_fn, predict_fn, k=k)

        # For error detection (is_error as label)
        # Evaluate feature ranking capability to rank errors higher
        metrics = {}
        for col in ['mean_dist', 'median_dist', 'min_dist', 'std_dist', 'kth_dist', 'lid', 'curv', 'maha', 'energy', 'S']:
            try:
                auc = roc_auc_score(feats_test['is_error'], feats_test[col])
                ap = average_precision_score(feats_test['is_error'], feats_test[col])
                metrics[f'{col}_auc'] = auc
                metrics[f'{col}_ap'] = ap
            except Exception:
                metrics[f'{col}_auc'] = np.nan
                metrics[f'{col}_ap'] = np.nan

        # For OOD detection: create binary labels (ID=0, OOD=1) and evaluate using e.g., min_dist or S or maha
        # Combine ID test and OOD
        combined_scores = {}
        X_comb = np.vstack([Z_test, Z_ood])
        y_comb = np.concatenate([np.zeros(Z_test.shape[0]), np.ones(Z_ood.shape[0])])

        # compute features on combined
        knn_comb = compute_knn_stats(Z_train, X_comb, k=k)
        lid_comb = compute_lid(knn_comb['dists'], k=k)
        maha_comb = class_conditional_mahalanobis(Z_train, y_train, X_comb)
        # msp
        probs_comb = probs_fn(X_comb)
        msp_comb = np.max(probs_comb, axis=1)
        # S combined
        U_comb = np.mean(knn_comb['dists'], axis=1)
        min_d_comb = knn_comb['dists'][:, 0]
        S_comb = U_comb * (min_d_comb + 1e-8)

        # compute AUROC & FPR @TPR95 for selection of scores
        for name, score in [('min_dist', min_d_comb), ('mean_dist', U_comb), ('maha', maha_comb), ('lid', lid_comb), ('msp', -msp_comb), ('energy', -energy_score(probs_comb)), ('S', S_comb)]:
            try:
                auc = roc_auc_score(y_comb, score)
                ap = average_precision_score(y_comb, score)
                fpr95 = fpr_at_tpr(y_comb, score, 0.95)
            except Exception:
                auc = ap = fpr95 = np.nan
            metrics[f'ood_{name}_auc'] = auc
            metrics[f'ood_{name}_ap'] = ap
            metrics[f'ood_{name}_fpr95'] = fpr95

        # boundary stratification analysis
        purity = boundary_strata(Z_train, y_train, Z_test, k=k)
        # define boundary mask
        boundary_mask = (purity <= 0.6)
        if np.sum(boundary_mask) >= 10:
            # compute AUC on boundary subset for 'S' and 'min_dist'
            try:
                metrics['boundary_S_auc'] = roc_auc_score(feats_test['is_error'][boundary_mask], feats_test['S'][boundary_mask])
                metrics['boundary_min_dist_auc'] = roc_auc_score(feats_test['is_error'][boundary_mask], feats_test['min_dist'][boundary_mask])
            except Exception:
                metrics['boundary_S_auc'] = metrics['boundary_min_dist_auc'] = np.nan
        else:
            metrics['boundary_S_auc'] = metrics['boundary_min_dist_auc'] = np.nan

        # correlation with actual error probability (for demonstration we use indicator as proxy)
        sr_S, pr_S = spearman_and_pearson(feats_test['S'], feats_test['is_error'])
        metrics['S_spearman'] = sr_S
        metrics['S_pearson'] = pr_S

        # collect meta
        metrics.update({'dataset': dname, 'embed': embed_method, 'k': k})
        results.append(metrics)

        if debug:
            print('Done', dname, embed_method, k)

    df = pd.DataFrame(results)
    outcsv = os.path.join(output_dir, 'knn_geometry_results.csv')
    df.to_csv(outcsv, index=False)
    print('Saved results to', outcsv)
    print(df.head())
    return df

# ---- CLI ----

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default='results')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()
    run_experiment(output_dir=args.out, debug=args.debug)
