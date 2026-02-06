"""
Core uncertainty computation for LLM outputs.

Adapts geometric/topological methods from PR-max research to token-level
log-probabilities and embedding vectors. Pure numpy/scipy — no torch dependency.
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform

from mirrorfield.geometry.phase2_weather_features import (
    compute_participation_ratio,
    compute_spectral_entropy,
)


# ---------------------------------------------------------------------------
# Token-level metrics
# ---------------------------------------------------------------------------

def compute_token_margins(top_logprobs: list[dict]) -> np.ndarray:
    """Compute per-token margin (top1 - top2 log-prob).

    Args:
        top_logprobs: List of dicts mapping token -> log-prob for each position.

    Returns:
        Array of margins (>= 0). Positions with fewer than 2 alternatives
        get margin = inf (maximally confident).
    """
    margins = np.full(len(top_logprobs), np.inf)
    for i, lp in enumerate(top_logprobs):
        if lp is None or len(lp) < 2:
            continue
        sorted_vals = sorted(lp.values(), reverse=True)
        # log-probs are negative; top1 is least negative, so margin >= 0
        margins[i] = sorted_vals[0] - sorted_vals[1]
    return margins


def compute_token_entropies(top_logprobs: list[dict]) -> np.ndarray:
    """Compute Shannon entropy at each token position from top-k probs.

    The top-k log-probs are converted to probabilities and renormalized
    so they sum to 1 before computing entropy.
    """
    entropies = np.zeros(len(top_logprobs))
    for i, lp in enumerate(top_logprobs):
        if lp is None or len(lp) == 0:
            continue
        logp = np.array(list(lp.values()), dtype=np.float64)
        probs = np.exp(logp)
        total = probs.sum()
        if total <= 0:
            continue
        probs = probs / total  # renormalize
        probs = probs[probs > 1e-12]
        entropies[i] = -np.sum(probs * np.log(probs))
    return entropies


def compute_sequence_pr(top_logprobs: list[dict]) -> float:
    """Participation ratio of the token x top-k log-prob matrix.

    Stacks top-k log-probs at each position into an (n_tokens, k) matrix
    and computes the participation ratio via SVD. High PR means the model
    spreads probability mass across many alternatives at many positions
    (= more uncertain). Low PR means a few dominant singular directions
    (= more certain).
    """
    if not top_logprobs:
        return 1.0

    # Determine the common k (max number of alternatives seen)
    k = max((len(lp) for lp in top_logprobs if lp), default=0)
    if k == 0:
        return 1.0

    n = len(top_logprobs)
    mat = np.full((n, k), -30.0)  # fill with very low log-prob
    for i, lp in enumerate(top_logprobs):
        if lp is None:
            continue
        vals = sorted(lp.values(), reverse=True)
        for j, v in enumerate(vals[:k]):
            mat[i, j] = v

    # Center columns
    mat = mat - mat.mean(axis=0, keepdims=True)

    try:
        _, S, _ = np.linalg.svd(mat, full_matrices=False)
        S_sq = S ** 2
        denom = np.sum(S_sq ** 2)
        if denom < 1e-15:
            return 1.0
        pr = (S_sq.sum() ** 2) / denom
        return float(pr)
    except np.linalg.LinAlgError:
        return 1.0


def compute_boundary_ratio(margins: np.ndarray, threshold: float = 0.5) -> float:
    """Fraction of tokens whose margin is below *threshold*."""
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        return 0.0
    return float(np.mean(finite < threshold))


# ---------------------------------------------------------------------------
# Embedding-level metrics
# ---------------------------------------------------------------------------

def compute_embedding_pr(embeddings: np.ndarray, k: int = 20) -> dict:
    """Geometric analysis of embedding vectors (numpy-only).

    Computes participation ratio, spectral entropy, effective dimensionality,
    and optionally correlation dimension.

    Args:
        embeddings: (N, D) array.
        k: Neighbors for local SVD (capped at N-1).

    Returns:
        Dict with pr_mean, pr_min, se_mean, g_ratio, effective_dim,
        and correlation_dim (if N > 50).
    """
    embeddings = np.asarray(embeddings, dtype=np.float64)
    if embeddings.ndim == 1:
        embeddings = embeddings.reshape(1, -1)
    N, D = embeddings.shape

    if N < 3:
        return {
            "pr_mean": 1.0, "pr_min": 1.0, "se_mean": 0.0,
            "g_ratio": 1.0, "effective_dim": 1.0,
        }

    k_use = min(k, N - 1, D)

    # Pairwise distances for neighbor lookup
    dists = squareform(pdist(embeddings))

    pr_vals = np.zeros(N)
    se_vals = np.zeros(N)

    for i in range(N):
        idx = np.argsort(dists[i])[1:k_use + 1]  # exclude self
        neighbors = embeddings[idx]
        centered = neighbors - neighbors.mean(axis=0)
        try:
            _, S, _ = np.linalg.svd(centered, full_matrices=False)
            S_sq = S ** 2
            pr_vals[i] = compute_participation_ratio(S_sq)
            se_vals[i] = compute_spectral_entropy(S_sq)
        except np.linalg.LinAlgError:
            pr_vals[i] = np.nan
            se_vals[i] = np.nan

    pr_valid = pr_vals[~np.isnan(pr_vals)]
    se_valid = se_vals[~np.isnan(se_vals)]
    pr_mean = float(np.mean(pr_valid)) if len(pr_valid) else 1.0
    pr_min = float(np.min(pr_valid)) if len(pr_valid) else 1.0
    se_mean = float(np.mean(se_valid)) if len(se_valid) else 0.0
    g_ratio = pr_min / pr_mean if pr_mean > 0 else 0.0

    # Global SVD for effective dimensionality
    centered_all = embeddings - embeddings.mean(axis=0)
    try:
        _, S_all, _ = np.linalg.svd(centered_all, full_matrices=False)
        S_sq_all = S_all ** 2
        eff_dim = float(compute_participation_ratio(S_sq_all))
    except np.linalg.LinAlgError:
        eff_dim = 1.0

    result = {
        "pr_mean": round(pr_mean, 4),
        "pr_min": round(pr_min, 4),
        "se_mean": round(se_mean, 4),
        "g_ratio": round(g_ratio, 4),
        "effective_dim": round(eff_dim, 4),
    }

    # Correlation dimension estimate (Grassberger-Procaccia) for larger sets
    if N > 50:
        result["correlation_dim"] = _estimate_correlation_dim(dists)

    return result


def _estimate_correlation_dim(dist_matrix: np.ndarray) -> float:
    """Grassberger-Procaccia correlation dimension estimate."""
    N = dist_matrix.shape[0]
    # Upper triangle distances
    triu_idx = np.triu_indices(N, k=1)
    all_dists = dist_matrix[triu_idx]
    all_dists = all_dists[all_dists > 0]
    if len(all_dists) < 10:
        return 0.0

    r_min = np.percentile(all_dists, 5)
    r_max = np.percentile(all_dists, 50)
    if r_min <= 0 or r_max <= r_min:
        return 0.0

    radii = np.geomspace(r_min, r_max, 20)
    n_pairs = len(all_dists)
    counts = np.array([np.sum(all_dists < r) / n_pairs for r in radii])
    counts = counts[counts > 0]
    if len(counts) < 5:
        return 0.0

    log_r = np.log(radii[:len(counts)])
    log_c = np.log(counts)
    # Linear fit in log-log space
    coeffs = np.polyfit(log_r, log_c, 1)
    return round(float(coeffs[0]), 4)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

def compute_confidence_score(
    margins: np.ndarray | None = None,
    entropies: np.ndarray | None = None,
    boundary_ratio: float | None = None,
) -> float:
    """Weighted combination of uncertainty signals, normalized to [0, 1].

    Higher = more confident.
    """
    components = []
    weights = []

    if margins is not None:
        finite = margins[np.isfinite(margins)]
        if len(finite) > 0:
            # Mean margin: higher margin = more confident.
            # Typical margins in [0, ~5]; sigmoid-like normalization.
            mean_m = float(np.mean(finite))
            margin_score = min(mean_m / 3.0, 1.0)
            components.append(margin_score)
            weights.append(0.4)

    if entropies is not None and len(entropies) > 0:
        mean_e = float(np.mean(entropies))
        # Low entropy = high confidence. Typical range [0, ~3].
        entropy_score = max(1.0 - mean_e / 2.5, 0.0)
        components.append(entropy_score)
        weights.append(0.35)

    if boundary_ratio is not None:
        # Low boundary ratio = high confidence.
        br_score = 1.0 - boundary_ratio
        components.append(br_score)
        weights.append(0.25)

    if not components:
        return 0.5  # no data → neutral

    weights = np.array(weights)
    weights /= weights.sum()
    score = float(np.dot(weights, components))
    return round(np.clip(score, 0.0, 1.0), 4)


def classify_confidence(score: float) -> str:
    """Map a 0-1 confidence score to a human-readable label."""
    if score >= 0.8:
        return "high"
    if score >= 0.6:
        return "moderate"
    if score >= 0.3:
        return "low"
    return "very_low"


def find_uncertain_spans(
    tokens: list[str],
    margins: np.ndarray,
    threshold: float = 0.5,
) -> list[dict]:
    """Identify contiguous spans of low-margin tokens.

    Returns a list of dicts with keys: start, end (token indices),
    char_start, char_end (character offsets into the joined token string),
    tokens, text, and min_margin.
    """
    # Pre-compute character offset of each token in the joined string
    char_offsets: list[int] = []
    pos = 0
    for tok in tokens:
        char_offsets.append(pos)
        pos += len(tok)

    spans: list[dict] = []
    in_span = False
    start = 0
    span_tokens: list[str] = []
    span_min = float("inf")

    for i, (tok, m) in enumerate(zip(tokens, margins)):
        low = np.isfinite(m) and m < threshold
        if low:
            if not in_span:
                start = i
                span_tokens = []
                span_min = float("inf")
                in_span = True
            span_tokens.append(tok)
            span_min = min(span_min, float(m))
        else:
            if in_span:
                text = "".join(span_tokens)
                margin_rounded = round(span_min, 4)
                spans.append({
                    "start": start,
                    "end": i,
                    "char_start": char_offsets[start],
                    "char_end": char_offsets[start] + len(text),
                    "tokens": span_tokens,
                    "text": text,
                    "min_margin": margin_rounded,
                    "display": _span_severity(span_min),
                })
                in_span = False
    # Close trailing span
    if in_span:
        text = "".join(span_tokens)
        margin_rounded = round(span_min, 4)
        spans.append({
            "start": start,
            "end": len(tokens),
            "char_start": char_offsets[start],
            "char_end": char_offsets[start] + len(text),
            "tokens": span_tokens,
            "text": text,
            "min_margin": margin_rounded,
            "display": _span_severity(span_min),
        })
    return spans


def _span_severity(min_margin: float) -> dict:
    """Map a span's minimum margin to a display hint for frontends."""
    if min_margin < 0.1:
        return {"severity": "critical", "color": "#e53e3e"}
    if min_margin < 0.25:
        return {"severity": "high", "color": "#dd6b20"}
    return {"severity": "moderate", "color": "#d69e2e"}


def compute_self_consistency(texts: list[str], n: int = 2) -> dict:
    """Measure agreement between multiple response texts using n-gram overlap.

    Returns pairwise Jaccard similarities and an aggregate agreement score.
    A high agreement score with high confidence = trustworthy.
    High confidence but low agreement = contradictory answers, be wary.
    """
    if len(texts) < 2:
        return {"agreement": 1.0, "pairwise": [], "n": n}

    def _ngrams(text: str, n: int) -> set[tuple[str, ...]]:
        words = text.lower().split()
        if len(words) < n:
            return {tuple(words)}
        return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}

    sets = [_ngrams(t, n) for t in texts]
    pairwise: list[dict] = []
    scores: list[float] = []

    for i in range(len(sets)):
        for j in range(i + 1, len(sets)):
            union = len(sets[i] | sets[j])
            if union == 0:
                sim = 1.0
            else:
                sim = len(sets[i] & sets[j]) / union
            sim = round(sim, 4)
            pairwise.append({"i": i, "j": j, "similarity": sim})
            scores.append(sim)

    return {
        "agreement": round(float(np.mean(scores)), 4),
        "pairwise": pairwise,
        "n": n,
    }


def generate_explanation(metrics: dict) -> str:
    """Produce a natural-language explanation of the uncertainty metrics."""
    parts: list[str] = []
    label = metrics.get("confidence_label", "unknown")
    score = metrics.get("confidence_score", None)

    if score is not None:
        parts.append(f"Overall confidence is {label} (score {score:.2f}).")

    br = metrics.get("boundary_ratio")
    if br is not None:
        pct = br * 100
        if pct > 30:
            parts.append(
                f"{pct:.0f}% of tokens are near the decision boundary, "
                "indicating significant uncertainty in word choices."
            )
        elif pct > 10:
            parts.append(
                f"{pct:.0f}% of tokens are near the decision boundary."
            )

    seq_pr = metrics.get("sequence_pr")
    if seq_pr is not None and seq_pr > 3.0:
        parts.append(
            f"The sequence participation ratio is {seq_pr:.1f}, suggesting "
            "the model considered many alternatives across positions."
        )

    n_uncertain = metrics.get("uncertain_span_count", 0)
    if n_uncertain > 0:
        parts.append(
            f"There are {n_uncertain} uncertain span(s) in the output."
        )

    if not parts:
        parts.append("Insufficient data for a detailed explanation.")

    return " ".join(parts)


# ---------------------------------------------------------------------------
# Novelty map — reinterprets uncertainty as exploration gradient
# ---------------------------------------------------------------------------

# The four epistemic signatures:
#
#   well_trodden       — high margin, low entropy.  Model is sure. Known ground.
#   decision_boundary  — low margin, low entropy.   Two competing frameworks.
#                        The question is well-formed enough to have rivals.
#   terra_incognita    — low margin, high entropy.   Probability smeared across
#                        many options. The model is extrapolating.
#   framework_collision — high confidence per-sample but low self-consistency.
#                        Multiple coherent reasoning paths reach different
#                        conclusions. The hallmark of an open question.

_NOVELTY_SIGNATURES = {
    "well_trodden": {
        "label": "Well-trodden",
        "description": "Established pattern with clear consensus.",
        "action": "Proceed — this is known ground.",
    },
    "decision_boundary": {
        "label": "Decision boundary",
        "description": (
            "The model is torn between a small number of specific alternatives. "
            "This is where two established frameworks collide — a productive "
            "frontier where the question is well-formed enough to have rivals."
        ),
        "action": "Explore both sides. The tension itself is informative.",
    },
    "terra_incognita": {
        "label": "Terra incognita",
        "description": (
            "Probability is smeared across many alternatives. The model is "
            "extrapolating beyond its training distribution. This could be "
            "noise, or it could be genuinely novel territory."
        ),
        "action": (
            "I'm extrapolating here. Treat this as a hypothesis, not a fact. "
            "Consider reframing the question or seeking external sources."
        ),
    },
    "framework_collision": {
        "label": "Framework collision",
        "description": (
            "Multiple independent reasoning paths each arrive confidently at "
            "different conclusions. This is the signature of an unresolved "
            "question where coherent but incompatible frameworks coexist."
        ),
        "action": (
            "This is genuinely contested. Surface the competing frameworks "
            "rather than picking one. The disagreement is the insight."
        ),
    },
}


def classify_novelty_signature(
    margin: float,
    entropy: float,
    consistency: float | None = None,
    margin_threshold: float = 0.5,
    entropy_threshold: float = 1.0,
    consistency_threshold: float = 0.3,
) -> str:
    """Classify a single position or span into one of four novelty signatures.

    Args:
        margin: Token margin (top1 - top2 log-prob). Low = uncertain.
        entropy: Shannon entropy of top-k distribution. High = spread.
        consistency: Self-consistency score (0-1). Low = disagreement.
            Only used if provided (requires multiple samples).
        margin_threshold: Below this margin = uncertain.
        entropy_threshold: Above this entropy = spread.
        consistency_threshold: Below this agreement = collision.

    Returns:
        One of: "well_trodden", "decision_boundary", "terra_incognita",
        "framework_collision".
    """
    low_margin = margin < margin_threshold
    high_entropy = entropy > entropy_threshold

    # Framework collision takes priority — it requires consistency data
    if consistency is not None and consistency < consistency_threshold:
        if not low_margin:
            # Confident but disagreeing — the most interesting signal
            return "framework_collision"

    if not low_margin and not high_entropy:
        return "well_trodden"
    if low_margin and not high_entropy:
        return "decision_boundary"
    # low margin + high entropy, or high margin + high entropy
    return "terra_incognita"


def compute_exploration_gradient(
    margins: np.ndarray,
    entropies: np.ndarray,
) -> float:
    """Overall exploration gradient from 0 (well-trodden) to 1 (uncharted).

    Combines margin and entropy signals into a single scalar.
    High gradient = the model is operating far from its training distribution.
    """
    finite = margins[np.isfinite(margins)]
    if len(finite) == 0:
        return 0.0

    # Margin component: low margin → high gradient
    mean_margin = float(np.mean(finite))
    margin_gradient = max(1.0 - mean_margin / 2.0, 0.0)

    # Entropy component: high entropy → high gradient
    mean_entropy = float(np.mean(entropies))
    entropy_gradient = min(mean_entropy / 2.0, 1.0)

    # Boundary density: what fraction of tokens are uncertain
    boundary_frac = float(np.mean(finite < 0.5))

    # Weighted combination
    gradient = 0.35 * margin_gradient + 0.35 * entropy_gradient + 0.3 * boundary_frac
    return round(float(np.clip(gradient, 0.0, 1.0)), 4)


def build_novelty_map(
    tokens: list[str],
    margins: np.ndarray,
    entropies: np.ndarray,
    consistency: float | None = None,
) -> dict:
    """Build a full novelty map from token-level signals.

    Returns a dict with:
      - exploration_gradient: 0 (known) to 1 (uncharted)
      - terrain_label: overall terrain classification
      - spans: list of novelty-classified spans (like uncertain_spans but
        with signature, description, and recommended action)
      - signature_counts: how many spans of each type
      - interpretation: natural-language summary
    """
    gradient = compute_exploration_gradient(margins, entropies)

    # Classify each token
    per_token_sigs: list[str] = []
    for i in range(len(tokens)):
        m = float(margins[i]) if np.isfinite(margins[i]) else 5.0
        e = float(entropies[i]) if i < len(entropies) else 0.0
        sig = classify_novelty_signature(m, e, consistency)
        per_token_sigs.append(sig)

    # Build contiguous spans of non-well-trodden territory
    char_offsets: list[int] = []
    pos = 0
    for tok in tokens:
        char_offsets.append(pos)
        pos += len(tok)

    spans: list[dict] = []
    in_span = False
    start = 0
    span_tokens: list[str] = []
    span_sig = "well_trodden"
    span_min_margin = float("inf")
    span_max_entropy = 0.0

    for i, (tok, sig) in enumerate(zip(tokens, per_token_sigs)):
        interesting = sig != "well_trodden"
        if interesting:
            if not in_span:
                start = i
                span_tokens = []
                span_sig = sig
                span_min_margin = float("inf")
                span_max_entropy = 0.0
                in_span = True
            span_tokens.append(tok)
            m = float(margins[i]) if np.isfinite(margins[i]) else 5.0
            e = float(entropies[i]) if i < len(entropies) else 0.0
            span_min_margin = min(span_min_margin, m)
            span_max_entropy = max(span_max_entropy, e)
            # Upgrade span signature to the most novel type seen
            if _sig_rank(sig) > _sig_rank(span_sig):
                span_sig = sig
        else:
            if in_span:
                text = "".join(span_tokens)
                info = _NOVELTY_SIGNATURES[span_sig]
                spans.append({
                    "start": start,
                    "end": i,
                    "char_start": char_offsets[start],
                    "char_end": char_offsets[start] + len(text),
                    "text": text,
                    "signature": span_sig,
                    "label": info["label"],
                    "description": info["description"],
                    "action": info["action"],
                    "min_margin": round(span_min_margin, 4),
                    "max_entropy": round(span_max_entropy, 4),
                })
                in_span = False

    if in_span:
        text = "".join(span_tokens)
        info = _NOVELTY_SIGNATURES[span_sig]
        spans.append({
            "start": start,
            "end": len(tokens),
            "char_start": char_offsets[start],
            "char_end": char_offsets[start] + len(text),
            "text": text,
            "signature": span_sig,
            "label": info["label"],
            "description": info["description"],
            "action": info["action"],
            "min_margin": round(span_min_margin, 4),
            "max_entropy": round(span_max_entropy, 4),
        })

    # Signature counts
    sig_counts: dict[str, int] = {}
    for s in per_token_sigs:
        sig_counts[s] = sig_counts.get(s, 0) + 1

    # Overall terrain
    if gradient < 0.2:
        terrain = "well_trodden"
    elif gradient < 0.45:
        terrain = "frontier"
    elif gradient < 0.7:
        terrain = "uncharted"
    else:
        terrain = "deep_unknown"

    interpretation = _novelty_interpretation(
        gradient, terrain, spans, sig_counts, consistency
    )

    return {
        "exploration_gradient": gradient,
        "terrain": terrain,
        "spans": spans,
        "signature_counts": sig_counts,
        "interpretation": interpretation,
    }


def _sig_rank(sig: str) -> int:
    """Rank signatures by novelty level for span upgrading."""
    return {
        "well_trodden": 0,
        "decision_boundary": 1,
        "terra_incognita": 2,
        "framework_collision": 3,
    }.get(sig, 0)


def _novelty_interpretation(
    gradient: float,
    terrain: str,
    spans: list[dict],
    sig_counts: dict[str, int],
    consistency: float | None,
) -> str:
    """Generate natural-language interpretation of the novelty map."""
    parts: list[str] = []

    terrain_labels = {
        "well_trodden": "This is well-trodden ground.",
        "frontier": "This is frontier territory — some established ground, some open questions.",
        "uncharted": "This is largely uncharted territory. The model is extrapolating significantly.",
        "deep_unknown": "This is deep unknown. The model has very little basis for its claims here.",
    }
    parts.append(terrain_labels.get(terrain, ""))

    n_boundary = sig_counts.get("decision_boundary", 0)
    n_terra = sig_counts.get("terra_incognita", 0)
    n_collision = sig_counts.get("framework_collision", 0)

    if n_boundary > 0:
        parts.append(
            f"{n_boundary} token(s) sit at decision boundaries where "
            f"competing frameworks meet."
        )
    if n_terra > 0:
        parts.append(
            f"{n_terra} token(s) are in terra incognita — the model is "
            f"genuinely unsure and spreading probability widely."
        )
    if n_collision > 0:
        parts.append(
            f"{n_collision} token(s) show framework collision — confident "
            f"but inconsistent across reasoning paths."
        )

    if consistency is not None and consistency < 0.3:
        parts.append(
            "Multiple reasoning paths reach different conclusions. "
            "The disagreement itself is informative — surface the "
            "competing views rather than choosing one."
        )

    if not spans:
        parts.append(
            "No regions of particular novelty detected. "
            "The model is operating within its training distribution."
        )

    return " ".join(parts)
