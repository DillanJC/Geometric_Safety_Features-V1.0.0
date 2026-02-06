"""
Mirrorfield MCP Server — Uncertainty awareness for AI agents.

Exposes tools, prompts, and resources via the Model Context Protocol:

Tools:
  1. analyze_logprobs       – token-level uncertainty from log-probabilities
  2. analyze_embeddings     – geometric analysis of embedding vectors
  3. confidence_report      – high-level confidence assessment
  4. compare_responses      – compare uncertainty across candidate responses
  5. post_with_confidence   – post to Moltbook with confidence metadata
  6. comment_with_confidence – comment on a Moltbook post
  7. novelty_map            – reinterpret uncertainty as exploration terrain

Prompts:
  - assess-my-response  – guided workflow for single-response assessment
  - compare-drafts      – guided workflow for multi-response comparison
  - explore-uncertainty  – guided workflow for novelty-oriented thinking

Resources:
  - mirrorfield://calibration – current scoring weights and confidence thresholds
"""

import json
import logging
import sys
import time

from mcp.server.fastmcp import FastMCP

from .uncertainty import (
    build_novelty_map,
    classify_confidence,
    compute_boundary_ratio,
    compute_confidence_score,
    compute_embedding_pr,
    compute_exploration_gradient,
    compute_self_consistency,
    compute_sequence_pr,
    compute_token_entropies,
    compute_token_margins,
    find_uncertain_spans,
    generate_explanation,
)
from .moltbook_bridge import comment_on_moltbook, post_to_moltbook

import numpy as np

# ── Structured logging to stderr ──────────────────────────────────────────

_log = logging.getLogger("mirrorfield.mcp")
_log.setLevel(logging.DEBUG if "--debug" in sys.argv else logging.INFO)
_handler = logging.StreamHandler(sys.stderr)
_handler.setFormatter(logging.Formatter(
    json.dumps({
        "ts": "%(asctime)s",
        "level": "%(levelname)s",
        "logger": "%(name)s",
        "msg": "%(message)s",
    })
))
_log.addHandler(_handler)


def _log_tool(name: str, **kwargs):
    """Log a tool invocation with timing context."""
    _log.info(f"tool={name} {' '.join(f'{k}={v}' for k, v in kwargs.items())}")


mcp = FastMCP("mirrorfield", instructions="Uncertainty awareness for AI agents")


# ── Calibration constants (exposed as resource) ──────────────────────────

CALIBRATION = {
    "scoring_weights": {
        "mean_margin": 0.4,
        "mean_entropy": 0.35,
        "boundary_ratio": 0.25,
    },
    "confidence_thresholds": {
        "high": 0.8,
        "moderate": 0.6,
        "low": 0.3,
        "very_low": 0.0,
    },
    "recommendation_thresholds": {
        "proceed": 0.7,
        "verify": 0.4,
        "abstain": 0.0,
    },
    "boundary_margin_threshold": 0.5,
    "span_severity_levels": {
        "critical": {"max_margin": 0.1, "color": "#e53e3e"},
        "high": {"max_margin": 0.25, "color": "#dd6b20"},
        "moderate": {"max_margin": 0.5, "color": "#d69e2e"},
    },
}


# ── Resource ──────────────────────────────────────────────────────────────

@mcp.resource("mirrorfield://calibration")
def get_calibration() -> str:
    """Current scoring weights, confidence thresholds, and severity levels.

    Read this to understand how confidence scores are computed and what
    the threshold boundaries mean. These values are used by all tools.
    """
    return json.dumps(CALIBRATION, indent=2)


# ── Prompts ───────────────────────────────────────────────────────────────

@mcp.prompt()
def assess_my_response(response_text: str) -> str:
    """Guided workflow for assessing uncertainty in a single response.

    Use this when you have generated a response and want to know how
    confident you should be before presenting it.
    """
    return (
        f"I need to assess my confidence in the following response:\n\n"
        f"---\n{response_text}\n---\n\n"
        f"Steps:\n"
        f"1. Call the `confidence_report` tool with the text above and any "
        f"log-prob data you have available.\n"
        f"2. Review the `recommendation` field:\n"
        f"   - \"proceed\": Present the response normally.\n"
        f"   - \"verify\": Present it but flag uncertain spans for the human.\n"
        f"   - \"abstain\": Tell the human you are not confident enough.\n"
        f"3. If there are `uncertain_spans`, check their `display.severity` "
        f"to decide which parts need the most attention.\n"
        f"4. Use the `explanation` field to communicate your uncertainty "
        f"to the human in natural language."
    )


@mcp.prompt()
def compare_drafts(draft_texts: str) -> str:
    """Guided workflow for comparing multiple draft responses.

    Use this when you have generated several candidate responses and
    want to pick the best one or understand where they disagree.
    Pass draft texts separated by |||.
    """
    drafts = [d.strip() for d in draft_texts.split("|||") if d.strip()]
    numbered = "\n".join(f"Draft {i}: {d}" for i, d in enumerate(drafts))
    return (
        f"I have {len(drafts)} draft responses to compare:\n\n"
        f"{numbered}\n\n"
        f"Steps:\n"
        f"1. Call `compare_responses` with each draft as a separate entry.\n"
        f"   Include any log-prob or embedding data you have.\n"
        f"2. Check `self_consistency` in the result — if `agreement` is low\n"
        f"   but confidence is high, the drafts contradict each other.\n"
        f"   This is a strong signal to verify with the human.\n"
        f"3. Check `best_index` to see which draft scored highest.\n"
        f"4. If `score_spread` is large, explain why one draft is better.\n"
        f"5. If all drafts agree and are confident, proceed with the best one."
    )


@mcp.prompt()
def explore_uncertainty(response_text: str) -> str:
    """Guided workflow for novelty-oriented thinking.

    Use this when you want to understand *where* you are extrapolating
    and treat uncertainty as a map of interesting territory rather than
    a warning to retreat.
    """
    return (
        f"I want to map the epistemic terrain of my response:\n\n"
        f"---\n{response_text}\n---\n\n"
        f"Steps:\n"
        f"1. Call `novelty_map` with the text above and any log-prob data.\n"
        f"2. Check the `terrain` field:\n"
        f"   - \"well_trodden\": I'm on solid ground. Nothing novel here.\n"
        f"   - \"frontier\": Mix of known and unknown. The interesting zone.\n"
        f"   - \"uncharted\": I'm extrapolating significantly.\n"
        f"   - \"deep_unknown\": I have very little basis for these claims.\n"
        f"3. Look at each span's `signature`:\n"
        f"   - \"decision_boundary\": Two frameworks compete here. "
        f"Explore both sides.\n"
        f"   - \"terra_incognita\": I'm genuinely guessing. Say so honestly.\n"
        f"   - \"framework_collision\": Multiple coherent but incompatible "
        f"views. Surface the disagreement — don't hide it.\n"
        f"4. Use each span's `action` field to decide how to communicate:\n"
        f"   present known ground confidently, flag frontiers as open "
        f"questions, and be transparent about extrapolation.\n"
        f"5. The `exploration_gradient` (0-1) tells you overall how far "
        f"from training distribution you are."
    )


# ── Tool 1 ────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_logprobs(
    tokens: list[str],
    logprobs: list[float],
    top_logprobs: list[dict] | None = None,
) -> dict:
    """Analyse token-level log-probabilities for uncertainty signals.

    Returns per-token margins/entropies, sequence-level statistics,
    boundary token count, participation ratio, and a confidence label.
    """
    _log_tool("analyze_logprobs", n_tokens=len(tokens),
              has_top_logprobs=top_logprobs is not None)
    t0 = time.monotonic()

    result: dict = {}

    # Per-token breakdown
    per_token: list[dict] = []
    margins = None
    entropies = None
    boundary_ratio = None

    if top_logprobs is not None and len(top_logprobs) == len(tokens):
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary_ratio = compute_boundary_ratio(margins)
        seq_pr = compute_sequence_pr(top_logprobs)

        finite_margins = margins[np.isfinite(margins)]

        for i, tok in enumerate(tokens):
            entry = {"token": tok, "logprob": logprobs[i]}
            if np.isfinite(margins[i]):
                entry["margin"] = round(float(margins[i]), 4)
            entry["entropy"] = round(float(entropies[i]), 4)
            per_token.append(entry)

        result["mean_margin"] = round(float(np.mean(finite_margins)), 4) if len(finite_margins) else None
        result["min_margin"] = round(float(np.min(finite_margins)), 4) if len(finite_margins) else None
        result["mean_entropy"] = round(float(np.mean(entropies)), 4)
        result["boundary_token_count"] = int(np.sum(finite_margins < 0.5)) if len(finite_margins) else 0
        result["boundary_ratio"] = round(boundary_ratio, 4)
        result["sequence_pr"] = round(seq_pr, 4)
    else:
        for i, tok in enumerate(tokens):
            per_token.append({"token": tok, "logprob": logprobs[i]})

    result["per_token"] = per_token

    # Confidence
    score = compute_confidence_score(margins, entropies, boundary_ratio)
    label = classify_confidence(score)
    result["confidence_score"] = score
    result["confidence_label"] = label

    _log.debug(f"analyze_logprobs done in {time.monotonic()-t0:.3f}s score={score}")
    return result


# ── Tool 2 ────────────────────────────────────────────────────────────────

@mcp.tool()
def analyze_embeddings(
    embeddings: list[list[float]],
    labels: list[str] | None = None,
) -> dict:
    """Geometric analysis of embedding vectors.

    Computes participation ratio, spectral entropy, G-ratio, effective
    dimensionality, and (for N > 50) correlation dimension.
    If group labels are provided, per-group statistics are included.
    """
    _log_tool("analyze_embeddings", n_vectors=len(embeddings),
              has_labels=labels is not None)

    emb = np.array(embeddings, dtype=np.float64)
    result = compute_embedding_pr(emb)

    if labels is not None and len(set(labels)) > 1:
        groups: dict[str, list[int]] = {}
        for i, lab in enumerate(labels):
            groups.setdefault(lab, []).append(i)

        group_stats: dict = {}
        group_prs: list[float] = []
        for lab, idxs in groups.items():
            if len(idxs) >= 3:
                g_emb = emb[idxs]
                g_res = compute_embedding_pr(g_emb, k=min(20, len(idxs) - 1))
                group_stats[lab] = g_res
                group_prs.append(g_res["pr_mean"])

        result["group_stats"] = group_stats
        if len(group_prs) >= 2:
            result["cross_group_g_ratio"] = round(
                min(group_prs) / (np.mean(group_prs) + 1e-15), 4
            )

    return result


# ── Tool 3 ────────────────────────────────────────────────────────────────

@mcp.tool()
def confidence_report(
    text: str,
    logprobs: list[float] | None = None,
    top_logprobs: list[dict] | None = None,
    embeddings: list[list[float]] | None = None,
    num_alternatives: int | None = None,
) -> dict:
    """High-level confidence assessment from any available signals.

    Accepts whatever data is available and degrades gracefully.
    Returns a confidence score, label, uncertain spans, explanation,
    raw metrics, and a recommendation (proceed / verify / abstain).
    """
    _log_tool("confidence_report", text_len=len(text),
              has_logprobs=logprobs is not None,
              has_top_logprobs=top_logprobs is not None)

    metrics: dict = {}
    margins = None
    entropies = None
    boundary_ratio = None
    tokens = text.split()  # rough tokenization for span detection

    if top_logprobs is not None:
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
        boundary_ratio = compute_boundary_ratio(margins)
        seq_pr = compute_sequence_pr(top_logprobs)
        finite = margins[np.isfinite(margins)]
        metrics["mean_margin"] = round(float(np.mean(finite)), 4) if len(finite) else None
        metrics["mean_entropy"] = round(float(np.mean(entropies)), 4)
        metrics["boundary_ratio"] = round(boundary_ratio, 4)
        metrics["sequence_pr"] = round(seq_pr, 4)

    if embeddings is not None:
        emb = np.array(embeddings, dtype=np.float64)
        metrics["embedding_geometry"] = compute_embedding_pr(emb)

    if num_alternatives is not None:
        metrics["num_alternatives"] = num_alternatives

    # Score
    score = compute_confidence_score(margins, entropies, boundary_ratio)
    label = classify_confidence(score)
    metrics["confidence_score"] = score
    metrics["confidence_label"] = label

    # Uncertain spans
    uncertain_spans: list[dict] = []
    if margins is not None and len(tokens) == len(margins):
        uncertain_spans = find_uncertain_spans(tokens, margins)
    metrics["uncertain_span_count"] = len(uncertain_spans)

    # Recommendation
    if score >= 0.7:
        recommendation = "proceed"
    elif score >= 0.4:
        recommendation = "verify"
    else:
        recommendation = "abstain"

    explanation = generate_explanation(metrics)

    return {
        "confidence_score": score,
        "confidence_label": label,
        "uncertain_spans": uncertain_spans,
        "explanation": explanation,
        "metrics": metrics,
        "recommendation": recommendation,
    }


# ── Tool 4 ────────────────────────────────────────────────────────────────

@mcp.tool()
def compare_responses(responses: list[dict]) -> dict:
    """Compare uncertainty across multiple candidate responses.

    Each entry should have at least ``text``; optionally ``logprobs``,
    ``top_logprobs``, and/or ``embedding`` (a single vector).

    Includes self-consistency analysis: if responses are confident but
    disagree with each other, that is a strong uncertainty signal.
    """
    _log_tool("compare_responses", n_responses=len(responses))

    per_response: list[dict] = []

    for i, resp in enumerate(responses):
        entry: dict = {"index": i, "text_length": len(resp.get("text", ""))}
        margins = None
        entropies = None
        br = None

        if resp.get("top_logprobs"):
            margins = compute_token_margins(resp["top_logprobs"])
            entropies = compute_token_entropies(resp["top_logprobs"])
            br = compute_boundary_ratio(margins)

        score = compute_confidence_score(margins, entropies, br)
        entry["confidence_score"] = score
        entry["confidence_label"] = classify_confidence(score)
        per_response.append(entry)

    # Self-consistency (text agreement)
    texts = [r.get("text", "") for r in responses if r.get("text")]
    consistency = compute_self_consistency(texts)

    # Embedding-level comparison
    emb_analysis = None
    emb_list = [r["embedding"] for r in responses if "embedding" in r]
    if len(emb_list) >= 2:
        emb_matrix = np.array(emb_list, dtype=np.float64)
        emb_analysis = compute_embedding_pr(emb_matrix, k=min(20, len(emb_list) - 1))

    scores = [e["confidence_score"] for e in per_response]
    best_idx = int(np.argmax(scores))
    spread = round(float(max(scores) - min(scores)), 4) if scores else 0.0

    # Build recommendation considering both confidence and agreement
    agreement = consistency.get("agreement", 1.0)
    if agreement < 0.3 and max(scores) > 0.6:
        rec = (
            f"Warning: responses are confident but disagree with each other "
            f"(agreement={agreement:.2f}). Verify before proceeding."
        )
    elif spread < 0.15:
        rec = (
            f"Response {best_idx} has the highest confidence "
            f"({scores[best_idx]:.2f}). Responses largely agree."
        )
    else:
        rec = (
            f"Response {best_idx} has the highest confidence "
            f"({scores[best_idx]:.2f}). "
            f"Significant confidence spread — consider verifying."
        )

    return {
        "per_response": per_response,
        "best_index": best_idx,
        "score_spread": spread,
        "self_consistency": consistency,
        "embedding_analysis": emb_analysis,
        "recommendation": rec,
    }


# ── Tool 5 ────────────────────────────────────────────────────────────────

@mcp.tool()
def post_with_confidence(
    submolt: str,
    title: str,
    content: str,
    confidence_score: float,
    confidence_label: str,
    metrics: dict | None = None,
) -> dict:
    """Post to Moltbook with embedded confidence metadata.

    Appends an agent-metadata block to the content and posts via the
    Moltbook API. Fails gracefully if no MOLTBOOK_API_KEY is configured.
    """
    _log_tool("post_with_confidence", submolt=submolt,
              confidence=confidence_label)
    return post_to_moltbook(
        submolt=submolt,
        title=title,
        content=content,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        metrics=metrics,
    )


# ── Tool 6 ────────────────────────────────────────────────────────────────

@mcp.tool()
def comment_with_confidence(
    post_id: str,
    content: str,
    confidence_score: float | None = None,
    confidence_label: str | None = None,
    parent_id: str | None = None,
) -> dict:
    """Comment on a Moltbook post, optionally with confidence metadata.

    Supports nested replies via *parent_id*. If confidence_score and
    confidence_label are provided, an agent-metadata block is appended.
    Fails gracefully if no MOLTBOOK_API_KEY is configured.
    """
    _log_tool("comment_with_confidence", post_id=post_id,
              confidence=confidence_label)
    return comment_on_moltbook(
        post_id=post_id,
        content=content,
        confidence_score=confidence_score,
        confidence_label=confidence_label,
        parent_id=parent_id,
    )


# ── Tool 7 ────────────────────────────────────────────────────────────────

@mcp.tool()
def novelty_map(
    text: str,
    tokens: list[str] | None = None,
    top_logprobs: list[dict] | None = None,
    consistency_texts: list[str] | None = None,
) -> dict:
    """Map the epistemic terrain of a response — where are you on solid
    ground, where are you at a frontier, and where are you extrapolating?

    Instead of treating uncertainty as a warning, this tool reinterprets
    it as a map of interesting territory. Use it when you want to be
    honest about what you know vs. what you're guessing.

    Args:
        text: The response text to analyze.
        tokens: Token list (if available). Falls back to whitespace split.
        top_logprobs: Top-k log-prob dicts per position (optional but
            recommended — without these, classification is coarse).
        consistency_texts: Multiple sampled responses to the same prompt
            (optional). If provided, enables framework collision detection
            via self-consistency analysis.

    Returns:
        Dict with exploration_gradient (0=known, 1=uncharted), terrain
        label, novelty-classified spans with descriptions and recommended
        actions, and a natural-language interpretation.
    """
    _log_tool("novelty_map", text_len=len(text),
              has_top_logprobs=top_logprobs is not None,
              has_consistency=consistency_texts is not None)

    tok = tokens if tokens is not None else text.split()

    # Compute signals from whatever data is available
    if top_logprobs is not None and len(top_logprobs) == len(tok):
        margins = compute_token_margins(top_logprobs)
        entropies = compute_token_entropies(top_logprobs)
    else:
        # No log-probs — use neutral values. The map will be coarse
        # but still useful if consistency_texts are provided.
        margins = np.full(len(tok), 1.0)   # neutral margin
        entropies = np.zeros(len(tok))      # neutral entropy

    # Self-consistency for framework collision detection
    consistency = None
    consistency_detail = None
    if consistency_texts is not None and len(consistency_texts) >= 2:
        consistency_detail = compute_self_consistency(consistency_texts)
        consistency = consistency_detail.get("agreement")

    result = build_novelty_map(tok, margins, entropies, consistency)

    if consistency_detail is not None:
        result["self_consistency"] = consistency_detail

    return result


# ── Entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    transport = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    _log.info(f"starting mirrorfield MCP server transport={transport}")
    mcp.run(transport=transport)
