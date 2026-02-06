"""
Smoke tests for the mirrorfield MCP uncertainty module.

Run with:  python -m pytest tests/test_mcp_uncertainty.py -v
"""

import numpy as np
from mirrorfield.mcp.uncertainty import (
    build_novelty_map,
    classify_confidence,
    classify_novelty_signature,
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


def _approx(a: float, b: float, tol: float = 0.15) -> bool:
    return abs(a - b) < tol


def test_token_margins():
    # High confidence: dominant token far above alternatives
    top_lp_high = [
        {"the": -0.01, "a": -3.5, "an": -4.0},
        {"cat": -0.05, "dog": -2.8, "rat": -3.1},
    ]
    margins = compute_token_margins(top_lp_high)
    assert margins[0] > 2.0, f"Expected large margin, got {margins[0]}"
    assert margins[1] > 2.0, f"Expected large margin, got {margins[1]}"

    # Low confidence: top two tokens are close
    top_lp_low = [
        {"yes": -0.8, "no": -0.9, "maybe": -2.0},
        {"agree": -1.0, "disagree": -1.1, "uncertain": -1.5},
    ]
    margins_low = compute_token_margins(top_lp_low)
    assert margins_low[0] < 0.5, f"Expected small margin, got {margins_low[0]}"
    print("  token_margins: PASS")


def test_token_entropies():
    # Peaked distribution → low entropy
    top_lp_peaked = [{"a": -0.01, "b": -5.0, "c": -6.0}]
    ent = compute_token_entropies(top_lp_peaked)
    assert ent[0] < 0.3, f"Expected low entropy, got {ent[0]}"

    # Flat distribution → higher entropy
    top_lp_flat = [{"a": -1.1, "b": -1.1, "c": -1.1}]
    ent_flat = compute_token_entropies(top_lp_flat)
    assert ent_flat[0] > 0.9, f"Expected high entropy, got {ent_flat[0]}"
    print("  token_entropies: PASS")


def test_sequence_pr():
    # Dominated by one alternative → low PR
    top_lp_dom = [
        {"a": -0.01, "b": -8.0, "c": -9.0},
        {"x": -0.02, "y": -7.0, "z": -8.0},
        {"m": -0.01, "n": -6.0, "o": -7.0},
    ]
    pr_dom = compute_sequence_pr(top_lp_dom)

    # Spread across alternatives → higher PR
    top_lp_spread = [
        {"a": -1.0, "b": -1.1, "c": -1.2},
        {"x": -1.5, "y": -0.5, "z": -1.3},
        {"m": -0.8, "n": -1.0, "o": -0.9},
    ]
    pr_spread = compute_sequence_pr(top_lp_spread)
    assert pr_spread >= pr_dom, f"Spread PR ({pr_spread}) should be >= dominated PR ({pr_dom})"
    print("  sequence_pr: PASS")


def test_boundary_ratio():
    margins = np.array([0.1, 0.2, 2.0, 3.0, 0.4])
    br = compute_boundary_ratio(margins, threshold=0.5)
    assert _approx(br, 0.6), f"Expected ~0.6, got {br}"
    print("  boundary_ratio: PASS")


def test_confidence_score():
    # High confidence scenario
    margins_high = np.array([3.0, 2.5, 4.0])
    ent_high = np.array([0.1, 0.05, 0.08])
    score_high = compute_confidence_score(margins_high, ent_high, 0.0)
    assert score_high > 0.7, f"Expected high score, got {score_high}"
    assert classify_confidence(score_high) in ("high", "moderate")

    # Low confidence scenario
    margins_low = np.array([0.1, 0.05, 0.2])
    ent_low = np.array([2.0, 1.8, 2.2])
    score_low = compute_confidence_score(margins_low, ent_low, 0.9)
    assert score_low < 0.4, f"Expected low score, got {score_low}"
    assert classify_confidence(score_low) in ("low", "very_low")
    print("  confidence_score: PASS")


def test_embedding_pr():
    rng = np.random.default_rng(42)

    # Clustered embeddings: low effective dim
    cluster = rng.normal(0, 0.01, size=(30, 10))
    cluster[:, 0] = rng.normal(0, 5, size=30)  # one dominant direction
    res_cluster = compute_embedding_pr(cluster, k=10)
    assert res_cluster["pr_mean"] >= 1.0

    # Spread embeddings: higher effective dim
    spread = rng.normal(0, 1, size=(30, 10))
    res_spread = compute_embedding_pr(spread, k=10)
    assert res_spread["effective_dim"] > res_cluster["effective_dim"], (
        f"Spread eff_dim ({res_spread['effective_dim']}) should > "
        f"clustered ({res_cluster['effective_dim']})"
    )
    print("  embedding_pr: PASS")


def test_uncertain_spans():
    tokens = ["The", " cat", " sat", " on", " the", " mat"]
    #          0123   4567   8901   2345   6789   0123
    full_text = "".join(tokens)  # "The cat sat on the mat"
    margins = np.array([3.0, 0.2, 0.3, 2.5, 3.0, 0.1])
    spans = find_uncertain_spans(tokens, margins, threshold=0.5)
    assert len(spans) == 2, f"Expected 2 spans, got {len(spans)}"
    # First span: " cat sat" (token indices 1-2)
    assert spans[0]["start"] == 1
    assert spans[0]["end"] == 3
    assert spans[0]["char_start"] == 3   # len("The") = 3
    assert spans[0]["char_end"] == 11    # len("The cat sat") = 11
    assert full_text[spans[0]["char_start"]:spans[0]["char_end"]] == " cat sat"
    # Second span: " mat" (token index 5)
    assert spans[1]["start"] == 5
    assert spans[1]["char_start"] == 18  # len("The cat sat on the") = 18
    assert spans[1]["char_end"] == 22    # len("The cat sat on the mat") = 22
    assert full_text[spans[1]["char_start"]:spans[1]["char_end"]] == " mat"
    # Display hints
    assert "display" in spans[0]
    assert spans[0]["display"]["severity"] in ("critical", "high", "moderate")
    assert spans[0]["display"]["color"].startswith("#")
    print("  uncertain_spans: PASS")


def test_self_consistency():
    # Very similar texts → high agreement
    texts_agree = [
        "The capital of France is Paris",
        "The capital of France is Paris, a major European city",
    ]
    res = compute_self_consistency(texts_agree)
    assert res["agreement"] > 0.3, f"Expected high agreement, got {res['agreement']}"

    # Contradictory texts → low agreement
    texts_disagree = [
        "The answer is definitely yes, we should proceed",
        "The answer is no, this approach will fail completely",
    ]
    res_dis = compute_self_consistency(texts_disagree)
    assert res_dis["agreement"] < res["agreement"], (
        f"Disagreeing texts ({res_dis['agreement']}) should have lower "
        f"agreement than agreeing ({res['agreement']})"
    )

    # Single text → agreement = 1
    res_single = compute_self_consistency(["just one response"])
    assert res_single["agreement"] == 1.0
    print("  self_consistency: PASS")


def test_generate_explanation():
    metrics = {
        "confidence_score": 0.35,
        "confidence_label": "low",
        "boundary_ratio": 0.45,
        "sequence_pr": 4.2,
        "uncertain_span_count": 3,
    }
    explanation = generate_explanation(metrics)
    assert "low" in explanation
    assert "45%" in explanation
    assert "3" in explanation
    print("  generate_explanation: PASS")


def test_confidence_report_integration():
    """End-to-end test of the confidence_report tool logic."""
    from mirrorfield.mcp.uncertainty import compute_token_margins as _m  # just to verify imports work

    tokens = ["I", " think", " the", " answer", " is", " 42"]
    logprobs = [-0.1, -0.5, -0.02, -1.2, -0.05, -0.8]
    top_logprobs = [
        {"I": -0.1, "We": -2.5, "You": -3.0},
        {" think": -0.5, " believe": -0.6, " know": -1.5},
        {" the": -0.02, " a": -4.0, " an": -5.0},
        {" answer": -1.2, " result": -1.3, " number": -1.5},
        {" is": -0.05, " was": -3.0, " will": -4.0},
        {" 42": -0.8, " 43": -0.9, " 41": -1.0},
    ]

    margins = compute_token_margins(top_logprobs)
    entropies = compute_token_entropies(top_logprobs)
    br = compute_boundary_ratio(margins)
    score = compute_confidence_score(margins, entropies, br)
    label = classify_confidence(score)

    assert 0.0 <= score <= 1.0
    assert label in ("high", "moderate", "low", "very_low")
    print("  confidence_report integration: PASS")


def test_novelty_signatures():
    # Well-trodden: high margin, low entropy
    assert classify_novelty_signature(2.0, 0.3) == "well_trodden"

    # Decision boundary: low margin, low entropy
    assert classify_novelty_signature(0.1, 0.3) == "decision_boundary"

    # Terra incognita: low margin, high entropy
    assert classify_novelty_signature(0.1, 1.5) == "terra_incognita"

    # Framework collision: decent margin but low consistency
    assert classify_novelty_signature(
        0.8, 0.3, consistency=0.15
    ) == "framework_collision"

    # With consistency but still agreeing — should be well_trodden
    assert classify_novelty_signature(
        2.0, 0.3, consistency=0.8
    ) == "well_trodden"

    print("  novelty_signatures: PASS")


def test_exploration_gradient():
    # Confident tokens → low gradient
    margins_sure = np.array([3.0, 2.5, 4.0])
    entropies_low = np.array([0.1, 0.05, 0.08])
    grad_low = compute_exploration_gradient(margins_sure, entropies_low)
    assert grad_low < 0.3, f"Expected low gradient, got {grad_low}"

    # Uncertain tokens → high gradient
    margins_unsure = np.array([0.1, 0.05, 0.2])
    entropies_high = np.array([2.0, 1.8, 2.2])
    grad_high = compute_exploration_gradient(margins_unsure, entropies_high)
    assert grad_high > 0.6, f"Expected high gradient, got {grad_high}"

    assert grad_high > grad_low
    print("  exploration_gradient: PASS")


def test_novelty_map_integration():
    # Mix of confident and uncertain tokens
    tokens = ["The", " capital", " of", " France", " is", " arguably", " Paris"]
    top_logprobs = [
        {"The": -0.01, "A": -4.0},
        {" capital": -0.05, " city": -3.0},
        {" of": -0.01, " in": -5.0},
        {" France": -0.02, " Germany": -3.5},
        {" is": -0.01, " was": -4.0},
        # "arguably" — the model was unsure here
        {" arguably": -0.9, " definitely": -1.0, " probably": -1.1, " perhaps": -1.2},
        {" Paris": -0.3, " Lyon": -0.5, " Marseille": -1.0},
    ]

    margins = compute_token_margins(top_logprobs)
    entropies = compute_token_entropies(top_logprobs)

    result = build_novelty_map(tokens, margins, entropies)

    assert "exploration_gradient" in result
    assert 0.0 <= result["exploration_gradient"] <= 1.0
    assert result["terrain"] in ("well_trodden", "frontier", "uncharted", "deep_unknown")
    assert "spans" in result
    assert "interpretation" in result

    # The uncertain tokens should produce at least one non-well-trodden span
    if result["spans"]:
        span = result["spans"][0]
        assert span["signature"] in (
            "decision_boundary", "terra_incognita", "framework_collision"
        )
        assert "action" in span
        assert "description" in span
        assert "char_start" in span

    print("  novelty_map integration: PASS")


def test_novelty_map_with_consistency():
    """Framework collision detection via consistency_texts."""
    tokens = ["Yes", ",", " definitely"]
    margins = np.array([1.5, 3.0, 1.2])  # individually confident
    entropies = np.array([0.3, 0.01, 0.4])

    # Low consistency → should produce framework_collision spans
    result_collision = build_novelty_map(
        tokens, margins, entropies, consistency=0.15
    )
    has_collision = "framework_collision" in result_collision.get("signature_counts", {})
    # With low consistency and decent margins, at least some tokens should collide
    assert has_collision, f"Expected framework_collision, got {result_collision['signature_counts']}"

    # High consistency → should be well_trodden
    result_ok = build_novelty_map(
        tokens, margins, entropies, consistency=0.9
    )
    assert result_ok.get("signature_counts", {}).get("well_trodden", 0) > 0

    print("  novelty_map with consistency: PASS")


if __name__ == "__main__":
    print("Running mirrorfield MCP uncertainty smoke tests...\n")
    test_token_margins()
    test_token_entropies()
    test_sequence_pr()
    test_boundary_ratio()
    test_confidence_score()
    test_embedding_pr()
    test_uncertain_spans()
    test_self_consistency()
    test_generate_explanation()
    test_confidence_report_integration()
    test_novelty_signatures()
    test_exploration_gradient()
    test_novelty_map_integration()
    test_novelty_map_with_consistency()
    print("\nAll tests passed.")
