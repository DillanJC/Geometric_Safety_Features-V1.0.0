"""
Moltbook REST API bridge for posting with confidence metadata.

Uses the real Moltbook API (https://github.com/moltbook/api) directly via
urllib — no third-party SDK needed. Reads MOLTBOOK_API_KEY and optionally
MOLTBOOK_API_URL from the environment. All functions degrade gracefully
when the API key is not configured.
"""

import json
import os
import urllib.request
import urllib.error

MOLTBOOK_DEFAULT_URL = "https://api.moltbook.com"


def _api_url() -> str:
    return os.environ.get("MOLTBOOK_API_URL", MOLTBOOK_DEFAULT_URL).rstrip("/")


def is_moltbook_configured() -> bool:
    """Return True if a Moltbook API key is present in the environment."""
    return bool(os.environ.get("MOLTBOOK_API_KEY"))


def post_to_moltbook(
    submolt: str,
    title: str,
    content: str,
    confidence_score: float,
    confidence_label: str,
    metrics: dict | None = None,
    api_key: str | None = None,
) -> dict:
    """Post content to Moltbook with embedded agent-metadata block.

    Uses ``POST /posts`` with Bearer auth per the Moltbook API spec.

    Returns a dict with ``ok``, ``post_id`` on success,
    or ``ok=False`` and an ``error`` message on failure.
    """
    api_key = api_key or os.environ.get("MOLTBOOK_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": (
                "MOLTBOOK_API_KEY not set. Configure it in the environment "
                "to enable Moltbook integration."
            ),
        }

    # Build metadata block
    meta = {
        "confidence_score": round(confidence_score, 4),
        "confidence_label": confidence_label,
    }
    if metrics:
        meta["metrics"] = metrics

    metadata_block = (
        "\n---\n```agent-metadata\n"
        + json.dumps(meta, indent=2)
        + "\n```"
    )
    full_content = content + metadata_block

    # POST /posts  —  requires submolt, title, content
    payload = json.dumps({
        "submolt": submolt,
        "title": title,
        "content": full_content,
    }).encode()

    req = urllib.request.Request(
        f"{_api_url()}/posts",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            body = json.loads(resp.read().decode())
            return {
                "ok": True,
                "post_id": body.get("id"),
            }
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode()
        except Exception:
            err_body = str(exc)
        return {
            "ok": False,
            "error": f"Moltbook API error {exc.code}: {err_body}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Moltbook post failed: {exc}",
        }


def comment_on_moltbook(
    post_id: str,
    content: str,
    confidence_score: float | None = None,
    confidence_label: str | None = None,
    parent_id: str | None = None,
    api_key: str | None = None,
) -> dict:
    """Add a comment to a Moltbook post via ``POST /posts/:id/comments``.

    Optionally appends an agent-metadata block if confidence info is provided.
    Supports nested replies via *parent_id*.
    """
    api_key = api_key or os.environ.get("MOLTBOOK_API_KEY")
    if not api_key:
        return {
            "ok": False,
            "error": (
                "MOLTBOOK_API_KEY not set. Configure it in the environment "
                "to enable Moltbook integration."
            ),
        }

    full_content = content
    if confidence_score is not None and confidence_label is not None:
        meta = {
            "confidence_score": round(confidence_score, 4),
            "confidence_label": confidence_label,
        }
        full_content += (
            "\n---\n```agent-metadata\n"
            + json.dumps(meta, indent=2)
            + "\n```"
        )

    body: dict = {"content": full_content}
    if parent_id is not None:
        body["parent_id"] = parent_id

    payload = json.dumps(body).encode()

    req = urllib.request.Request(
        f"{_api_url()}/posts/{post_id}/comments",
        data=payload,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            resp_body = json.loads(resp.read().decode())
            return {
                "ok": True,
                "comment_id": resp_body.get("id"),
            }
    except urllib.error.HTTPError as exc:
        try:
            err_body = exc.read().decode()
        except Exception:
            err_body = str(exc)
        return {
            "ok": False,
            "error": f"Moltbook API error {exc.code}: {err_body}",
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"Moltbook comment failed: {exc}",
        }
