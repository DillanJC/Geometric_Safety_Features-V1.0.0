"""
Mirrorfield MCP sub-package — uncertainty awareness tools for AI agents.

Requires the optional ``mcp[cli]`` dependency::

    pip install "mcp[cli]>=1.20"
"""

try:
    from .server import mcp  # noqa: F401
except ImportError:  # mcp[cli] not installed
    mcp = None
