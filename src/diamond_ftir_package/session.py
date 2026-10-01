"""A workbench session with the diamond FTIR analysis registered (for scripts and tests).

The app finds the diamond plug-in through its entry point; this works without installing.
"""

from __future__ import annotations

from .plugin import register
from .workbench.plugins import default_registry
from .workbench.session import Dataset, Session


def diamond_session() -> Session:
    registry = default_registry(with_entry_points=False)
    register(registry)
    return Session(registry=registry)


__all__ = ["Dataset", "Session", "diamond_session"]
