"""Diamond FTIR desktop app. Start with ``diamond-ftir-app`` (optionally followed by files).

This is the generic spectral workbench (:mod:`diamond_ftir_package.workbench.qt`) with the
diamond FTIR analysis registered; the analysis itself is ``pipeline`` / ``maps``, the same
code the CLI uses.
"""

from __future__ import annotations

from ..session import diamond_session
from ..workbench.qt.window import MainWindow


def main() -> int:
    from ..workbench.qt.window import main as run

    return run(diamond_session, title="Diamond FTIR", settings_key="diamond-ftir")


__all__ = ["MainWindow", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
