#!/usr/bin/env python
"""
Launcher script for the Diamond FTIR GUI application.

Usage:
    python -m diamond_ftir_package.gui_launcher
    or
    python src/diamond_ftir_package/gui_launcher.py
"""

import sys
import traceback

# Ensure errors are visible (e.g. when launched from Finder or IDE)
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except (AttributeError, OSError):
    pass


def _check_tk():
    """Verify Tk is available (required for the GUI)."""
    try:
        import tkinter as tk
        tk.Tk().destroy()
        return True
    except Exception as e:
        print("Tk (tkinter) is not available:", e, file=sys.stderr)
        return False


def _print_uv_help():
    print("""
The GUI needs Tk. With uv-managed Python, Tk support can be missing or broken.

Fix (recommended):
  1. Upgrade uv:  uv self update
  2. Reinstall Python:  uv python upgrade --reinstall

Alternatively, run the GUI with a Python that has Tk (e.g. python.org installer):
  /usr/bin/python3 -m diamond_ftir_package.gui_launcher
""", file=sys.stderr)


if __name__ == "__main__":
    if not _check_tk():
        _print_uv_help()
        sys.exit(1)

    try:
        from diamond_ftir_package.DiamondFTIR_GUI import main
        main()
    except ImportError as e:
        if "tkagg" in str(e).lower() or "tk" in str(e).lower():
            print("Could not load the GUI (Tk/matplotlib backend issue):", e, file=sys.stderr)
            _print_uv_help()
            sys.exit(1)
        raise
    except Exception as e:
        print("GUI crashed with the following error:\n", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        sys.exit(1)

# If you see SEGV (segmentation fault) with no Python traceback, the Tk+matplotlib
# stack may be incompatible with your Python build. Try:
#   - Official macOS Python: https://www.python.org/downloads/
#   - Run:  /usr/bin/python3 -m diamond_ftir_package.gui_launcher
# The GUI defers creating the plot panel until after the window is shown to reduce SEGV risk.
