"""
Launcher for the minimal/simple Diamond FTIR GUI (no matplotlib, minimal imports).

Use this to confirm the window opens without SEGV. Then we add features step by step.

  python -m diamond_ftir_package.gui_launcher_simple
"""

import sys

if __name__ == "__main__":
    try:
        from diamond_ftir_package.DiamondFTIR_GUI_simple import main
        main()
    except Exception as e:
        import traceback
        print("Error:", e, file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(1)
