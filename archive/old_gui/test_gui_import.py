#!/usr/bin/env python
"""
Simple test script to verify the GUI can be imported and basic components work.
"""

import sys
from pathlib import Path

# Add src to path if needed
src_path = Path(__file__).parent / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

try:
    from diamond_ftir_package.DiamondFTIR_GUI import DiamondFTIRGUI
    print("✓ GUI module imported successfully")
    
    # Test if tkinter is available
    import tkinter as tk
    print("✓ tkinter is available")
    
    # Test if matplotlib backend works
    import matplotlib
    matplotlib.use('TkAgg')  # Set backend before importing pyplot
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    print("✓ matplotlib TkAgg backend is available")
    
    print("\nAll checks passed! The GUI should work correctly.")
    print("\nTo launch the GUI, run:")
    print("  python -m diamond_ftir_package.gui_launcher")
    print("  or")
    print("  python src/diamond_ftir_package/gui_launcher.py")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("\nTroubleshooting:")
    print("1. Make sure you're running from the project root")
    print("2. Install dependencies: pip install matplotlib pandas numpy")
    print("3. For tkinter on Linux: sudo apt-get install python3-tk")
    sys.exit(1)
except Exception as e:
    print(f"✗ Error: {e}")
    sys.exit(1)
