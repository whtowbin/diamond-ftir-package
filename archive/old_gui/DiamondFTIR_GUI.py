"""
GUI Application for Diamond FTIR Spectrum Processing

This module provides a graphical user interface for loading, processing, and visualizing
diamond FTIR spectra with configurable fit parameters.
"""

import sys
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Dict, List
import threading

from .LoadCSV import CSV_to_IR_Diamond_Spectrum
from .DiamondSpectrum import Diamond_Spectrum

# Matplotlib is imported lazily in _init_plot_panel() to avoid SEGV on macOS
# when Tk and TkAgg initialize together.


def _gui_excepthook(exc_type, exc_value, exc_tb):
    """Print uncaught exceptions so they are visible when the GUI crashes."""
    traceback.print_exception(exc_type, exc_value, exc_tb, file=sys.stderr)
    sys.stderr.flush()


class ParameterPanel(ttk.Frame):
    """A collapsible panel for organizing parameter controls."""
    
    def __init__(self, parent, title, **kwargs):
        super().__init__(parent, **kwargs)
        self.title = title
        self.is_expanded = True
        
        # Title frame with toggle button
        self.title_frame = ttk.Frame(self)
        self.title_frame.pack(fill=tk.X, padx=5, pady=2)
        
        self.toggle_btn = ttk.Button(
            self.title_frame, 
            text=f"▼ {title}", 
            command=self.toggle,
            width=20
        )
        self.toggle_btn.pack(side=tk.LEFT)
        
        # Content frame
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
    
    def toggle(self):
        """Toggle panel expansion."""
        if self.is_expanded:
            self.content_frame.pack_forget()
            self.toggle_btn.config(text=f"▶ {self.title}")
            self.is_expanded = False
        else:
            self.content_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
            self.toggle_btn.config(text=f"▼ {self.title}")
            self.is_expanded = True


class DiamondFTIRGUI:
    """Main GUI application for Diamond FTIR spectrum processing."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Diamond FTIR Spectrum Processor")
        self.root.geometry("1400x900")
        
        # Current spectrum and results
        self.current_spectrum: Optional[Diamond_Spectrum] = None
        self.processed_results: Optional[Dict] = None
        self.batch_files: List[Path] = []
        
        # Default parameters
        self.params = {
            # Saturation detection
            'saturation_cutoff': 2.5,
            'stdev_cut_off': 0.5,
            
            # Baseline fitting
            'baseline_algorithm': 'Whittaker',
            'baseline_lam': 1e10,
            'baseline_p': 0.000005,
            
            # Nitrogen fitting
            'nitrogen_max_C_or_B': 0.01,
            'nitrogen_plot_fit': False,
            'nitrogen_wn_low': 950,
            'nitrogen_wn_high': 1350,
            
            # Platelet measurement
            'platelet_lam': 1000,
            'platelet_p': 0.001,
            
            # Hydrogen peak measurement
            'hydrogen_lam': 0.1,
            'hydrogen_p': 6e-6,
            
            # Processing options
            'measure_platelets': True,
            'measure_hydrogen': True,
            'measure_amber': False,
        }
        
        self.setup_ui()
    
    def setup_ui(self):
        """Set up the user interface."""
        # Create main paned window for resizable panels
        main_paned = ttk.PanedWindow(self.root, orient=tk.HORIZONTAL)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Left panel: Controls and parameters
        left_frame = ttk.Frame(main_paned)
        main_paned.add(left_frame, weight=1)
        
        # Right panel: Visualization and results
        right_frame = ttk.Frame(main_paned)
        main_paned.add(right_frame, weight=2)
        
        self.setup_left_panel(left_frame)
        self.setup_right_panel(right_frame)
    
    def setup_left_panel(self, parent):
        """Set up the left control panel."""
        # File operations
        file_frame = ttk.LabelFrame(parent, text="File Operations", padding=10)
        file_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            file_frame, 
            text="Load Single Spectrum", 
            command=self.load_single_file
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            file_frame, 
            text="Load Batch Files", 
            command=self.load_batch_files
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            file_frame, 
            text="Process Current", 
            command=self.process_spectrum
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            file_frame, 
            text="Process Batch", 
            command=self.process_batch
        ).pack(fill=tk.X, pady=2)
        
        # File info and status
        self.file_info_label = ttk.Label(file_frame, text="No file loaded", wraplength=200)
        self.file_info_label.pack(fill=tk.X, pady=5)
        
        self.status_label = ttk.Label(file_frame, text="Ready", foreground="green")
        self.status_label.pack(fill=tk.X, pady=2)
        
        # Parameters notebook
        params_notebook = ttk.Notebook(parent)
        params_notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Saturation parameters
        sat_frame = ttk.Frame(params_notebook)
        params_notebook.add(sat_frame, text="Saturation")
        self.setup_saturation_params(sat_frame)
        
        # Baseline parameters
        baseline_frame = ttk.Frame(params_notebook)
        params_notebook.add(baseline_frame, text="Baseline")
        self.setup_baseline_params(baseline_frame)
        
        # Nitrogen parameters
        nitrogen_frame = ttk.Frame(params_notebook)
        params_notebook.add(nitrogen_frame, text="Nitrogen")
        self.setup_nitrogen_params(nitrogen_frame)
        
        # Platelet parameters
        platelet_frame = ttk.Frame(params_notebook)
        params_notebook.add(platelet_frame, text="Platelets")
        self.setup_platelet_params(platelet_frame)
        
        # Hydrogen parameters
        hydrogen_frame = ttk.Frame(params_notebook)
        params_notebook.add(hydrogen_frame, text="Hydrogen")
        self.setup_hydrogen_params(hydrogen_frame)
        
        # Processing options
        options_frame = ttk.Frame(params_notebook)
        params_notebook.add(options_frame, text="Options")
        self.setup_processing_options(options_frame)
        
        # Export buttons
        export_frame = ttk.LabelFrame(parent, text="Export", padding=10)
        export_frame.pack(fill=tk.X, padx=5, pady=5)
        
        ttk.Button(
            export_frame, 
            text="Export Results (CSV)", 
            command=self.export_results
        ).pack(fill=tk.X, pady=2)
        
        ttk.Button(
            export_frame, 
            text="Save Plot", 
            command=self.save_plot
        ).pack(fill=tk.X, pady=2)
    
    def setup_saturation_params(self, parent):
        """Set up saturation detection parameters."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Saturation cutoff
        ttk.Label(frame, text="Saturation Cutoff:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.sat_cutoff_var = tk.DoubleVar(value=self.params['saturation_cutoff'])
        sat_cutoff_spin = ttk.Spinbox(
            frame, 
            from_=0.1, 
            to=10.0, 
            increment=0.1, 
            textvariable=self.sat_cutoff_var,
            width=15
        )
        sat_cutoff_spin.grid(row=0, column=1, sticky=tk.E, pady=5)
        
        # Stdev cutoff
        ttk.Label(frame, text="Stdev Cutoff:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.stdev_cutoff_var = tk.DoubleVar(value=self.params['stdev_cut_off'])
        stdev_cutoff_spin = ttk.Spinbox(
            frame, 
            from_=0.01, 
            to=2.0, 
            increment=0.01, 
            textvariable=self.stdev_cutoff_var,
            width=15
        )
        stdev_cutoff_spin.grid(row=1, column=1, sticky=tk.E, pady=5)
        
        ttk.Label(
            frame, 
            text="Parameters for detecting saturated diamond peaks.\nLower values = more sensitive detection.",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def setup_baseline_params(self, parent):
        """Set up baseline fitting parameters."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Algorithm selection
        ttk.Label(frame, text="Algorithm:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.baseline_algo_var = tk.StringVar(value=self.params['baseline_algorithm'])
        algo_combo = ttk.Combobox(
            frame, 
            textvariable=self.baseline_algo_var,
            values=['Whittaker', 'ALS'],
            state='readonly',
            width=12
        )
        algo_combo.grid(row=0, column=1, sticky=tk.E, pady=5)
        
        # Lambda parameter
        ttk.Label(frame, text="Lambda (×10¹⁰):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.baseline_lam_var = tk.DoubleVar(value=self.params['baseline_lam'] / 1e10)
        baseline_lam_spin = ttk.Spinbox(
            frame, 
            from_=0.1, 
            to=100.0, 
            increment=0.1, 
            textvariable=self.baseline_lam_var,
            width=15,
            format="%.2f"
        )
        baseline_lam_spin.grid(row=1, column=1, sticky=tk.E, pady=5)
        
        # P parameter
        ttk.Label(frame, text="P (×10⁻⁶):").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.baseline_p_var = tk.DoubleVar(value=self.params['baseline_p'] * 1e6)
        baseline_p_spin = ttk.Spinbox(
            frame, 
            from_=0.1, 
            to=100.0, 
            increment=0.1, 
            textvariable=self.baseline_p_var,
            width=15,
            format="%.2f"
        )
        baseline_p_spin.grid(row=2, column=1, sticky=tk.E, pady=5)
        
        ttk.Label(
            frame, 
            text="Baseline correction parameters.\nHigher lambda = smoother baseline.\nHigher p = more asymmetric.",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def setup_nitrogen_params(self, parent):
        """Set up nitrogen fitting parameters."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Max C or B
        ttk.Label(frame, text="Max C/B Ratio:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.nitrogen_max_C_or_B_var = tk.DoubleVar(value=self.params['nitrogen_max_C_or_B'])
        nitrogen_max_C_or_B_spin = ttk.Spinbox(
            frame, 
            from_=0.001, 
            to=0.1, 
            increment=0.001, 
            textvariable=self.nitrogen_max_C_or_B_var,
            width=15,
            format="%.3f"
        )
        nitrogen_max_C_or_B_spin.grid(row=0, column=1, sticky=tk.E, pady=5)
        
        # Wavenumber range
        ttk.Label(frame, text="Wavenumber Low:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.nitrogen_wn_low_var = tk.IntVar(value=self.params['nitrogen_wn_low'])
        nitrogen_wn_low_spin = ttk.Spinbox(
            frame, 
            from_=800, 
            to=1200, 
            increment=10, 
            textvariable=self.nitrogen_wn_low_var,
            width=15
        )
        nitrogen_wn_low_spin.grid(row=1, column=1, sticky=tk.E, pady=5)
        
        ttk.Label(frame, text="Wavenumber High:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.nitrogen_wn_high_var = tk.IntVar(value=self.params['nitrogen_wn_high'])
        nitrogen_wn_high_spin = ttk.Spinbox(
            frame, 
            from_=1200, 
            to=1500, 
            increment=10, 
            textvariable=self.nitrogen_wn_high_var,
            width=15
        )
        nitrogen_wn_high_spin.grid(row=2, column=1, sticky=tk.E, pady=5)
        
        # Plot fit checkbox
        self.nitrogen_plot_fit_var = tk.BooleanVar(value=self.params['nitrogen_plot_fit'])
        ttk.Checkbutton(
            frame, 
            text="Plot Nitrogen Fit", 
            variable=self.nitrogen_plot_fit_var
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=5)
        
        ttk.Label(
            frame, 
            text="Nitrogen component fitting parameters.\nControls balance between C and B centers.",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def setup_platelet_params(self, parent):
        """Set up platelet measurement parameters."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Lambda parameter
        ttk.Label(frame, text="Baseline Lambda:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.platelet_lam_var = tk.DoubleVar(value=self.params['platelet_lam'])
        platelet_lam_spin = ttk.Spinbox(
            frame, 
            from_=100, 
            to=10000, 
            increment=100, 
            textvariable=self.platelet_lam_var,
            width=15
        )
        platelet_lam_spin.grid(row=0, column=1, sticky=tk.E, pady=5)
        
        # P parameter
        ttk.Label(frame, text="Baseline P:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.platelet_p_var = tk.DoubleVar(value=self.params['platelet_p'])
        platelet_p_spin = ttk.Spinbox(
            frame, 
            from_=0.0001, 
            to=0.01, 
            increment=0.0001, 
            textvariable=self.platelet_p_var,
            width=15,
            format="%.4f"
        )
        platelet_p_spin.grid(row=1, column=1, sticky=tk.E, pady=5)
        
        ttk.Label(
            frame, 
            text="Baseline correction parameters for platelet peak measurement\nin the 1340-1500 cm⁻¹ region.",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def setup_hydrogen_params(self, parent):
        """Set up hydrogen peak measurement parameters."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Lambda parameter
        ttk.Label(frame, text="Baseline Lambda:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.hydrogen_lam_var = tk.DoubleVar(value=self.params['hydrogen_lam'])
        hydrogen_lam_spin = ttk.Spinbox(
            frame, 
            from_=0.01, 
            to=1.0, 
            increment=0.01, 
            textvariable=self.hydrogen_lam_var,
            width=15,
            format="%.2f"
        )
        hydrogen_lam_spin.grid(row=0, column=1, sticky=tk.E, pady=5)
        
        # P parameter
        ttk.Label(frame, text="Baseline P (×10⁻⁶):").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.hydrogen_p_var = tk.DoubleVar(value=self.params['hydrogen_p'] * 1e6)
        hydrogen_p_spin = ttk.Spinbox(
            frame, 
            from_=1.0, 
            to=100.0, 
            increment=1.0, 
            textvariable=self.hydrogen_p_var,
            width=15,
            format="%.1f"
        )
        hydrogen_p_spin.grid(row=1, column=1, sticky=tk.E, pady=5)
        
        ttk.Label(
            frame, 
            text="Baseline correction parameters for hydrogen peak measurement\nin the 3060-3180 cm⁻¹ region (3107 cm⁻¹ peak).",
            font=('TkDefaultFont', 8),
            foreground='gray'
        ).grid(row=2, column=0, columnspan=2, sticky=tk.W, pady=10)
    
    def setup_processing_options(self, parent):
        """Set up processing options checkboxes."""
        frame = ttk.Frame(parent)
        frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.measure_platelets_var = tk.BooleanVar(value=self.params['measure_platelets'])
        ttk.Checkbutton(
            frame, 
            text="Measure Platelet Peaks", 
            variable=self.measure_platelets_var
        ).pack(anchor=tk.W, pady=5)
        
        self.measure_hydrogen_var = tk.BooleanVar(value=self.params['measure_hydrogen'])
        ttk.Checkbutton(
            frame, 
            text="Measure Hydrogen Peaks (3107 cm⁻¹)", 
            variable=self.measure_hydrogen_var
        ).pack(anchor=tk.W, pady=5)
        
        self.measure_amber_var = tk.BooleanVar(value=self.params['measure_amber'])
        ttk.Checkbutton(
            frame, 
            text="Measure Amber Center Peaks", 
            variable=self.measure_amber_var
        ).pack(anchor=tk.W, pady=5)
    
    def setup_right_panel(self, parent):
        """Set up the right visualization and results panel."""
        # Create notebook for tabs
        notebook = ttk.Notebook(parent)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Spectrum visualization tab
        viz_frame = ttk.Frame(notebook)
        notebook.add(viz_frame, text="Spectrum")
        self.setup_visualization(viz_frame)
        
        # Results tab
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Results")
        self.setup_results_display(results_frame)
    
    def setup_visualization(self, parent):
        """Set up the spectrum visualization. Matplotlib canvas is created later to avoid SEGV on macOS."""
        self.fig = None
        self.ax = None
        self.canvas = None

        # Placeholder until we create the real canvas (after window is shown)
        self._plot_placeholder = ttk.Frame(parent)
        self._plot_placeholder.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        ttk.Label(
            self._plot_placeholder,
            text="Loading plot…",
            font=("TkDefaultFont", 14),
        ).pack(expand=True)

        # Plot controls (above placeholder)
        controls_frame = ttk.Frame(parent)
        controls_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)
        ttk.Label(controls_frame, text="Plot Type:").pack(side=tk.LEFT, padx=5)
        self.plot_type_var = tk.StringVar(value="Raw")
        plot_type_combo = ttk.Combobox(
            controls_frame,
            textvariable=self.plot_type_var,
            values=["Raw", "Baseline Corrected", "Normalized", "Nitrogen Fit"],
            state="readonly",
            width=20,
        )
        plot_type_combo.pack(side=tk.LEFT, padx=5)
        plot_type_combo.bind("<<ComboboxSelected>>", lambda e: self.update_plot())

        # Create matplotlib canvas after window is shown (avoids Tk+TkAgg SEGV on macOS)
        self._plot_parent = parent
        self.root.after(300, self._init_plot_panel)

    def _init_plot_panel(self):
        """Create matplotlib figure and canvas (called after window is visible)."""
        if self.canvas is not None:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import (
                FigureCanvasTkAgg,
                NavigationToolbar2Tk,
            )
            self.fig, self.ax = plt.subplots(figsize=(10, 6))
            self.fig.patch.set_facecolor("white")
            self.canvas = FigureCanvasTkAgg(self.fig, self._plot_parent)
            self.canvas.draw()
            toolbar = NavigationToolbar2Tk(self.canvas, self._plot_parent)
            toolbar.update()
        except Exception as e:
            if self._plot_placeholder.winfo_exists():
                children = self._plot_placeholder.winfo_children()
                if children:
                    children[0].config(text=f"Plot unavailable: {e}")
            return
        self._plot_placeholder.destroy()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        toolbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.update_plot()
    
    def setup_results_display(self, parent):
        """Set up the results display."""
        # Text widget with scrollbar
        text_frame = ttk.Frame(parent)
        text_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(text_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.results_text = tk.Text(text_frame, wrap=tk.WORD, yscrollcommand=scrollbar.set)
        self.results_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.results_text.yview)
        
        # Initial message
        self.results_text.insert(tk.END, "No results yet. Process a spectrum to see results here.")
        self.results_text.config(state=tk.DISABLED)
    
    def load_single_file(self):
        """Load a single spectrum file."""
        filepath = filedialog.askopenfilename(
            title="Select Spectrum File",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.current_spectrum = CSV_to_IR_Diamond_Spectrum(filepath)
                self.file_info_label.config(text=f"Loaded: {Path(filepath).name}")
                self.status_label.config(text="Ready", foreground="green")
                self.update_plot()
                messagebox.showinfo("Success", f"Loaded spectrum: {Path(filepath).name}")
            except Exception as e:
                self.status_label.config(text="Error", foreground="red")
                messagebox.showerror("Error", f"Failed to load file:\n{str(e)}")
    
    def load_batch_files(self):
        """Load multiple spectrum files for batch processing."""
        filepaths = filedialog.askopenfilenames(
            title="Select Spectrum Files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepaths:
            self.batch_files = [Path(f) for f in filepaths]
            self.file_info_label.config(text=f"Batch: {len(self.batch_files)} files")
            self.status_label.config(text="Ready", foreground="green")
            messagebox.showinfo("Success", f"Loaded {len(self.batch_files)} files for batch processing")
    
    def get_parameters(self):
        """Get current parameter values from UI."""
        return {
            'saturation_cutoff': self.sat_cutoff_var.get(),
            'stdev_cut_off': self.stdev_cutoff_var.get(),
            'baseline_algorithm': self.baseline_algo_var.get(),
            'baseline_lam': self.baseline_lam_var.get() * 1e10,
            'baseline_p': self.baseline_p_var.get() * 1e-6,
            'nitrogen_max_C_or_B': self.nitrogen_max_C_or_B_var.get(),
            'nitrogen_plot_fit': self.nitrogen_plot_fit_var.get(),
            'nitrogen_wn_low': self.nitrogen_wn_low_var.get(),
            'nitrogen_wn_high': self.nitrogen_wn_high_var.get(),
            'platelet_lam': self.platelet_lam_var.get(),
            'platelet_p': self.platelet_p_var.get(),
            'hydrogen_lam': self.hydrogen_lam_var.get(),
            'hydrogen_p': self.hydrogen_p_var.get() * 1e-6,
            'measure_platelets': self.measure_platelets_var.get(),
            'measure_hydrogen': self.measure_hydrogen_var.get(),
            'measure_amber': self.measure_amber_var.get(),
        }
    
    def process_spectrum(self):
        """Process the current spectrum with current parameters."""
        if self.current_spectrum is None:
            messagebox.showwarning("Warning", "Please load a spectrum first.")
            return
        
        params = self.get_parameters()
        
        try:
            # Update status
            self.status_label.config(text="Processing...", foreground="orange")
            self.file_info_label.config(text="Processing...")
            self.root.update()
            
            # Test saturation (using the same range as in the backup script)
            saturation = self.current_spectrum.test_saturation(
                900, 1400, 
                params['saturation_cutoff'], 
                params['stdev_cut_off']
            )
            
            if saturation:
                response = messagebox.askyesno(
                    "Saturation Warning", 
                    "Spectrum shows saturation. Results may be unreliable.\n\nContinue anyway?"
                )
                if not response:
                    return
            
            # Fit baseline
            self.current_spectrum.fit_baseline(
                saturation_cutoff=params['saturation_cutoff'],
                stdev_cut_off=params['stdev_cut_off']
            )
            
            # Normalize
            self.current_spectrum.normalize_diamond()
            
            # Nitrogen fit
            self.current_spectrum.Nitrogen_fit(
                plot_fit=params['nitrogen_plot_fit'],
                max_C_or_B=params['nitrogen_max_C_or_B']
            )
            
            # Measure additional peaks
            results = {'nitrogen': self.current_spectrum.nitrogen_dict.copy()}
            
            if params['measure_hydrogen']:
                try:
                    self.current_spectrum.measure_3107_peak()
                    results['hydrogen'] = {
                        'normed_area_3107': getattr(self.current_spectrum, 'normed_area_3107', None),
                        'normed_area_3085': getattr(self.current_spectrum, 'normed_area_3085', None),
                    }
                except Exception as e:
                    print(f"Warning: Hydrogen peak measurement failed: {e}")
                    results['hydrogen'] = {'error': str(e)}
            
            if params['measure_platelets']:
                try:
                    self.current_spectrum.measure_platelets_and_adjacent(
                        baseline1_param={
                            'lam': params['platelet_lam'],
                            'p': params['platelet_p']
                        }
                    )
                    results['platelets'] = {
                        'normed_area_platelet': getattr(self.current_spectrum, 'normed_area_platelet', None),
                        'normed_height_platelet': getattr(self.current_spectrum, 'normed_height_platelet', None),
                        'platelet_peak_position': getattr(self.current_spectrum, 'platelet_peak_position', None),
                    }
                except Exception as e:
                    print(f"Warning: Platelet measurement failed: {e}")
                    results['platelets'] = {'error': str(e)}
            
            if params['measure_amber']:
                try:
                    self.current_spectrum.measure_amber_center()
                    results['amber'] = {
                        'peak_positions': getattr(self.current_spectrum, 'amber_center_peak_positions', None),
                        'peak_heights_normed': getattr(self.current_spectrum, 'amber_center_peak_heights_normed', None),
                    }
                except Exception as e:
                    print(f"Warning: Amber center measurement failed: {e}")
                    results['amber'] = {'error': str(e)}
            
            self.processed_results = results
            self.update_plot()
            self.update_results_display()
            self.status_label.config(text="Complete", foreground="green")
            self.file_info_label.config(text="Processing complete")
            
            messagebox.showinfo("Success", "Spectrum processed successfully!")
            
        except Exception as e:
            self.status_label.config(text="Error", foreground="red")
            messagebox.showerror("Error", f"Processing failed:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def process_batch(self):
        """Process multiple spectra in batch mode."""
        if not self.batch_files:
            messagebox.showwarning("Warning", "Please load batch files first.")
            return
        
        params = self.get_parameters()
        results_list = []
        
        # Progress dialog
        progress_window = tk.Toplevel(self.root)
        progress_window.title("Batch Processing")
        progress_window.geometry("400x150")
        
        progress_label = ttk.Label(progress_window, text="Processing files...")
        progress_label.pack(pady=10)
        
        progress_var = tk.DoubleVar()
        progress_bar = ttk.Progressbar(
            progress_window, 
            variable=progress_var, 
            maximum=len(self.batch_files)
        )
        progress_bar.pack(fill=tk.X, padx=20, pady=10)
        
        status_label = ttk.Label(progress_window, text="")
        status_label.pack(pady=5)
        
        def process_files():
            for i, filepath in enumerate(self.batch_files):
                try:
                    status_label.config(text=f"Processing: {filepath.name}")
                    progress_window.update()
                    
                    spectrum = CSV_to_IR_Diamond_Spectrum(filepath)
                    
                    # Process spectrum
                    saturation = spectrum.test_saturation(
                        900, 1400, 
                        params['saturation_cutoff'], 
                        params['stdev_cut_off']
                    )
                    
                    # Process even if saturated (user can decide)
                    try:
                        spectrum.fit_baseline(
                            saturation_cutoff=params['saturation_cutoff'],
                            stdev_cut_off=params['stdev_cut_off']
                        )
                        spectrum.normalize_diamond()
                        spectrum.Nitrogen_fit(
                            plot_fit=False,
                            max_C_or_B=params['nitrogen_max_C_or_B']
                        )
                        
                        result = spectrum.nitrogen_dict.copy()
                        result['filename'] = filepath.name
                        result['name'] = filepath.name.split("-")[0]
                        result['saturated'] = saturation
                        
                        if params['measure_hydrogen']:
                            spectrum.measure_3107_peak()
                            result['Normed_3107_Area'] = getattr(spectrum, 'normed_area_3107', None)
                            result['Normed_3085_Area'] = getattr(spectrum, 'normed_area_3085', None)
                        
                        if params['measure_platelets']:
                            spectrum.measure_platelets_and_adjacent(
                                baseline1_param={
                                    'lam': params['platelet_lam'],
                                    'p': params['platelet_p']
                                }
                            )
                            result['Normed_Platelet_Area'] = getattr(spectrum, 'normed_area_platelet', None)
                            result['Normed_Platelet_Height'] = getattr(spectrum, 'normed_height_platelet', None)
                            result['platelet_peak_position'] = getattr(spectrum, 'platelet_peak_position', None)
                        
                        if params['measure_amber']:
                            spectrum.measure_amber_center()
                            result['amber_center_peak_positions'] = getattr(spectrum, 'amber_center_peak_positions', None)
                        
                        results_list.append(result)
                    except Exception as e:
                        print(f"Error processing {filepath.name}: {str(e)}")
                    
                    progress_var.set(i + 1)
                    progress_window.update()
                    
                except Exception as e:
                    print(f"Error processing {filepath.name}: {str(e)}")
                    continue
            
            progress_window.destroy()
            
            # Create results DataFrame
            if results_list:
                df = pd.DataFrame(results_list)
                self.processed_results = {'batch_results': df}
                self.update_results_display()
                messagebox.showinfo(
                    "Batch Complete", 
                    f"Processed {len(results_list)}/{len(self.batch_files)} files successfully."
                )
            else:
                messagebox.showwarning("Warning", "No files were processed successfully.")
        
        # Run in separate thread to avoid blocking
        thread = threading.Thread(target=process_files)
        thread.daemon = True
        thread.start()
    
    def update_plot(self):
        """Update the spectrum plot."""
        if self.ax is None or self.canvas is None:
            return
        self.ax.clear()
        if self.current_spectrum is None:
            self.ax.text(0.5, 0.5, 'No spectrum loaded', 
                        ha='center', va='center', transform=self.ax.transAxes)
            self.canvas.draw()
            return
        
        plot_type = self.plot_type_var.get()
        
        if plot_type == "Raw":
            self.ax.plot(self.current_spectrum.X, self.current_spectrum.Y, 'b-', linewidth=0.5)
            self.ax.set_ylabel('Absorbance')
            if hasattr(self.current_spectrum, 'baseline'):
                self.ax.plot(self.current_spectrum.X, self.current_spectrum.baseline, 'r--', 
                           linewidth=1, label='Baseline')
                self.ax.legend()
        
        elif plot_type == "Baseline Corrected":
            if hasattr(self.current_spectrum, 'baseline'):
                corrected = self.current_spectrum.Y - self.current_spectrum.baseline
                self.ax.plot(self.current_spectrum.X, corrected, 'g-', linewidth=0.5)
                self.ax.set_ylabel('Baseline Corrected Absorbance')
            else:
                self.ax.text(0.5, 0.5, 'Baseline not fitted yet', 
                           ha='center', va='center', transform=self.ax.transAxes)
        
        elif plot_type == "Normalized":
            if hasattr(self.current_spectrum, 'normalized_spectrum'):
                self.ax.plot(
                    self.current_spectrum.normalized_spectrum.X,
                    self.current_spectrum.normalized_spectrum.Y,
                    'm-', linewidth=0.5
                )
                self.ax.set_ylabel('Normalized Absorbance (1 cm)')
            else:
                self.ax.text(0.5, 0.5, 'Spectrum not normalized yet', 
                           ha='center', va='center', transform=self.ax.transAxes)
        
        elif plot_type == "Nitrogen Fit":
            if hasattr(self.current_spectrum, 'nitrogen_plot_fit_params'):
                params = self.current_spectrum.nitrogen_plot_fit_params
                wn_array = params['wn_array']
                spec_intensity = params['spec_intensity']
                fit_params = params['fit_params']
                fit_component_df = params['fit_component_df']
                
                self.ax.plot(wn_array, spec_intensity, 'b-', linewidth=1, label='Spectrum')
                
                fit_comp = fit_component_df * fit_params
                model_spectrum = fit_comp.sum(axis=1, numeric_only=True)
                self.ax.plot(wn_array, model_spectrum, 'r-', linewidth=1.5, label='Fit')
                
                # Plot individual components
                for col in fit_component_df.columns[:6]:
                    if fit_params[fit_component_df.columns.get_loc(col)] > 0:
                        self.ax.plot(wn_array, fit_comp[col], '--', alpha=0.6, label=col)
                
                self.ax.set_xlabel('Wavenumber (cm⁻¹)')
                self.ax.set_ylabel('Absorptivity (cm⁻¹)')
                self.ax.legend()
            else:
                self.ax.text(0.5, 0.5, 'Nitrogen fit not available', 
                           ha='center', va='center', transform=self.ax.transAxes)
        
        self.ax.set_xlabel('Wavenumber (cm⁻¹)')
        self.ax.grid(True, alpha=0.3)
        self.canvas.draw()
    
    def update_results_display(self):
        """Update the results text display."""
        self.results_text.config(state=tk.NORMAL)
        self.results_text.delete(1.0, tk.END)
        
        if self.processed_results is None:
            self.results_text.insert(tk.END, "No results available.")
        elif 'batch_results' in self.processed_results:
            # Batch results
            df = self.processed_results['batch_results']
            self.results_text.insert(tk.END, f"Batch Processing Results\n")
            self.results_text.insert(tk.END, f"{'='*60}\n\n")
            self.results_text.insert(tk.END, df.to_string())
        else:
            # Single spectrum results
            self.results_text.insert(tk.END, "Processing Results\n")
            self.results_text.insert(tk.END, f"{'='*60}\n\n")
            
            if 'nitrogen' in self.processed_results:
                self.results_text.insert(tk.END, "Nitrogen Analysis:\n")
                for key, value in self.processed_results['nitrogen'].items():
                    self.results_text.insert(tk.END, f"  {key}: {value}\n")
                self.results_text.insert(tk.END, "\n")
            
            if 'hydrogen' in self.processed_results:
                self.results_text.insert(tk.END, "Hydrogen Peaks:\n")
                for key, value in self.processed_results['hydrogen'].items():
                    self.results_text.insert(tk.END, f"  {key}: {value}\n")
                self.results_text.insert(tk.END, "\n")
            
            if 'platelets' in self.processed_results:
                self.results_text.insert(tk.END, "Platelet Peaks:\n")
                for key, value in self.processed_results['platelets'].items():
                    self.results_text.insert(tk.END, f"  {key}: {value}\n")
                self.results_text.insert(tk.END, "\n")
            
            if 'amber' in self.processed_results:
                self.results_text.insert(tk.END, "Amber Center:\n")
                for key, value in self.processed_results['amber'].items():
                    self.results_text.insert(tk.END, f"  {key}: {value}\n")
        
        self.results_text.config(state=tk.DISABLED)
    
    def export_results(self):
        """Export results to CSV file."""
        if self.processed_results is None:
            messagebox.showwarning("Warning", "No results to export.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Results",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                if 'batch_results' in self.processed_results:
                    self.processed_results['batch_results'].to_csv(filepath, index=False)
                else:
                    # Convert single results to DataFrame
                    data = {}
                    for category, values in self.processed_results.items():
                        if isinstance(values, dict):
                            data.update(values)
                    df = pd.DataFrame([data])
                    df.to_csv(filepath, index=False)
                
                messagebox.showinfo("Success", f"Results exported to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Export failed:\n{str(e)}")
    
    def save_plot(self):
        """Save the current plot to a file."""
        if self.fig is None:
            messagebox.showwarning("Warning", "Plot not ready yet. Wait a moment and try again.")
            return
        if self.current_spectrum is None:
            messagebox.showwarning("Warning", "No spectrum loaded to save.")
            return
        
        filepath = filedialog.asksaveasfilename(
            title="Save Plot",
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
        )
        
        if filepath:
            try:
                self.fig.savefig(filepath, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filepath}")
            except Exception as e:
                messagebox.showerror("Error", f"Save failed:\n{str(e)}")


def main():
    """Main entry point for the GUI application."""
    sys.excepthook = _gui_excepthook
    root = tk.Tk()
    try:
        app = DiamondFTIRGUI(root)
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
