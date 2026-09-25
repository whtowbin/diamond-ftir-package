"""
Minimal Diamond FTIR GUI — build up from pure Tk to isolate SEGV.

Load/process (single or batch), plot with type selector, export CSV and plot.
Plotting uses lazy matplotlib import. Key parameters (saturation, max C/B, amber) are adjustable.

Launch: python -m diamond_ftir_package.gui_launcher_simple
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from pathlib import Path

import pandas as pd


def main():
    root = tk.Tk()
    root.title("Diamond FTIR (simple)")
    root.geometry("800x560")

    # State
    current_spectrum = None
    current_path = None
    last_single_result = None  # dict for single-file export
    batch_files = []
    batch_results = []  # list of dicts
    plot_canvas = None
    plot_fig = None
    plot_ax = None
    plot_toolbar = None
    plot_container = None

    # Header
    ttk.Label(root, text="Diamond FTIR — load, process, plot & export", font=("", 14)).pack(pady=6)

    # Parameters row
    params_frame = ttk.LabelFrame(root, text="Parameters", padding=6)
    params_frame.pack(fill=tk.X, padx=8, pady=2)
    p_row = ttk.Frame(params_frame)
    p_row.pack(fill=tk.X)
    ttk.Label(p_row, text="Saturation cutoff:").pack(side=tk.LEFT, padx=(0, 2))
    sat_var = tk.DoubleVar(value=2.5)
    ttk.Spinbox(p_row, from_=0.5, to=10, increment=0.1, textvariable=sat_var, width=6).pack(side=tk.LEFT, padx=2)
    ttk.Label(p_row, text="Stdev cutoff:").pack(side=tk.LEFT, padx=(8, 2))
    stdev_var = tk.DoubleVar(value=0.5)
    ttk.Spinbox(p_row, from_=0.01, to=2, increment=0.01, textvariable=stdev_var, width=6).pack(side=tk.LEFT, padx=2)
    ttk.Label(p_row, text="Max C/B:").pack(side=tk.LEFT, padx=(8, 2))
    max_cb_var = tk.DoubleVar(value=0.01)
    ttk.Spinbox(p_row, from_=0.001, to=0.1, increment=0.001, textvariable=max_cb_var, width=6).pack(side=tk.LEFT, padx=2)
    measure_amber_var = tk.BooleanVar(value=False)
    ttk.Checkbutton(p_row, text="Measure amber center", variable=measure_amber_var).pack(side=tk.LEFT, padx=8)

    # Status
    status = ttk.Label(root, text="Ready. No file loaded.")
    status.pack(pady=2)

    # Results and plot
    paned = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
    paned.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    results_frame = ttk.LabelFrame(paned, text="Results", padding=4)
    results_text = ScrolledText(results_frame, wrap=tk.WORD, height=14)
    results_text.pack(fill=tk.BOTH, expand=True)
    paned.add(results_frame, weight=1)

    plot_frame = ttk.LabelFrame(paned, text="Plot", padding=4)
    paned.add(plot_frame, weight=1)

    # Plot type and placeholder
    plot_top = ttk.Frame(plot_frame)
    plot_top.pack(fill=tk.X)
    ttk.Label(plot_top, text="Type:").pack(side=tk.LEFT, padx=2)
    plot_type_var = tk.StringVar(value="Raw")
    plot_type_combo = ttk.Combobox(
        plot_top, textvariable=plot_type_var,
        values=["Raw", "Baseline corrected", "Normalized", "Nitrogen fit"],
        state="readonly", width=18,
    )
    plot_type_combo.pack(side=tk.LEFT, padx=2)

    plot_container = ttk.Frame(plot_frame)
    plot_container.pack(fill=tk.BOTH, expand=True)
    plot_placeholder = ttk.Label(
        plot_container, text="Load a spectrum, then click 'Show plot'.", font=("", 10),
    )
    plot_placeholder.pack(expand=True)

    results_text.insert(tk.END, "Load a spectrum and click Process to see results here.")
    results_text.config(state=tk.DISABLED)

    def set_results(content):
        results_text.config(state=tk.NORMAL)
        results_text.delete(1.0, tk.END)
        results_text.insert(tk.END, content)
        results_text.config(state=tk.DISABLED)

    def result_dict_from_spec(spec, filepath=None, saturated=False):
        """Build one result dict for export from a processed spectrum."""
        d = {}
        if filepath:
            d["filename"] = Path(filepath).name
            d["name"] = Path(filepath).name.split("-")[0]
        d["saturated"] = saturated
        if hasattr(spec, "nitrogen_dict") and spec.nitrogen_dict:
            d.update(spec.nitrogen_dict)
        d["Normed_3107_Area"] = getattr(spec, "normed_area_3107", None)
        d["Normed_3085_Area"] = getattr(spec, "normed_area_3085", None)
        d["Normed_Platelet_Area"] = getattr(spec, "normed_area_platelet", None)
        d["Normed_Platelet_Height"] = getattr(spec, "normed_height_platelet", None)
        d["platelet_peak_position"] = getattr(spec, "platelet_peak_position", None)
        if measure_amber_var.get():
            d["amber_center_peak_positions"] = getattr(spec, "amber_center_peak_positions", None)
        return d

    def run_process(spec, filepath=None):
        """Run full pipeline on spec; returns result dict or None on error."""
        sat_c, stdev_c = sat_var.get(), stdev_var.get()
        saturated = spec.test_saturation(900, 1400, sat_c, stdev_c)
        spec.fit_baseline(saturation_cutoff=sat_c, stdev_cut_off=stdev_c)
        spec.normalize_diamond()
        spec.Nitrogen_fit(plot_fit=False, max_C_or_B=max_cb_var.get())
        spec.measure_3107_peak()
        spec.measure_platelets_and_adjacent()
        if measure_amber_var.get():
            spec.measure_amber_center()
        return result_dict_from_spec(spec, filepath, saturated)

    def on_load():
        nonlocal current_spectrum, current_path, last_single_result, batch_results
        path = filedialog.askopenfilename(
            title="Select spectrum CSV", filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not path:
            return
        current_path = Path(path)
        last_single_result = None
        batch_results = []
        try:
            from .LoadCSV import CSV_to_IR_Diamond_Spectrum
            current_spectrum = CSV_to_IR_Diamond_Spectrum(path)
            status.config(text=f"Loaded: {current_path.name}")
            set_results(f"Loaded: {current_path.name}\n\nClick Process to run analysis.")
        except Exception as e:
            current_spectrum = None
            status.config(text="Load failed")
            set_results(f"Load error:\n{e}")
            messagebox.showerror("Load error", str(e))

    def on_process():
        nonlocal current_spectrum, last_single_result, batch_results
        if current_spectrum is None:
            messagebox.showwarning("No file", "Load a spectrum first.")
            return
        last_single_result = None
        batch_results = []
        status.config(text="Processing…")
        root.update()
        try:
            spec = current_spectrum
            saturated = spec.test_saturation(900, 1400, sat_var.get(), stdev_var.get())
            if saturated:
                set_results("Spectrum appears saturated. Processing anyway.\n\n")
            res = run_process(spec, current_path)
            last_single_result = res
            lines = ["Results", "=" * 50, ""]
            for k, v in res.items():
                lines.append(f"  {k}: {v}")
            set_results("\n".join(lines))
            status.config(text="Done.")
        except Exception as e:
            import traceback
            set_results(f"Processing error:\n\n{traceback.format_exc()}")
            status.config(text="Error")
            messagebox.showerror("Processing error", str(e))

    def on_load_batch():
        nonlocal batch_files, batch_results, last_single_result
        paths = filedialog.askopenfilenames(
            title="Select spectrum CSVs", filetypes=[("CSV", "*.csv"), ("All", "*.*")],
        )
        if not paths:
            return
        batch_files[:] = [Path(p) for p in paths]
        batch_results = []
        last_single_result = None
        status.config(text=f"Batch: {len(batch_files)} files loaded.")
        set_results(f"Loaded {len(batch_files)} files.\nClick 'Process batch' to run.")

    def on_process_batch():
        nonlocal batch_results, last_single_result
        if not batch_files:
            messagebox.showwarning("No files", "Load batch first.")
            return
        last_single_result = None
        batch_results = []
        status.config(text="Batch processing…")
        root.update()
        from .LoadCSV import CSV_to_IR_Diamond_Spectrum
        for i, path in enumerate(batch_files):
            status.config(text=f"Processing {i+1}/{len(batch_files)}: {path.name}")
            root.update()
            try:
                spec = CSV_to_IR_Diamond_Spectrum(path)
                saturated = spec.test_saturation(900, 1400, sat_var.get(), stdev_var.get())
                res = run_process(spec, path)
                batch_results.append(res)
            except Exception as e:
                batch_results.append({"filename": path.name, "name": path.name.split("-")[0], "error": str(e)})
        status.config(text="Batch done.")
        if batch_results:
            df = pd.DataFrame(batch_results)
            set_results(df.to_string())
        else:
            set_results("No results.")

    def on_export_csv():
        if last_single_result:
            path = filedialog.asksaveasfilename(
                title="Save results CSV", defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            )
            if path:
                pd.DataFrame([last_single_result]).to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Saved to {path}")
                return
        if batch_results:
            path = filedialog.asksaveasfilename(
                title="Save batch results CSV", defaultextension=".csv",
                filetypes=[("CSV", "*.csv"), ("All", "*.*")],
            )
            if path:
                pd.DataFrame(batch_results).to_csv(path, index=False)
                messagebox.showinfo("Saved", f"Saved {len(batch_results)} rows to {path}")
                return
        messagebox.showwarning("No data", "Process a spectrum or batch first.")

    def ensure_plot_widgets():
        nonlocal plot_canvas, plot_fig, plot_ax, plot_toolbar, plot_container
        if plot_canvas is not None:
            return
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
        except Exception as e:
            messagebox.showerror("Plot error", f"Could not load matplotlib: {e}")
            return
        for w in plot_container.winfo_children():
            w.destroy()
        plot_fig, plot_ax = plt.subplots(figsize=(5, 4))
        plot_fig.patch.set_facecolor("white")
        plot_canvas = FigureCanvasTkAgg(plot_fig, master=plot_container)
        plot_canvas.draw()
        plot_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        plot_toolbar = NavigationToolbar2Tk(plot_canvas, plot_container)
        plot_toolbar.update()
        plot_toolbar.pack(fill=tk.X)
        plot_type_combo.bind("<<ComboboxSelected>>", lambda e: update_plot())

    def update_plot():
        nonlocal plot_canvas, plot_fig, plot_ax, current_spectrum, current_path
        if current_spectrum is None:
            messagebox.showwarning("No file", "Load a spectrum first.")
            return
        ensure_plot_widgets()
        if plot_canvas is None:
            return
        spec = current_spectrum
        ptype = plot_type_var.get()
        plot_ax.clear()
        if ptype == "Raw":
            plot_ax.plot(spec.X, spec.Y, "b-", linewidth=0.6, label="Raw")
            if getattr(spec, "baseline", None) is not None:
                plot_ax.plot(spec.X, spec.baseline, "r--", linewidth=0.8, label="Baseline")
        elif ptype == "Baseline corrected":
            if getattr(spec, "baseline", None) is not None:
                plot_ax.plot(spec.X, spec.Y - spec.baseline, "g-", linewidth=0.6, label="Baseline corrected")
            else:
                plot_ax.plot(spec.X, spec.Y, "b-", linewidth=0.6)
                plot_ax.set_title("(Baseline not fitted)")
        elif ptype == "Normalized":
            if getattr(spec, "normalized_spectrum", None) is not None:
                n = spec.normalized_spectrum
                plot_ax.plot(n.X, n.Y, "m-", linewidth=0.5, label="Normalized")
            else:
                plot_ax.text(0.5, 0.5, "Run Process first", ha="center", va="center", transform=plot_ax.transAxes)
        elif ptype == "Nitrogen fit":
            if hasattr(spec, "nitrogen_plot_fit_params"):
                p = spec.nitrogen_plot_fit_params
                wn, intensity = p["wn_array"], p["spec_intensity"]
                params, comp_df = p["fit_params"], p["fit_component_df"]
                plot_ax.plot(wn, intensity, "b-", linewidth=1, label="Spectrum")
                fit_curve = (comp_df * params).sum(axis=1, numeric_only=True)
                plot_ax.plot(wn, fit_curve, "r-", linewidth=1.2, label="Fit")
                for i, col in enumerate(comp_df.columns[:6]):
                    if i < len(params) and params[i] > 0:
                        plot_ax.plot(wn, comp_df[col].values * params[i], "--", alpha=0.7, label=col)
                plot_ax.set_ylabel("Absorptivity (cm⁻¹)")
            else:
                plot_ax.text(0.5, 0.5, "Run Process first", ha="center", va="center", transform=plot_ax.transAxes)
        plot_ax.set_xlabel("Wavenumber (cm⁻¹)")
        if ptype != "Nitrogen fit":
            plot_ax.set_ylabel("Absorbance")
        plot_ax.legend()
        plot_ax.grid(True, alpha=0.3)
        plot_ax.set_title(current_path.name if current_path else "Spectrum")
        plot_fig.tight_layout()
        plot_canvas.draw()

    def on_save_plot():
        if plot_fig is None:
            messagebox.showwarning("No plot", "Show a plot first.")
            return
        path = filedialog.asksaveasfilename(
            title="Save plot", defaultextension=".png",
            filetypes=[("PNG", "*.png"), ("PDF", "*.pdf"), ("All", "*.*")],
        )
        if path:
            try:
                plot_fig.savefig(path, dpi=150, bbox_inches="tight")
                messagebox.showinfo("Saved", f"Plot saved to {path}")
            except Exception as e:
                messagebox.showerror("Save error", str(e))

    # Buttons
    btn_frame = ttk.Frame(root)
    btn_frame.pack(pady=6)
    ttk.Button(btn_frame, text="Load (single)", command=on_load).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Load batch", command=on_load_batch).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Process", command=on_process).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Process batch", command=on_process_batch).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Show plot", command=update_plot).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Export CSV", command=on_export_csv).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Save plot", command=on_save_plot).pack(side=tk.LEFT, padx=2)
    ttk.Button(btn_frame, text="Quit", command=root.destroy).pack(side=tk.LEFT, padx=2)

    root.mainloop()


if __name__ == "__main__":
    main()
