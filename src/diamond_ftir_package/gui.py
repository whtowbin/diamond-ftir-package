"""Diamond FTIR desktop app (tkinter).

A thin front end over :mod:`diamond_ftir_package.pipeline`: every control here maps to a
field of :class:`~diamond_ftir_package.params.AnalysisParams`, so the app, the command line
and Python scripts always give the same numbers for the same settings.

Start it with ``diamond-ftir-gui`` (or ``python -m diamond_ftir_package.gui``).
"""

from __future__ import annotations

import json
import os
import queue
import sys
import threading
import tkinter as tk
from dataclasses import fields
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import matplotlib
import pandas as pd

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from .params import CHOICES, AnalysisParams
from .pipeline import SUPPORTED_SUFFIXES, analyze_file, collect_files
from .plotting import plot_spectrum_fit

SECTION_TITLES = {
    "diamond": "Thickness & saturation",
    "nitrogen": "Nitrogen",
    "hydrogen": "Hydrogen (3107)",
    "platelet": "Platelets / 1405",
}
RESULT_COLUMNS = (
    "Filename",
    "Status",
    "Total_N ppm",
    "A_Nitrogen ppm",
    "B_Nitrogen ppm",
    "C_Nitrogen ppm",
    "B_percent",
    "Normed_3107_Area",
    "Normed_Platelet_Area",
)


class Tooltip:
    """Small hover tip; explains what a setting does in plain language."""

    def __init__(self, widget: tk.Widget, text: str):
        self.widget, self.text, self.tip = widget, text, None
        widget.bind("<Enter>", self._show)
        widget.bind("<Leave>", self._hide)

    def _show(self, _event=None):
        x, y = self.widget.winfo_rootx() + 20, self.widget.winfo_rooty() + 24
        self.tip = tk.Toplevel(self.widget)
        self.tip.wm_overrideredirect(True)
        self.tip.wm_geometry(f"+{x}+{y}")
        ttk.Label(
            self.tip,
            text=self.text,
            background="#ffffe0",
            relief="solid",
            wraplength=320,
        ).pack()

    def _hide(self, _event=None):
        if self.tip:
            self.tip.destroy()
            self.tip = None


def _format_value(value: Any) -> str:
    if isinstance(value, tuple):
        return ", ".join(str(v) for v in value)
    return str(value)


def _parse_value(default: Any, text: str) -> Any:
    """Turn the text in an entry box back into the type of the field's default."""
    text = text.strip()
    if isinstance(default, tuple):
        return tuple(
            _parse_number(t) for t in text.replace(";", ",").split(",") if t.strip()
        )
    if isinstance(default, bool):  # before int: bool is a subclass of int
        if text.lower() in ("true", "yes", "1", "on"):
            return True
        if text.lower() in ("false", "no", "0", "off"):
            return False
        raise ValueError(f"expected True or False, got {text!r}")
    if isinstance(default, str):
        return text
    return _parse_number(text, prefer_int=isinstance(default, int))


def _parse_number(text: str, prefer_int: bool = False) -> int | float:
    text = text.strip()
    if prefer_int:
        try:
            return int(text)
        except ValueError:
            pass
    return float(text)


class ParamsForm:
    """Entry boxes generated from the parameter dataclasses; stays in sync with the library."""

    def __init__(self, parent: ttk.Notebook, params: AnalysisParams):
        self.params = params
        self.vars: dict[tuple[str, str], tk.StringVar] = {}
        self.parent = parent
        for section, title in SECTION_TITLES.items():
            frame = ttk.Frame(parent, padding=10)
            parent.add(frame, text=title)
            self._build_section(frame, section)

    def _build_section(self, frame: ttk.Frame, section: str) -> None:
        sub = getattr(self.params, section)
        for row, f in enumerate(fields(sub)):
            label = ttk.Label(frame, text=f.name.replace("_", " "))
            label.grid(row=row, column=0, sticky="w", padx=(0, 12), pady=2)
            var = tk.StringVar(value=_format_value(getattr(sub, f.name)))
            self.vars[(section, f.name)] = var
            default = getattr(sub, f.name)
            choices = CHOICES.get(f.name) or (
                ("True", "False") if isinstance(default, bool) else None
            )
            if choices:
                widget: tk.Widget = ttk.Combobox(
                    frame, textvariable=var, values=choices, state="readonly", width=18
                )
            else:
                widget = ttk.Entry(frame, textvariable=var, width=20)
            widget.grid(row=row, column=1, sticky="w", pady=2)
            description = f.metadata.get("description", "")
            Tooltip(label, description)
            Tooltip(widget, description)

    def read(self) -> AnalysisParams:
        """Current form values as parameters. Raises ValueError naming the bad field."""
        params = AnalysisParams()
        for (section, name), var in self.vars.items():
            sub = getattr(params, section)
            default = getattr(sub, name)
            try:
                setattr(sub, name, _parse_value(default, var.get()))
            except ValueError as e:
                raise ValueError(
                    f"'{name.replace('_', ' ')}' in {SECTION_TITLES[section]}: {e}"
                ) from e
        return params

    def write(self, params: AnalysisParams) -> None:
        for (section, name), var in self.vars.items():
            var.set(_format_value(getattr(getattr(params, section), name)))


def _ensure_tcl_paths() -> None:
    """Point Tcl/Tk at the base Python's libraries when running inside a virtual env.

    uv-managed Pythons in a venv can fail with "Cannot find a usable init.tcl" because Tcl
    searches relative to the venv, not the base install. Existing settings are respected.
    """
    base = Path(sys.base_prefix) / "lib"
    for var, pattern, marker in (
        ("TCL_LIBRARY", "tcl[0-9]*", "init.tcl"),
        ("TK_LIBRARY", "tk[0-9]*", "tk.tcl"),
    ):
        if var in os.environ:
            continue
        for candidate in sorted(base.glob(pattern), reverse=True):
            if (candidate / marker).exists():
                os.environ[var] = str(candidate)
                break


class App(tk.Tk):
    def __init__(self):
        _ensure_tcl_paths()
        super().__init__()
        self.title("Diamond FTIR")
        self.geometry("1200x760")
        self.spectra: dict[str, Any] = {}
        self.rows: list[dict[str, Any]] = []
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.amber_band_params = AnalysisParams().amber  # edited via settings JSON only

        self._build_layout()
        self.after(100, self._poll_events)

    # ---- layout -----------------------------------------------------------
    def _build_layout(self) -> None:
        top = ttk.Frame(self, padding=8)
        top.pack(fill="x")
        ttk.Button(top, text="Open file…", command=self.open_file).pack(side="left")
        ttk.Button(top, text="Open folder…", command=self.open_folder).pack(
            side="left", padx=4
        )
        self.path_var = tk.StringVar(value="No file selected")
        ttk.Label(top, textvariable=self.path_var).pack(side="left", padx=12)
        self.run_button = ttk.Button(top, text="Run analysis", command=self.run)
        self.run_button.pack(side="right")

        body = ttk.PanedWindow(self, orient="horizontal")
        body.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        left = ttk.Frame(body)
        body.add(left, weight=1)
        self._build_basic(left)
        self.notebook = ttk.Notebook(left)
        self.notebook.pack(fill="both", expand=True, pady=(8, 0))
        self.form = ParamsForm(self.notebook, AnalysisParams())
        buttons = ttk.Frame(left)
        buttons.pack(fill="x", pady=4)
        ttk.Button(buttons, text="Reset to defaults", command=self.reset_defaults).pack(
            side="left"
        )
        ttk.Button(buttons, text="Save settings…", command=self.save_settings).pack(
            side="left", padx=4
        )
        ttk.Button(buttons, text="Load settings…", command=self.load_settings).pack(
            side="left"
        )

        right = ttk.Frame(body)
        body.add(right, weight=3)
        self.table = ttk.Treeview(
            right, columns=RESULT_COLUMNS, show="headings", height=8
        )
        for col in RESULT_COLUMNS:
            self.table.heading(col, text=col)
            self.table.column(col, width=110 if col != "Status" else 160, anchor="w")
        self.table.pack(fill="x")
        self.table.bind("<<TreeviewSelect>>", self._on_select)
        ttk.Button(right, text="Export results to CSV…", command=self.export_csv).pack(
            anchor="e", pady=4
        )

        self.figure = Figure(figsize=(7, 5), constrained_layout=True)
        self.canvas = FigureCanvasTkAgg(self.figure, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self, textvariable=self.status_var, relief="sunken", anchor="w").pack(
            fill="x"
        )
        self.progress = ttk.Progressbar(self, mode="determinate")
        self.progress.pack(fill="x")

    def _build_basic(self, parent: ttk.Frame) -> None:
        box = ttk.LabelFrame(parent, text="What to measure", padding=8)
        box.pack(fill="x")
        defaults = AnalysisParams()
        self.run_vars = {
            "run_nitrogen": tk.BooleanVar(value=defaults.run_nitrogen),
            "run_hydrogen": tk.BooleanVar(value=defaults.run_hydrogen),
            "run_platelets": tk.BooleanVar(value=defaults.run_platelets),
            "run_amber": tk.BooleanVar(value=defaults.run_amber),
        }
        labels = {
            "run_nitrogen": "Nitrogen (A, B, C centres)",
            "run_hydrogen": "Hydrogen (3107 / 3085)",
            "run_platelets": "Platelets and 1405 peak",
            "run_amber": "Amber centres",
        }
        for key, var in self.run_vars.items():
            ttk.Checkbutton(box, text=labels[key], variable=var).pack(anchor="w")

    # ---- settings ---------------------------------------------------------
    def current_params(self) -> AnalysisParams:
        params = self.form.read()
        params.amber = self.amber_band_params
        for key, var in self.run_vars.items():
            setattr(params, key, var.get())
        return params

    def _apply_params(self, params: AnalysisParams) -> None:
        self.form.write(params)
        self.amber_band_params = params.amber
        for key, var in self.run_vars.items():
            var.set(getattr(params, key))

    def reset_defaults(self) -> None:
        self._apply_params(AnalysisParams())

    def save_settings(self) -> None:
        try:
            params = self.current_params()
        except ValueError as e:
            messagebox.showerror("Invalid setting", str(e))
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("Settings", "*.json")]
        )
        if path:
            Path(path).write_text(json.dumps(params.to_dict(), indent=2))

    def load_settings(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Settings", "*.json")])
        if path:
            try:
                self._apply_params(
                    AnalysisParams.from_dict(json.loads(Path(path).read_text()))
                )
            except (ValueError, TypeError, KeyError) as e:
                messagebox.showerror("Could not load settings", str(e))

    # ---- running ----------------------------------------------------------
    def open_file(self) -> None:
        patterns = " ".join(f"*{s}" for s in SUPPORTED_SUFFIXES)
        path = filedialog.askopenfilename(
            filetypes=[("Spectra", patterns), ("All files", "*.*")]
        )
        if path:
            self.path_var.set(path)

    def open_folder(self) -> None:
        path = filedialog.askdirectory()
        if path:
            self.path_var.set(path)

    def run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        path = self.path_var.get()
        if not Path(path).exists():
            messagebox.showinfo(
                "Nothing to analyse", "Choose a spectrum file or folder first."
            )
            return
        try:
            params = self.current_params()
        except ValueError as e:
            messagebox.showerror("Invalid setting", str(e))
            return
        files = collect_files(path)
        if not files:
            messagebox.showinfo("Nothing to analyse", "No CSV, SPA or SPC files found.")
            return

        self.table.delete(*self.table.get_children())
        self.spectra.clear()
        self.rows.clear()
        self.progress.configure(maximum=len(files), value=0)
        self.run_button.state(["disabled"])
        # The worker never touches tkinter: it only puts events on the queue.
        self.worker = threading.Thread(
            target=self._work, args=(files, params), daemon=True
        )
        self.worker.start()

    def _work(self, files: list[Path], params: AnalysisParams) -> None:
        for i, file in enumerate(files, start=1):
            spectrum, row = analyze_file(file, params)
            self.events.put(("result", i, len(files), file.name, spectrum, row))
        self.events.put(("done",))

    def _poll_events(self) -> None:
        try:
            while True:
                event = self.events.get_nowait()
                if event[0] == "result":
                    _, i, n, name, spectrum, row = event
                    self._add_result(name, spectrum, row)
                    self.progress.configure(value=i)
                    self.status_var.set(f"Analysed {i} of {n}: {name}")
                else:
                    self.run_button.state(["!disabled"])
                    failed = sum(r["Status"] != "OK" for r in self.rows)
                    self.status_var.set(
                        f"Done. {len(self.rows) - failed} succeeded, {failed} failed."
                    )
        except queue.Empty:
            pass
        self.after(100, self._poll_events)

    def _add_result(self, name: str, spectrum: Any, row: dict[str, Any]) -> None:
        self.rows.append(row)
        if spectrum is not None:
            self.spectra[name] = spectrum
        values = [self._cell(row.get(col)) for col in RESULT_COLUMNS]
        self.table.insert("", "end", iid=str(len(self.rows) - 1), values=values)
        if len(self.rows) == 1:
            self.table.selection_set("0")

    @staticmethod
    def _cell(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return "" if value is None else str(value)

    # ---- output -----------------------------------------------------------
    def export_csv(self) -> None:
        if not self.rows:
            messagebox.showinfo("No results", "Run an analysis first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", filetypes=[("CSV", "*.csv")]
        )
        if path:
            pd.DataFrame(self.rows).to_csv(path, index=False)
            self.status_var.set(f"Saved {len(self.rows)} rows to {path}")

    def _on_select(self, _event=None) -> None:
        selected = self.table.selection()
        if not selected:
            return
        row = self.rows[int(selected[0])]
        spectrum = self.spectra.get(row["Filename"])
        self.figure.clear()
        if spectrum is None or getattr(spectrum, "baseline", None) is None:
            ax = self.figure.add_subplot(111)
            ax.text(
                0.5,
                0.5,
                row.get("Status", "No data"),
                ha="center",
                va="center",
                wrap=True,
            )
            ax.set_axis_off()
        else:
            self._plot_spectrum(spectrum, row["Filename"])
        self.canvas.draw_idle()

    def _plot_spectrum(self, spectrum: Any, name: str) -> None:
        plot_spectrum_fit(self.figure, spectrum, name)


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()


__all__ = ["App", "ParamsForm", "main"]
