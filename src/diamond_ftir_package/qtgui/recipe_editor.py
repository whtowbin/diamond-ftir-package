"""Recipe editor: define baselines, single peaks and peak clusters, with a live preview.

Generic: works for any x/y spectrum (IR, Raman, UV-VIS). Give it a spectrum with
``set_spectrum(x, y, thickness)``; every edit re-runs the selected feature through the same
engine the batch and map analysis use, so the preview is the analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QMessageBox,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..core import (
    BASELINE_METHODS,
    PEAK_MODELS,
    BaselineStep,
    Feature,
    PeakDef,
    Recipe,
    builtin_recipes,
    resolve_recipe,
    run_feature,
)
from .forms import DataclassForm
from .spectrum_view import SpectrumView

CHOICES = {
    "method": BASELINE_METHODS,
    "model": PEAK_MODELS,
    "normalize_by": ("thickness", "none"),
}


class RecipeEditor(QWidget):
    """Edit a Recipe; ``recipe_changed`` fires after every edit."""

    recipe_changed = Signal()

    def __init__(
        self, recipe: Recipe | None = None, parent=None, x_label="Wavenumber (cm⁻¹)"
    ):
        super().__init__(parent)
        self.recipe = recipe or Recipe(name="new recipe")
        self.x = self.y = None
        self.thickness = None
        self.current = None  # (kind, feature_index, sub_index)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Recipe"])
        self.tree.currentItemChanged.connect(self._select)
        buttons = QVBoxLayout()
        for text, slot in (
            ("Add single peak", lambda: self._add_feature(1)),
            ("Add peak cluster", lambda: self._add_feature(3)),
            ("Add baseline step", self._add_step),
            ("Add peak to feature", self._add_peak),
            ("Remove selected", self._remove),
            ("Load recipe…", self._load),
            ("Load built-in…", self._load_builtin),
            ("Save recipe…", self._save),
        ):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch()

        left = QWidget()
        ll = QHBoxLayout(left)
        ll.addWidget(self.tree, 3)
        ll.addLayout(buttons, 1)

        self.form_holder = QVBoxLayout()
        self.form_holder.addWidget(QLabel("Select an item to edit it."))
        middle = QWidget()
        middle.setLayout(self.form_holder)

        self.view = SpectrumView(x_label=x_label)
        self.view.region_changed.connect(self._region_dragged)
        self.results = QTableWidget(0, 2)
        self.results.setHorizontalHeaderLabels(["output", "value"])
        self.results.horizontalHeader().setStretchLastSection(True)
        self.status = QLabel("")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.view, 3)
        rl.addWidget(self.status)
        rl.addWidget(self.results, 1)

        split = QSplitter(Qt.Orientation.Horizontal)
        for w in (left, middle, right):
            split.addWidget(w)
        split.setStretchFactor(2, 3)
        layout = QVBoxLayout(self)
        layout.addWidget(split)
        self._rebuild_tree()

    # ------------------------------------------------------------------ public API
    def set_spectrum(self, x, y, thickness: float | None = None) -> None:
        self.x, self.y, self.thickness = (
            np.asarray(x, float),
            np.asarray(y, float),
            thickness,
        )
        self.preview()

    def set_recipe(self, recipe: Recipe) -> None:
        self.recipe = recipe
        self._rebuild_tree()
        self.recipe_changed.emit()

    # ------------------------------------------------------------------ tree
    def _rebuild_tree(self, select=None) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        root = QTreeWidgetItem([f"{self.recipe.name}  (settings)"])
        root.setData(0, Qt.ItemDataRole.UserRole, ("recipe", -1, -1))
        self.tree.addTopLevelItem(root)
        target = root
        for i, feat in enumerate(self.recipe.features):
            kind = "cluster" if feat.is_cluster else "peak"
            fi = QTreeWidgetItem(
                [
                    f"{feat.name}  [{kind}, {feat.model}, {feat.region[0]:g}-{feat.region[1]:g}]"
                ]
            )
            fi.setData(0, Qt.ItemDataRole.UserRole, ("feature", i, -1))
            root.addChild(fi)
            for j, step in enumerate(feat.baseline):
                it = QTreeWidgetItem([f"baseline {j + 1}: {step.method}"])
                it.setData(0, Qt.ItemDataRole.UserRole, ("step", i, j))
                fi.addChild(it)
            for j, peak in enumerate(feat.peaks):
                it = QTreeWidgetItem([f"peak {peak.name} @ {peak.center:g}"])
                it.setData(0, Qt.ItemDataRole.UserRole, ("peak", i, j))
                fi.addChild(it)
            for k in range(fi.childCount()):
                if select == fi.child(k).data(0, Qt.ItemDataRole.UserRole):
                    target = fi.child(k)
            if select == ("feature", i, -1):
                target = fi
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self.tree.setCurrentItem(target)

    def _select(self, item, _prev=None) -> None:
        if item is None:
            return
        kind, i, j = item.data(0, Qt.ItemDataRole.UserRole)
        self.current = (kind, i, j)
        obj = {
            "recipe": lambda: self.recipe,
            "feature": lambda: self.recipe.features[i],
            "step": lambda: self.recipe.features[i].baseline[j],
            "peak": lambda: self.recipe.features[i].peaks[j],
        }[kind]()
        while self.form_holder.count():
            w = self.form_holder.takeAt(0).widget()
            if w:
                # deleteLater only acts on a later event-loop pass; detach and hide now so
                # the old form is never drawn under the new one
                w.hide()
                w.setParent(None)
                w.deleteLater()
        form = DataclassForm(obj, CHOICES, skip=("features",))
        form.changed.connect(self._edited)
        self.form_holder.addWidget(form)
        self.form_holder.addStretch()
        self.preview()

    def _edited(self) -> None:
        self._rebuild_tree(select=self.current)
        self.recipe_changed.emit()
        self.preview()

    def _feature_index(self) -> int | None:
        if self.current and self.current[1] >= 0:
            return self.current[1]
        return 0 if self.recipe.features else None

    # ------------------------------------------------------------------ edits
    def _add_feature(self, n_peaks: int) -> None:
        lo, hi = self._visible_range()
        centre = (lo + hi) / 2
        width = (hi - lo) / 10
        peaks = [
            PeakDef(
                name=chr(ord("a") + k),
                center=lo + (k + 1) * (hi - lo) / (n_peaks + 1),
                fwhm=width,
            )
            for k in range(n_peaks)
        ]
        name = f"feature{len(self.recipe.features) + 1}"
        self.recipe.features.append(
            Feature(
                name=name,
                region=[lo, hi],
                model="gaussian",
                baseline=[BaselineStep()],
                peaks=peaks,
            )
        )
        if n_peaks == 1:
            self.recipe.features[-1].peaks[0].center = centre
        self._rebuild_tree(select=("feature", len(self.recipe.features) - 1, -1))
        self.recipe_changed.emit()

    def _add_step(self) -> None:
        i = self._feature_index()
        if i is None:
            return
        self.recipe.features[i].baseline.append(BaselineStep())
        self._rebuild_tree(
            select=("step", i, len(self.recipe.features[i].baseline) - 1)
        )
        self.recipe_changed.emit()

    def _add_peak(self) -> None:
        i = self._feature_index()
        if i is None:
            return
        f = self.recipe.features[i]
        f.peaks.append(
            PeakDef(
                name=f"p{len(f.peaks) + 1}",
                center=sum(f.region) / 2,
                fwhm=(f.region[1] - f.region[0]) / 10,
            )
        )
        self._rebuild_tree(select=("peak", i, len(f.peaks) - 1))
        self.recipe_changed.emit()

    def _remove(self) -> None:
        if not self.current:
            return
        kind, i, j = self.current
        if kind == "feature":
            del self.recipe.features[i]
        elif kind == "step":
            del self.recipe.features[i].baseline[j]
        elif kind == "peak":
            del self.recipe.features[i].peaks[j]
        self._rebuild_tree()
        self.recipe_changed.emit()

    def _load(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Load recipe", "", "Recipes (*.toml *.json)"
        )
        if path:
            self._try(lambda: self.set_recipe(Recipe.load(path)))

    def _load_builtin(self) -> None:
        names = sorted(builtin_recipes())
        name, ok = QInputDialog.getItem(
            self, "Built-in recipes", "Recipe:", names, 0, False
        )
        if ok:
            self.set_recipe(resolve_recipe(name))

    def _save(self) -> None:
        problems = self.recipe.validate()
        if problems:
            QMessageBox.warning(self, "Recipe has problems", "\n".join(problems))
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Save recipe",
            f"{self.recipe.name}.toml",
            "TOML (*.toml);;JSON (*.json)",
        )
        if path:
            self._try(lambda: self.recipe.save(Path(path)))

    def _try(self, fn) -> None:
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - show any load/save problem to the user
            QMessageBox.warning(self, "Problem", str(e))

    # ------------------------------------------------------------------ preview
    def _visible_range(self) -> tuple[float, float]:
        if self.x is None:
            return 1000.0, 1100.0
        (xmin, xmax), _ = self.view.plot.viewRange()
        lo, hi = max(xmin, self.x.min()), min(xmax, self.x.max())
        span = hi - lo
        return round(lo + span * 0.3, 1), round(hi - span * 0.3, 1)

    def _region_dragged(self, lo: float, hi: float) -> None:
        i = self._feature_index()
        if i is None:
            return
        self.recipe.features[i].region = [round(min(lo, hi), 2), round(max(lo, hi), 2)]
        self._edited()

    def preview(self) -> None:
        self.view.clear()
        self.results.setRowCount(0)
        if self.x is None:
            self.status.setText("Load a spectrum to preview.")
            return
        i = self._feature_index()
        if i is None:
            self.view.add_curve(self.x, self.y, "spectrum", color="#555555")
            self.view.hide_region()
            self.status.setText("Add a feature to start.")
            return
        feat = self.recipe.features[i]
        scale = 1.0
        if self.recipe.normalize_by == "thickness" and self.thickness:
            scale = 1.0 / self.thickness
        try:
            res = run_feature(self.x, self.y, feat, scale=scale)
        except Exception as e:  # noqa: BLE001 - preview must never crash the editor
            self.view.add_curve(self.x, self.y, "spectrum", color="#555555")
            self.status.setText(f"Cannot preview: {e}")
            return
        pad = 0.25 * (feat.region[1] - feat.region[0])
        m = (self.x >= feat.region[0] - pad) & (self.x <= feat.region[1] + pad)
        self.view.add_curve(self.x[m], self.y[m], "spectrum", color="#555555")
        running = np.zeros_like(res.y)
        for k, b in enumerate(res.baselines):
            running = running + b
            self.view.add_curve(
                res.x,
                running,
                f"baseline {k + 1}",
                dash=k < len(res.baselines) - 1,
                width=1.5,
            )
        if res.model is not None:
            self.view.add_curve(
                res.x, running + res.model, "fit", color="#000000", width=1.5, dash=True
            )
            for name, curve in res.peak_curves.items():
                self.view.add_curve(res.x, running + curve, f"peak {name}", width=1)
        self.view.show_region(*feat.region)
        self.status.setText(
            res.error
            or f"feature '{feat.name}': {len(feat.peaks)} peak(s), model {feat.model}"
        )
        self.results.setRowCount(len(res.values))
        for r, (k, v) in enumerate(res.values.items()):
            self.results.setItem(r, 0, QTableWidgetItem(k))
            self.results.setItem(r, 1, QTableWidgetItem(f"{v:.5g}"))
