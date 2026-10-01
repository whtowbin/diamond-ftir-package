"""Settings for every registered analysis, one set for all data (spectra and maps alike).

Each analysis is a tickable group; its settings dataclass becomes a form, and nested
settings (e.g. nitrogen, hydrogen) become folding sections, so the panel stays short and
scrolls instead of running off the screen. Below: the recipes to run.
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path

from qtpy.QtCore import Qt, Signal
from qtpy.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QToolBox,
    QVBoxLayout,
    QWidget,
)

from ...core import Recipe, builtin_recipes
from .forms import DataclassForm


def sections(obj, choices, skip=()) -> tuple[QToolBox, list[DataclassForm]]:
    """General settings first, then one folding section per nested settings dataclass."""
    box = QToolBox()
    forms: list[DataclassForm] = []
    general = DataclassForm(obj, choices, skip=skip)
    if general.editors:
        box.addItem(general, "General")
        forms.append(general)
    for f in fields(obj):
        value = getattr(obj, f.name)
        if is_dataclass(value) and f.name not in skip:
            form = DataclassForm(value, choices)
            box.addItem(form, f.name.replace("_", " ").capitalize())
            forms.append(form)
    return box, forms


class SettingsPanel(QScrollArea):
    changed = Signal()

    def __init__(self, session, parent=None):
        super().__init__(parent)
        self.session = session
        self.setWidgetResizable(True)
        # fit the panel's width: long settings shrink instead of scrolling sideways
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.build()

    def build(self) -> None:
        s = self.session
        body = QWidget()
        layout = QVBoxLayout(body)
        self.groups: dict[str, QGroupBox] = {}
        for name, analysis in s.registry.analyses.items():
            group = QGroupBox(name)
            group.setCheckable(True)
            group.setChecked(s.enabled.get(name, False))
            group.toggled.connect(lambda on, n=name: self._enable(n, on))
            inner = QVBoxLayout(group)
            box, forms = sections(
                s.params[name], analysis.choices, analysis.skip_fields
            )
            if name in s.map_params:
                form = DataclassForm(
                    s.map_params[name],
                    analysis.choices,
                    skip=("example_grid", "block_size", "seed"),
                )
                box.addItem(form, "Map processing")
                forms.append(form)
            for form in forms:
                form.changed.connect(self._edited)
            inner.addWidget(box)
            self.groups[name] = group
            layout.addWidget(group)

        rec = QGroupBox("Recipes")
        rl = QVBoxLayout(rec)
        note = QLabel("Ticked recipes run with the analyses that use recipes.")
        note.setWordWrap(True)
        rl.addWidget(note)
        self.recipe_list = QListWidget()
        self.recipe_list.setMaximumHeight(140)
        self.recipe_list.itemChanged.connect(self._recipes_changed)
        ticked = {getattr(r, "name", r) for r in s.recipes}
        for n in sorted(builtin_recipes()):
            self._add_recipe(n, n, n in ticked)
        for r in s.recipes:
            if isinstance(r, Recipe) or Path(str(r)).suffix:
                self._add_recipe(getattr(r, "name", Path(str(r)).name), r, True)
        rl.addWidget(self.recipe_list)
        row = QHBoxLayout()
        add = QPushButton("Add recipe file…")
        add.clicked.connect(self._add_file)
        row.addWidget(add)
        rl.addLayout(row)
        layout.addWidget(rec)
        layout.addStretch(1)
        self.setWidget(body)

    def _enable(self, name: str, on: bool) -> None:
        self.session.enabled[name] = on
        self._edited()

    def _edited(self) -> None:
        self.session.settings_dirty = True
        self.changed.emit()

    def _add_recipe(self, label: str, ref, checked: bool) -> None:
        self.recipe_list.blockSignals(True)
        item = QListWidgetItem(label)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
        item.setCheckState(
            Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked
        )
        item.setData(Qt.ItemDataRole.UserRole, ref)
        self.recipe_list.addItem(item)
        self.recipe_list.blockSignals(False)

    def use_recipe(self, recipe: Recipe) -> None:
        """Add (or replace) an in-memory recipe from the editor, ticked."""
        label = f"{recipe.name} (editor)"
        for k in range(self.recipe_list.count()):
            if self.recipe_list.item(k).text() == label:
                self.recipe_list.takeItem(k)
                break
        self._add_recipe(label, recipe, True)
        self._recipes_changed()

    def _add_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Add recipe", "", "Recipes (*.toml *.json)"
        )
        if path:
            self._add_recipe(Path(path).name, path, True)
            self._recipes_changed()

    def _recipes_changed(self, _item=None) -> None:
        self.session.recipes = [
            self.recipe_list.item(k).data(Qt.ItemDataRole.UserRole)
            for k in range(self.recipe_list.count())
            if self.recipe_list.item(k).checkState() == Qt.CheckState.Checked
        ]
        self._edited()
