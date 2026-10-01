"""A form generated from any dataclass: the reusable settings widget.

Works for analysis parameters, recipe baseline steps, peaks, or any other settings class
that follows the convention ``field(metadata={"description": ...})``. Choices for text
fields come from a ``choices`` mapping (field name -> allowed values). Runs standalone or
inside napari (plain qtpy widgets).
"""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QCheckBox, QComboBox, QFormLayout, QLabel, QLineEdit, QWidget


def format_value(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        if value and isinstance(value[0], (list, tuple)):
            return "; ".join(", ".join(_num(v) for v in pair) for pair in value)
        return ", ".join(_num(v) for v in value)
    if isinstance(value, dict):
        return "; ".join(f"{k} = {v}" for k, v in value.items())
    return _num(value)


def _num(v: Any) -> str:
    return f"{v:g}" if isinstance(v, float) else str(v)


def parse_value(default: Any, text: str) -> Any:
    """Text back to the type of ``default``. Raises ValueError with a readable message."""
    text = text.strip()
    if isinstance(default, bool):
        if text.lower() in ("true", "yes", "1", "on"):
            return True
        if text.lower() in ("false", "no", "0", "off"):
            return False
        raise ValueError(f"expected yes/no, got {text!r}")
    if isinstance(default, dict):
        out = {}
        for part in filter(None, (p.strip() for p in text.split(";"))):
            key, _, expr = part.partition("=")
            if not expr:
                raise ValueError(f"expected 'name = expression', got {part!r}")
            out[key.strip()] = expr.strip()
        return out
    if isinstance(default, (list, tuple)):
        if not text:
            return type(default)()
        if ";" in text or (default and isinstance(default[0], (list, tuple))):
            return [
                [float(v) for v in pair.split(",")]
                for pair in text.split(";")
                if pair.strip()
            ]
        items = [t.strip() for t in text.split(",") if t.strip()]
        if default and isinstance(default[0], str):
            return type(default)(items)
        try:
            return type(default)(float(t) for t in items)
        except ValueError:
            return type(default)(items)
    if isinstance(default, str):
        return text
    if isinstance(default, int):
        return int(float(text))
    return float(text)


class DataclassForm(QWidget):
    """Editable form for one dataclass instance. Nested dataclass/list-of-dataclass fields
    are skipped (edit them with their own form). ``changed`` fires after every valid edit."""

    changed = Signal()

    def __init__(
        self, obj: Any, choices: dict[str, tuple] | None = None, parent=None, skip=()
    ):
        super().__init__(parent)
        if not is_dataclass(obj):
            raise TypeError("DataclassForm needs a dataclass instance")
        self.obj = obj
        self.choices = choices or {}
        self.editors: dict[str, QWidget] = {}
        layout = QFormLayout(self)
        layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)  # narrow panels
        layout.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.error = QLabel("")
        self.error.setStyleSheet("color: #b2182b")
        for f in fields(obj):
            value = getattr(obj, f.name)
            if (
                f.name in skip
                or is_dataclass(value)
                or (isinstance(value, list) and value and is_dataclass(value[0]))
            ):
                continue
            editor = self._editor(f.name, value)
            tip = f.metadata.get("description", "")
            label = QLabel(f.name.replace("_", " "))
            label.setToolTip(tip)
            editor.setToolTip(tip)
            layout.addRow(label, editor)
            self.editors[f.name] = editor
        layout.addRow(self.error)

    def _editor(self, name: str, value: Any) -> QWidget:
        if isinstance(value, bool):
            box = QCheckBox()
            box.setChecked(value)
            box.toggled.connect(lambda _=None, n=name: self._commit(n))
            return box
        if name in self.choices:
            combo = QComboBox()
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(8)
            combo.addItems([str(c) for c in self.choices[name]])
            combo.setCurrentText(str(value))
            combo.currentTextChanged.connect(lambda _=None, n=name: self._commit(n))
            return combo
        line = QLineEdit(format_value(value))
        line.setMinimumWidth(60)
        line.editingFinished.connect(lambda n=name: self._commit(n))
        return line

    def _commit(self, name: str) -> None:
        editor = self.editors[name]
        default = getattr(self.obj, name)
        try:
            if isinstance(editor, QCheckBox):
                value = editor.isChecked()
            elif isinstance(editor, QComboBox):
                value = parse_value(default, editor.currentText())
            else:
                value = parse_value(default, editor.text())
        except ValueError as e:
            self.error.setText(f"{name.replace('_', ' ')}: {e}")
            return
        self.error.setText("")
        setattr(self.obj, name, value)
        self.changed.emit()

    def refresh(self) -> None:
        """Re-read values from the object (after it was changed elsewhere)."""
        for name, editor in self.editors.items():
            value = getattr(self.obj, name)
            editor.blockSignals(True)
            if isinstance(editor, QCheckBox):
                editor.setChecked(bool(value))
            elif isinstance(editor, QComboBox):
                editor.setCurrentText(str(value))
            else:
                editor.setText(format_value(value))
            editor.blockSignals(False)
