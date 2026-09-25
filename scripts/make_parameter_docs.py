"""Regenerate docs/parameters.md from the parameter dataclasses.

Run with: uv run python scripts/make_parameter_docs.py
"""

from dataclasses import fields
from pathlib import Path

from diamond_ftir_package.params import AnalysisParams

TITLES = {
    "diamond": "Thickness and saturation",
    "nitrogen": "Nitrogen",
    "hydrogen": "Hydrogen (3107 / 3085)",
    "platelet": "Platelets and the 1405 peak",
    "amber": "Amber centres",
}


def build() -> str:
    params = AnalysisParams()
    out = [
        "# Parameter reference",
        "",
        (
            "Generated from `src/diamond_ftir_package/params.py` by "
            "`scripts/make_parameter_docs.py`; edit descriptions there. Every default equals the "
            "value that was hardcoded before parameters were exposed. Settings are saved and "
            "loaded as JSON (`diamond-ftir defaults` prints them all)."
        ),
        "",
    ]
    for section, title in TITLES.items():
        sub = getattr(params, section)
        out += [
            f"## {title}",
            "",
            "| Setting | Default | What it does |",
            "|---|---|---|",
        ]
        for f in fields(sub):
            description = f.metadata.get("description", "")
            out.append(f"| `{f.name}` | `{getattr(sub, f.name)}` | {description} |")
        out.append("")
    out += ["## Which measurements run", "", "| Setting | Default |", "|---|---|"]
    for key in ("run_nitrogen", "run_hydrogen", "run_platelets", "run_amber"):
        out.append(f"| `{key}` | `{getattr(params, key)}` |")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    target = Path(__file__).resolve().parent.parent / "docs" / "parameters.md"
    target.parent.mkdir(exist_ok=True)
    target.write_text(build())
    print(f"Wrote {target}")
