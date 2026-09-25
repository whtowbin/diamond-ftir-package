# Vendored: SNAC (Simultaneous Nitrogen Aggregation and Cooling)

- Source: https://github.com/LauraSp/SNAC
- Commit: 20f33753f54bf663131d89f9e6c801fac188744d (2026-05-29)
- Licence: MIT (see LICENSE in this folder), Copyright (c) 2025 Laura Speich / University of Bristol
- Reference: Wincott et al. (2026) [CITATION NEEDED: full reference for the SNAC paper]

Why vendored: SNAC is not on PyPI, and the PyPI name `snac` belongs to an unrelated
package (an audio codec), so it cannot be declared as a dependency by name.

Changes from upstream: imports changed from `from snac.X import ...` to relative
`from .X import ...` so the package works inside `diamond_ftir_package._vendor`.
No other changes. Use it through `diamond_ftir_package.thermometry` (see docs/thermometry.md).
