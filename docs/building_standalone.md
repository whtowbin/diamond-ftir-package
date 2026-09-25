# Building a standalone app (untested draft)

Not yet verified on any platform. The plan is PyInstaller; Nuitka is an optional experiment
(see `TODO.md`). Build on each target OS, since PyInstaller does not cross-compile.

```bash
uv run --with pyinstaller pyinstaller --windowed --name "Diamond FTIR" \
  --collect-submodules diamond_ftir_package \
  src/diamond_ftir_package/gui.py
```

Things to check on the first build: matplotlib's TkAgg backend and scipy load correctly, the
bundled `CAXBDY`/`typeIIA` data are found, and startup time is acceptable. macOS builds need
code signing and notarisation to avoid Gatekeeper warnings; Windows builds may trigger antivirus
false positives. [PLACEHOLDER: record results and final command here.]
