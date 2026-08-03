# PressureFit figure assets

This directory contains editable LaTeX/Graphviz sources and generated
publication assets embedded by `../PRESSUREFIT_ALGO.md`.

| Source | Rendered outputs |
|---|---|
| `pressurefit_algorithm.tex` | `pressurefit_algorithm.pdf`, `pressurefit_algorithm.svg` |
| `pressurefit_simulator.tex` | `pressurefit_simulator.pdf`, `pressurefit_simulator.svg` |
| `pressurefit_architecture.dot` | `pressurefit_architecture.pdf`, `pressurefit_architecture.svg` |

Regenerate all three figures from this directory with:

```bash
build_dir=$(mktemp -d)

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="$build_dir" pressurefit_algorithm.tex
pdfcrop --margins 8 \
  "$build_dir/pressurefit_algorithm.pdf" pressurefit_algorithm.pdf
pdftocairo -svg pressurefit_algorithm.pdf pressurefit_algorithm.svg

latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir="$build_dir" pressurefit_simulator.tex
pdfcrop --margins 8 \
  "$build_dir/pressurefit_simulator.pdf" pressurefit_simulator.pdf
pdftocairo -svg pressurefit_simulator.pdf pressurefit_simulator.svg

dot -Tpdf pressurefit_architecture.dot -o pressurefit_architecture.pdf
dot -Tsvg pressurefit_architecture.dot -o pressurefit_architecture.svg
```

The SVG files are used for inline Markdown rendering. The PDFs preserve the
same vector content for papers and presentations. The algorithms require a
standard TeX Live installation; the architecture requires Graphviz. Do not
edit generated files by hand.
