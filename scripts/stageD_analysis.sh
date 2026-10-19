#!/usr/bin/env bash
# Stage D (CPU only): data checks, statistics, figures, failure cases, report figures + PDF.
# Re-run after replacing outputs/results/test_original and outputs/gen/test with the cluster results.
#   bash scripts/stageD_analysis.sh
#   PY=python bash scripts/stageD_analysis.sh
# CPU deps: numpy pandas scipy matplotlib Pillow PyYAML   (pip install numpy pandas scipy matplotlib Pillow PyYAML)
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-python3}
"$PY" -c "import numpy, pandas, scipy, matplotlib, PIL, yaml" || { echo "missing CPU deps: pip install numpy pandas scipy matplotlib Pillow PyYAML"; exit 2; }
"$PY" data_tools/check_data_isolation.py            # exits non-zero if isolation fails -> pipeline stops
"$PY" data_tools/annotate_difficulty.py > /dev/null
"$PY" analysis/final_statistics.py > /dev/null
"$PY" analysis/colour_threshold_sensitivity.py > /dev/null
"$PY" analysis/plot_training.py > /dev/null
"$PY" analysis/draw_architecture.py
"$PY" analysis/make_figures.py
"$PY" analysis/failure_cases.py
# ---- connect analysis outputs to the report
mkdir -p report/figures
cp outputs/final_analysis/figures/*.png report/figures/
"$PY" analysis/make_report_tables.py          # report/tables/*.tex: every table cell and prose number
"$PY" analysis/check_report_numbers.py        # fails (and stops here) if report.tex still carries stale or hand-typed numbers
if command -v pdflatex >/dev/null 2>&1; then
  (cd report && pdflatex -interaction=nonstopmode report.tex >/dev/null && pdflatex -interaction=nonstopmode report.tex >/dev/null && echo "report/report.pdf rebuilt")
else
  echo "pdflatex not found: report/figures updated, rebuild report.pdf manually"
fi
echo
echo "Stage D analysis written to outputs/final_analysis/ (data status: see PROVENANCE.md)."
echo "Report tables and prose numbers come from report/tables/*.tex (generated + checked above)."
echo "NOTE: analysis/*_analysis.md are hand-written; re-read them against outputs/final_analysis/*.md after replacing results."
