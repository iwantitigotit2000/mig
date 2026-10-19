# Independent Mac CPU reproduction receipt

- Date: 2026-10-18
- Operator: Yanyu Wen
- Platform: Apple Silicon Mac (`Darwin arm64`)
- Environment: Conda `ca6127_cpu`, Python 3.12
- Command: `bash scripts/stageD_analysis.sh 2>&1 | tee stageD_mac_run.log`

## Result

- Train/evaluation isolation check: **PASS**
- Fixed-manifest checks: 30 validation layouts, 100 test layouts, and 3,100 COCO train2014 images checked
- Evaluation layouts flagged as copied from training layouts at mean matched IoU >= 0.9: **0 of 130**
- Stage D statistics, subgroup analyses, colour-threshold sensitivity results, figures, and failure-case panels: **regenerated successfully**
- Generated report tables: **current**
- Automated report-number consistency check: **PASS**
- PDF rebuild on this Mac: **not run** because `pdflatex` was unavailable; this does not affect the CPU analysis checks above

## Evidence boundary

This reproduction reran the CPU analysis from the result CSVs, metadata, caches, and images committed in the repository. It did **not** rerun GPU training, model inference, GroundingDINO/SAM evaluation, or CLIP-IQA inference. The committed test-set metadata points to the local RTX 4070 run, so its provenance remains `PROVISIONAL`; the A40 cluster result files quoted elsewhere were not available for independent verification.
