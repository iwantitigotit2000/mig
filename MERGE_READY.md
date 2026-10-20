# Stage D merge-ready note

Prepared on 20 October 2026 against upstream `main` commit
`2e6de0b4fca10a7ffaf723a851093ff731b105ff`.

## What this branch adds

- deterministic CPU analysis and report-generation scripts;
- split, difficulty, and train/evaluation-isolation checks;
- paired layout-bootstrap statistics, failure analysis, evaluator sensitivity,
  fine-tuning analysis, figures, and generated report tables;
- the submitted report `report/group_7_report.pdf` (15 pages, single column, Group 7) and the
  two-column LaTeX draft `report/report.pdf` that the number-checking pipeline targets;
- a receipt for the independent Apple Silicon Mac CPU reproduction.

## Validation completed

On a clean copy of the base commit, `bash scripts/stageD_analysis.sh` completed
successfully. The split/isolation check passed, all statistics and figures were
regenerated, every number referenced by the report passed the consistency check,
and `report/report.pdf` rebuilt successfully. All eight PDF pages were also
rendered and inspected for clipping, overlap, and readability.

The separate Mac reproduction is recorded in
`docs/MAC_CPU_REPRODUCTION.md`. It independently passed the CPU checks and
analysis. The Mac did not rerun GPU generation, model inference, or training.

## Evidence boundary

The committed test artifacts identify a local RTX 4070 run and are marked
**PROVISIONAL** in `outputs/final_analysis/PROVENANCE.md`. The A40 cluster result
mentioned in earlier group notes is not treated as verified because its raw files
are not in the repository. This limitation is explicit in the report and does not
prevent merging the analysis branch.

## Group actions before final submission

1. Remove or complete the dangling footnote marker (¹) in Section 4.1 of `group_7_report.pdf` at the source document.
2. Confirm the final contribution statement in `report/contribution.md` after
   roles are assigned.
3. If the missing A40 artifacts later become available, choose one canonical run,
   update provenance, regenerate Stage D, and quote only that run in the report.

Suggested commit message:

`Add validated Stage D analysis and corrected report`
