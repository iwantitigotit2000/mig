# Group 7 — Contribution statement (to be filled in by the group)

> Rule: each member lists only what they actually implemented, ran or verified. Training runs are
> credited to the member who executed them. Assignment below is intentionally **blank**; agree it
> in the group, then convert with `pandoc contribution.md -o group_7_contribution.pdf`.

| Member (name as on student card) | Contribution (pick from the task list below, add specifics) |
|---|---|
| Zhichen Liu | |
| Xinkuo Wu | |
| Yanyu Wen | |

## Task list (unassigned)

Code and experiments
- COCO-MIG test / validation split and difficulty annotations (`data_tools/build_cocomig_manifest.py`, `annotate_difficulty.py`)
- COCO train2014 training subset and manifests (`data_tools/prepare_coco_train.py`)
- Independent train/eval isolation check and manifest versioning (`data_tools/check_data_isolation.py`)
- GLIGEN adapter + official-script parity check; SDPA check
- MIGC adapter + official-script parity check
- Unified generation config (`configs/gen.yaml`); grounding-ratio experiment (Stage C-1)
- MIG-Bench evaluator re-implementation (`evaluation/`) and aggregation (`analysis/aggregate.py`)
- GLIGEN fine-tuning code (`training/`), G1 / G2 cluster runs, gradient and weight checks
- Stage B generation and evaluation on the cluster; Stage C-3 validation evaluation
- Structure probes (`analysis/probe_*.py`)

Analysis (CPU)
- Layout-bootstrap statistics and colour-threshold sensitivity (`final_statistics.py`, `colour_threshold_sensitivity.py`)
- Result tables / figures, difficulty-condition analysis (`make_figures.py`, `results_analysis.md`)
- Rule-based failure-case selection and manual case review (`failure_cases.py`, `failure_cases/cases.csv`)
- Training-curve and learning-rate / checkpoint analysis (`plot_training.py`, `finetune_analysis.md`)
- Architecture analysis and diagrams (`draw_architecture.py`, `architecture_analysis.md`)

Writing
- Report sections: introduction; methods; dataset & evaluation; results; fine-tuning; discussion & limitations; conclusion
- Report integration, number verification against `outputs/final_analysis/`, proof-reading
