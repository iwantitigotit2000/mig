#!/bin/bash
# Stage C-1: grounding-duration ratio on the VALIDATION split only (30 layouts x 2 seeds).
#   GLIGEN alpha_type = [r, 0, 1-r];  MIGC MIGCsteps = r*50   for r in {0.3, 0.5, 0.8}
source "$(dirname "$0")/lib.sh"
set -x
V=manifests/cocomig_val.jsonl
for r in 0.3 0.5 0.8; do
  $G scripts/gen_gligen.py --manifest $V --out outputs/gen/val/G0_r$r --alpha_type $r 0 $($G -c "print(round(1-$r,2))")
  $M scripts/gen_migc.py   --manifest $V --out outputs/gen/val/MIGC_r$r --migc_steps $($G -c "print(int(round($r*50)))")
done
DIRS=$(ls -d outputs/gen/val/G0_r* outputs/gen/val/MIGC_r*)
$E evaluation/mig_eval.py --manifest $V --gen_dir $DIRS
$E evaluation/quality.py --gen_dir $DIRS
RUNS=$(for d in $DIRS; do echo "$(basename $d)=$d"; done)
$E analysis/aggregate.py --manifest $V --runs $RUNS --ref G0_r0.3 --out outputs/results/val_ratio
echo STAGEC_RATIO_DONE
