#!/bin/bash
# Stage C-3: fine-tuned checkpoints. Validation: both G2 learning rates and early(500)/late(2000)
# checkpoints of G1/G2 at the ratio selected on val. Test: G1 and the selected G2 (late ckpt).
# Usage: RATIO=0.3 G2_LR=5e-5 bash scripts/stageC_eval_ft.sh
source "$(dirname "$0")/lib.sh"
set -x
R=${RATIO:?set RATIO selected on val}; A="$R 0 $($G -c "print(round(1-$R,2))")"
RUNS=$($G -c "from common.utils import load_paths;print(load_paths()['runs_root'])")
V=manifests/cocomig_val.jsonl; T=manifests/cocomig_test.jsonl
for run in g1_lr5e-5 g2_lr5e-5 g2_lr1e-5; do for s in 000500 002000; do
  $G scripts/gen_gligen.py --manifest $V --out outputs/gen/val/${run}_s$s --alpha_type $A --trainable $RUNS/$run/ckpt_$s.pt
done; done
DIRS=$(ls -d outputs/gen/val/g*_s*)
$E evaluation/mig_eval.py --manifest $V --gen_dir $DIRS
$E evaluation/quality.py --gen_dir $DIRS
$E analysis/aggregate.py --manifest $V --runs G0=outputs/gen/val/G0_r$R $(for d in $DIRS; do echo "$(basename $d)=$d"; done) --ref G0 --out outputs/results/val_finetune
if [ -n "$G2_LR" ]; then
  $G scripts/gen_gligen.py --manifest $T --out outputs/gen/test/G1 --alpha_type $A --trainable $RUNS/g1_lr5e-5/ckpt_002000.pt
  $G scripts/gen_gligen.py --manifest $T --out outputs/gen/test/G2 --alpha_type $A --trainable $RUNS/g2_lr$G2_LR/ckpt_002000.pt
  [ "$R" != "0.3" ] && $G scripts/gen_gligen.py --manifest $T --out outputs/gen/test/G0_r$R --alpha_type $A
  $E evaluation/mig_eval.py --manifest $T --gen_dir outputs/gen/test/G1 outputs/gen/test/G2 $( [ "$R" != "0.3" ] && echo outputs/gen/test/G0_r$R )
  $E evaluation/quality.py --gen_dir outputs/gen/test/G1 outputs/gen/test/G2 $( [ "$R" != "0.3" ] && echo outputs/gen/test/G0_r$R )
fi
echo STAGEC_FT_DONE
