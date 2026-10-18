#!/bin/bash
# Stage C-2: G1, G2(lr 5e-5), G2(lr 1e-5); identical data / effective batch 8 / 2000 updates.
# Resumable: re-running continues from ckpt_latest.pt. --max_hours keeps each call inside a job slot.
source "$(dirname "$0")/lib.sh"
set -x
for c in g1_lr5e-5 g2_lr5e-5 g2_lr1e-5; do
  $G training/train_gligen.py --config configs/train/$c.yaml --max_hours ${MAX_HOURS:-5}
done
echo STAGEC_TRAIN_DONE
