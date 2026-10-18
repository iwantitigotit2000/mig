#!/bin/bash
# Stage A (training part): 60-update pilot runs for G1 and G2 (timing, memory, update checks).
source "$(dirname "$0")/lib.sh"
set -x
$G training/train_gligen.py --config configs/train/g1_lr5e-5.yaml --name pilot_g1 --max_steps 60
$G training/train_gligen.py --config configs/train/g2_lr5e-5.yaml --name pilot_g2 --max_steps 60
echo STAGEA_TRAIN_DONE
