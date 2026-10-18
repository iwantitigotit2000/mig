#!/bin/bash
# Stage A: real samples for both models on the 10 pilot layouts + short training runs.
set -x
cd "$(dirname "$0")/.."
G=/d/condamini/envs/mig_gligen/python.exe; M=/d/condamini/envs/mig_migc/python.exe; E=/d/condamini/envs/mig_eval/python.exe
$G scripts/gen_gligen.py --manifest manifests/cocomig_pilot.jsonl --out outputs/gen/pilot/G0
$M scripts/gen_migc.py   --manifest manifests/cocomig_pilot.jsonl --out outputs/gen/pilot/MIGC
$E evaluation/mig_eval.py --manifest manifests/cocomig_pilot.jsonl --gen_dir outputs/gen/pilot/G0 outputs/gen/pilot/MIGC
$E evaluation/quality.py --gen_dir outputs/gen/pilot/G0 outputs/gen/pilot/MIGC
$G training/train_gligen.py --config configs/train/g1_lr5e-5.yaml --name pilot_g1 --max_steps 60
$G training/train_gligen.py --config configs/train/g2_lr5e-5.yaml --name pilot_g2 --max_steps 60
echo STAGEA_DONE
