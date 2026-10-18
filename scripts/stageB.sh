#!/bin/bash
# Stage B: original models on the 100-layout test set (x2 seeds) + evaluation + architecture probes.
# Estimated on RTX 4070: GLIGEN 5.4 s/img, MIGC ~6.5 s/img -> ~40 min generation for 2x200 images.
source "$(dirname "$0")/lib.sh"
set -x
T=manifests/cocomig_test.jsonl
for pass in 1 2; do   # pass 2 only fills samples that failed transiently (skip-existing)
  $G scripts/gen_gligen.py --manifest $T --out outputs/gen/test/G0
  $M scripts/gen_migc.py   --manifest $T --out outputs/gen/test/MIGC
done
$E evaluation/mig_eval.py --manifest $T --gen_dir outputs/gen/test/G0 outputs/gen/test/MIGC
$E evaluation/quality.py --gen_dir outputs/gen/test/G0 outputs/gen/test/MIGC
$E analysis/aggregate.py --manifest $T --runs G0=outputs/gen/test/G0 MIGC=outputs/gen/test/MIGC --ref G0 --out outputs/results/test_original
$G analysis/probe_gligen.py
$M analysis/probe_migc.py
echo STAGEB_DONE
