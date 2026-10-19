# Provenance of the test-set numbers

instances.csv: `outputs/results/test_original/instances.csv`

- G0: 200 images; recorded GPU: not recorded; checkpoint artifact `gligen_generation_text_box.bin` (absolute source path redacted; Windows drive path (local workstation)); mean 5.75 s/image; peak 9.36 GB
- MIGC: 200 images; recorded GPU: not recorded; checkpoint artifact `MIGC_SD14.ckpt` (absolute source path redacted; Windows drive path (local workstation)); mean 6.52 s/image; peak 7.82 GB

**Status: PROVISIONAL — metadata points to a local-workstation run, but no explicit provenance.json confirms the machine and run owner**

To mark a results directory as final, add `provenance.json` next to instances.csv, e.g. `{"machine": "CCDS TC2 cluster, NVIDIA A40", "declared_by": "<name>", "job_id": "...", "note": "..."}`.

Earlier group notes mention an A40 result of 0.319 / 0.669, while the committed run gives 0.321 / 0.665. The A40 raw files are not in the repository, so that result is neither verified nor used as the report's canonical result. If the files become available, add explicit provenance, select one run, and regenerate every Stage D artifact.
