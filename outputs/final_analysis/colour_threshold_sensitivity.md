# Colour threshold sensitivity

Only the target-colour mask-coverage threshold changes. Position matching, images, detections and SAM masks are fixed. The original 0.2 baseline is reconstructed with zero mismatches.

| Colour coverage threshold | GLIGEN joint success | MIGC joint success | MIGC minus GLIGEN and 95% CI |
|---|---:|---:|---|
| 0.1 | 37.9% | 73.8% | +35.9 pp [+31.0, +40.8] |
| 0.2 | 32.1% | 66.5% | +34.4 pp [+29.4, +39.3] |
| 0.3 | 26.4% | 59.9% | +33.5 pp [+28.6, +38.4] |

The MIGC advantage remains positive across these three evaluator thresholds. Absolute accuracy changes appreciably, so report the threshold with every score. This supports ranking stability within the checked range, not perceptual correctness or a causal architecture claim. Thresholds are exploratory and must not replace the official 0.2 result with the most favourable score.

This cache is not sufficient to lower the position IoU threshold: colour masks were not computed for every target below the original position threshold. New detector or segmentation settings would require new GPU evaluation.
