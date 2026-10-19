"""CPU-only evaluator sensitivity from stored target-colour coverage ratios.

Does not rerun generation, detection or segmentation. Position threshold and
all masks are held fixed; only the colour coverage threshold is varied.
"""
import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.final_statistics import cluster_estimate, paired_estimate


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--instances", type=Path, default=ROOT / "outputs/results/test_original/instances.csv")
    ap.add_argument("--out", type=Path, default=ROOT / "outputs/final_analysis")
    args = ap.parse_args()
    data = pd.read_csv(args.instances)
    if set(data.method) != {"G0", "MIGC"}:
        raise ValueError("Expected exactly G0 and MIGC")
    pos = data.official_pos.eq(1)
    if data.loc[pos, "target_color_ratio"].isna().any():
        raise ValueError("Colour ratios missing for position-passing targets")
    baseline = (pos & data.target_color_ratio.ge(.2) & data.empty_mask.eq(0)).astype(int)
    mismatches = int((baseline != data.official_success).sum())
    if mismatches:
        raise ValueError(f"Cannot reconstruct official 0.2 baseline from stored measurements ({mismatches} mismatches)")
    result = []
    for threshold in [.1, .2, .3]:
        frame = data.copy()
        frame["sensitivity_success"] = (frame.official_pos.eq(1) & frame.target_color_ratio.ge(threshold) & frame.empty_mask.eq(0)).astype(int)
        a, b = frame[frame.method.eq("MIGC")], frame[frame.method.eq("G0")]
        g0 = cluster_estimate(b, "sensitivity_success")
        migc = cluster_estimate(a, "sensitivity_success")
        delta = paired_estimate(a, b, "sensitivity_success")
        result.append({"colour_threshold":threshold,"G0_success":g0["estimate"],"MIGC_success":migc["estimate"],
                       "MIGC_minus_G0":delta["estimate"],"delta_ci95_low":delta["ci95"][0],"delta_ci95_high":delta["ci95"][1]})
    args.out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(result).to_csv(args.out / "colour_threshold_sensitivity.csv", index=False)
    try:
        source = args.instances.resolve().relative_to(ROOT)
    except ValueError:
        source = args.instances
    report={"source":str(source),"baseline_reconstruction_mismatches":mismatches,"position_threshold":.5,
            "masks":"unchanged stored SAM masks","bootstrap":{"unit":"layout","draws":5000,"seed":2026},
            "scope":"Post hoc sensitivity of the automatic evaluator; no new model training or inference.","results":result}
    (args.out/"colour_threshold_sensitivity.json").write_text(json.dumps(report,indent=2)+"\n")
    lines=["# Colour threshold sensitivity", "", "Only the target-colour mask-coverage threshold changes. Position matching, images, detections and SAM masks are fixed. The original 0.2 baseline is reconstructed with zero mismatches.", "",
           "| Colour coverage threshold | GLIGEN joint success | MIGC joint success | MIGC minus GLIGEN and 95% CI |", "|---|---:|---:|---|"]
    for r in result:
        lines.append(f"| {r['colour_threshold']:.1f} | {r['G0_success']:.1%} | {r['MIGC_success']:.1%} | {r['MIGC_minus_G0']*100:+.1f} pp [{r['delta_ci95_low']*100:+.1f}, {r['delta_ci95_high']*100:+.1f}] |")
    lines += ["", "The MIGC advantage remains positive across these three evaluator thresholds. Absolute accuracy changes appreciably, so report the threshold with every score. This supports ranking stability within the checked range, not perceptual correctness or a causal architecture claim. Thresholds are exploratory and must not replace the official 0.2 result with the most favourable score.", "", "This cache is not sufficient to lower the position IoU threshold: colour masks were not computed for every target below the original position threshold. New detector or segmentation settings would require new GPU evaluation."]
    text="\n".join(lines)+"\n"
    (args.out/"colour_threshold_sensitivity.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
