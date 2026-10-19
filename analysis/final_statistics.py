"""Reanalyse committed results with layout-clustered, paired bootstrap intervals.

No neural inference. Both seeds and all objects of each sampled layout stay
together. Existing CSVs/results are never overwritten. CIs are descriptive,
unadjusted for multiple comparisons; validation comparisons are exploratory.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def cluster_estimate(frame, column, *, n_boot=5000, seed=2026, universe=None):
    if frame.empty or not np.isfinite(frame[column].to_numpy(dtype=float)).all():
        raise ValueError(f"Empty or non-finite data for {column}")
    groups = frame.groupby("layout_id")[column].agg(["sum", "count"])
    if universe is not None:
        groups = groups.reindex(universe, fill_value=0)
    if not groups["count"].sum():
        raise ValueError("No observations")
    rng = np.random.default_rng(seed)
    picks = rng.integers(len(groups), size=(n_boot, len(groups)))
    sums = groups["sum"].to_numpy()[picks].sum(axis=1)
    counts = groups["count"].to_numpy()[picks].sum(axis=1)
    valid = counts > 0
    values = sums[valid] / counts[valid]
    return {"estimate": float(groups["sum"].sum() / groups["count"].sum()),
            "ci95": np.percentile(values, [2.5, 97.5]).tolist(),
            "n_observations": len(frame), "n_layouts_with_observations": int((groups["count"] > 0).sum()),
            "valid_bootstraps": int(valid.sum())}


def paired_estimate(a, b, column, *, n_boot=5000, seed=2026):
    keys = ["sample_id", "idx"] if "idx" in a else ["sample_id"]
    merged = a[keys + ["layout_id", column]].merge(b[keys + ["layout_id", column]], on=keys,
                suffixes=("_a", "_b"), how="outer", indicator=True, validate="one_to_one")
    if not merged._merge.eq("both").all() or not merged.layout_id_a.eq(merged.layout_id_b).all():
        raise ValueError("Paired comparisons require identical observations and layout IDs")
    delta = pd.DataFrame({"layout_id": merged.layout_id_a, "delta": merged[column+"_a"] - merged[column+"_b"]})
    result = cluster_estimate(delta, "delta", n_boot=n_boot, seed=seed)
    result["direction"] = "a minus b"
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()
    if args.n_boot < 100:
        ap.error("Use at least 100 bootstrap draws")
    root = args.root.resolve()
    out = args.out or root / "outputs/final_analysis"
    out.mkdir(parents=True, exist_ok=True)
    estimate = lambda f, c, **kw: cluster_estimate(f, c, n_boot=args.n_boot, **kw)
    pair = lambda a, b, c: paired_estimate(a, b, c, n_boot=args.n_boot)
    testdir = root / "outputs/results/test_original"
    I, M = pd.read_csv(testdir / "instances.csv"), pd.read_csv(testdir / "images.csv")
    I["colour_wrong"] = I.failure.eq("colour_wrong").astype(float)
    I["missing"] = I.failure.eq("missing").astype(float)
    I["mislocated"] = I.failure.eq("mislocated").astype(float)
    metrics = {"joint_success": (I, "official_success"), "image_success": (M, "official_image_success"),
               "official_miou": (I, "official_miou"), "localisation_iou": (I, "loc_iou"),
               "position_success": (I, "official_pos"), "one_to_one_presence": (I, "o2o_present"),
               "one_to_one_success": (I, "o2o_success"), "colour_wrong": (I, "colour_wrong"),
               "missing": (I, "missing"), "mislocated": (I, "mislocated"), "clipiqa": (M, "clipiqa")}
    main_results = {m: {} for m in ["G0", "MIGC"]}
    paired_results = {}
    for metric, (frame, col) in metrics.items():
        a, b = frame[frame.method == "MIGC"], frame[frame.method == "G0"]
        for method, f in [("G0", b), ("MIGC", a)]:
            main_results[method][metric] = estimate(f, col)
        paired_results[metric] = pair(a, b, col)
    for method in main_results:
        f = I[I.method == method]
        layouts = sorted(f.layout_id.unique())
        main_results[method]["colour_given_position"] = estimate(f[f.official_pos == 1], "official_color", universe=layouts)
        main_results[method]["crosstalk_among_colour_errors"] = estimate(f[f.failure == "colour_wrong"], "crosstalk", universe=layouts)
        imgmean = f.groupby(["sample_id", "layout_id"], as_index=False).official_success.mean()
        main_results[method]["layout_macro_joint_success"] = estimate(imgmean, "official_success")

    subgroup = []
    for field in ["level", "overlap_bucket", "same_cat_diff_color"]:
        for group, frame in I.groupby(field):
            for method, f in frame.groupby("method"):
                subgroup.append({"group_by": field, "group": str(group), "method": method,
                                 **estimate(f, "official_success")})
    # Add input-derived relative size diagnostics without modifying fixed manifests.
    rows = [json.loads(s) for s in (root / "manifests/cocomig_test.jsonl").read_text().splitlines() if s.strip()]
    areas = pd.DataFrame([{"sample_id": r["sample_id"], "idx": i,
                          "normalised_area": (o["box"][2]-o["box"][0])*(o["box"][3]-o["box"][1])}
                         for r in rows for i, o in enumerate(r["objects"])])
    cutoff = float(areas.normalised_area.quantile(.25))
    sized = I.merge(areas, on=["sample_id", "idx"], validate="many_to_one")
    sized["relative_size"] = np.where(sized.normalised_area <= cutoff, "lower_quartile", "other")
    for group, frame in sized.groupby("relative_size"):
        for method, f in frame.groupby("method"):
            subgroup.append({"group_by": "relative_size", "group": group, "method": method,
                             **estimate(f, "official_success")})
    pd.DataFrame(subgroup).to_csv(out / "subgroup_intervals.csv", index=False)
    sized.to_csv(out / "test_instances_with_size.csv", index=False)

    vd = root / "outputs/results/val_finetune"
    V = pd.read_csv(vd / "instances.csv")
    validation = {m: estimate(f, "official_success") for m, f in V.groupby("method")}
    comparisons = {}
    names = [m for m in validation if m != "G0"]
    pairs = [(n, "G0") for n in names]
    pairs += [(f"{run}_s002000", f"{run}_s000500") for run in ["g1_lr5e-5", "g2_lr5e-5", "g2_lr1e-5"]]
    pairs += [(f"g2_lr5e-5_s{s}", f"g2_lr1e-5_s{s}") for s in ["000500", "002000"]]
    pairs += [(f"g2_lr5e-5_s{s}", f"g1_lr5e-5_s{s}") for s in ["000500", "002000"]]
    for a, b in pairs:
        comparisons[f"{a} minus {b}"] = pair(V[V.method == a], V[V.method == b], "official_success")
    report = {"bootstrap": {"unit": "layout_id", "n_draws": args.n_boot, "seed": 2026,
                             "interval": "95 percent percentile, unadjusted", "validation": "exploratory, not independent test evidence"},
              "test": main_results, "test_migc_minus_g0": paired_results,
              "validation": validation, "validation_paired": comparisons,
              "size_diagnostic": {"definition": "lower quartile of target normalised box area in the fixed test inputs; post hoc descriptive diagnostic",
                                  "normalised_area_cutoff": cutoff, "coco_small_at_512_count": int((areas.normalised_area < (32/512)**2).sum()),
                                  "target_instances_with_seeds": len(areas)},
              "subgroups": subgroup}
    (out / "statistics.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Analysis from committed evidence", "", "Layout bootstrap with 5000 draws by default; validation contrasts are exploratory and unadjusted.", "",
             "| Test metric | G0 | MIGC | MIGC minus G0 and 95% CI |", "|---|---:|---:|---|"]
    for metric in metrics:
        d = paired_results[metric]
        lines.append(f"| {metric} | {main_results['G0'][metric]['estimate']:.4f} | {main_results['MIGC'][metric]['estimate']:.4f} | {d['estimate']:+.4f} [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    lines += ["", "| Validation contrast | Difference | 95% CI |", "|---|---:|---|"]
    for contrast, d in comparisons.items():
        lines.append(f"| {contrast} | {d['estimate']:+.4f} | [{d['ci95'][0]:+.4f}, {d['ci95'][1]:+.4f}] |")
    (out / "analysis_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
