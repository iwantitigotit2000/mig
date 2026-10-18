"""Aggregate cached evaluation results into tables (traceable to sample_id and run config).

Usage:
  python analysis/aggregate.py --manifest manifests/cocomig_test.jsonl \
      --runs MIGC=outputs/gen/test/MIGC G0=outputs/gen/test/G0 ... --ref G0 --out outputs/results/test

Writes  instances.csv  images.csv  summary.json  summary.md  (+ grouped tables)
Metric naming:
  official_*  : MIG_Bench rule (max-IoU position check, SAM on GT box, colour >= 0.2, mIoU zeroes colour errors)
  loc_iou     : pure localisation IoU (no colour zeroing)          [added]
  o2o_*       : one-to-one Hungarian matching diagnostic           [added]
CIs: 95% percentile bootstrap over layouts (both seeds of a layout resampled together).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import dump_json, read_jsonl  # noqa: E402

N_BOOT = 1000


def load_run(name, gen_dir, manifest_rows):
    gd = Path(gen_dir)
    iqa_p = gd / "eval_cache" / "clipiqa.json"
    iqa = json.load(open(iqa_p)) if iqa_p.exists() else {}
    inst_rows, img_rows, missing = [], [], []
    for s in manifest_rows:
        sid = s["sample_id"]
        ev_p = gd / "eval_cache" / f"{sid}.json"
        meta_p = gd / "meta" / f"{sid}.json"
        if not ev_p.exists() or not meta_p.exists():
            missing.append(sid)
            continue
        ev = json.load(open(ev_p))
        meta = json.load(open(meta_p))
        insts = ev["instances"]
        for o, r in zip(s["objects"], insts):
            inst_rows.append({
                "method": name, "sample_id": sid, "layout_id": s["layout_id"], "seed": s["seed"],
                "level": s["level"], "idx": r["idx"], "desc": o["desc"], "category": o["category"],
                "color": o["color"], "overlap_bucket": o["overlap_bucket"], "max_iou_other": o["max_iou_other"],
                "same_cat_diff_color": o["same_cat_diff_color"], "status": r.get("status", "ok"),
                "present_any": int(r.get("present_any", False)), "loc_iou": r.get("loc_iou", 0.0),
                "official_pos": r["official_pos"], "official_color": r["official_color"],
                "official_success": r["official_success"], "official_miou": r["official_miou"],
                "o2o_present": r["o2o_present"], "o2o_pos": r["o2o_pos"], "o2o_iou": r["o2o_iou"],
                "o2o_success": r["o2o_success"], "failure": r.get("failure"),
                "crosstalk": int(bool(r.get("crosstalk", False))), "dominant_color": r.get("dominant_color"),
                "target_color_ratio": (r.get("color_ratios") or {}).get(o["color"]),
                "empty_mask": int(bool(r.get("empty_mask", False))),
            })
        img_rows.append({
            "method": name, "sample_id": sid, "layout_id": s["layout_id"], "seed": s["seed"], "level": s["level"],
            "overlap_bucket": s["overlap_bucket"], "has_same_cat_diff_color": s["has_same_cat_diff_color"],
            "official_image_success": int(all(r["official_success"] for r in insts)),
            "o2o_image_success": int(all(r["o2o_success"] for r in insts)),
            "all_present_o2o": int(all(r["o2o_present"] for r in insts)),
            "clipiqa": iqa.get(sid, np.nan), "seconds": meta["seconds"],
            "peak_mem_allocated_gb": meta["peak_mem_allocated_gb"],
            "n_eval_errors": sum(r.get("status") == "error" for r in insts),
        })
    return pd.DataFrame(inst_rows), pd.DataFrame(img_rows), missing


def metrics(I, M):
    """I: instance rows, M: image rows (same method)."""
    out = {
        "n_images": len(M), "n_instances": len(I),
        "official_instance_success": I.official_success.mean(),
        "official_image_success": M.official_image_success.mean(),
        "official_miou": I.official_miou.mean(),
        "official_position_rate": I.official_pos.mean(),
        "official_colour_given_position": I.loc[I.official_pos == 1, "official_color"].mean() if (I.official_pos == 1).any() else np.nan,
        "loc_iou": I.loc_iou.mean(),
        "presence_any": I.present_any.mean(),
        "o2o_presence": I.o2o_present.mean(),
        "o2o_position_rate": I.o2o_pos.mean(),
        "o2o_iou": I.o2o_iou.mean(),
        "o2o_instance_success": I.o2o_success.mean(),
        "o2o_image_success": M.o2o_image_success.mean(),
        "fail_missing": (I.failure == "missing").mean(),
        "fail_mislocated": (I.failure == "mislocated").mean(),
        "fail_colour_wrong": (I.failure == "colour_wrong").mean(),
        "crosstalk_share_of_colour_wrong": I.loc[I.failure == "colour_wrong", "crosstalk"].mean() if (I.failure == "colour_wrong").any() else np.nan,
        "eval_errors": int((I.status == "error").sum()),
        "clipiqa": M.clipiqa.mean(),
        "sec_per_image": M.seconds.mean(),
        "peak_mem_allocated_gb_max": M.peak_mem_allocated_gb.max(),
    }
    return {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in out.items()}


CI_KEYS = ["official_instance_success", "official_image_success", "official_miou", "loc_iou",
           "o2o_instance_success", "o2o_presence", "clipiqa"]


def bootstrap(I, M, rng, keys=CI_KEYS):
    layouts = M.layout_id.unique()
    Ig, Mg = dict(tuple(I.groupby("layout_id"))), dict(tuple(M.groupby("layout_id")))
    vals = {k: [] for k in keys}
    for _ in range(N_BOOT):
        pick = rng.choice(layouts, len(layouts), replace=True)
        Ib = pd.concat([Ig[l] for l in pick])
        Mb = pd.concat([Mg[l] for l in pick])
        m = metrics(Ib, Mb)
        for k in keys:
            vals[k].append(m[k])
    return {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for k, v in vals.items()}


def paired_diff(Ia, Ma, Ib, Mb, rng, keys=CI_KEYS):
    """method a minus method b on identical samples; bootstrap over layouts."""
    common = sorted(set(Ma.sample_id) & set(Mb.sample_id))
    Ia, Ib = Ia[Ia.sample_id.isin(common)], Ib[Ib.sample_id.isin(common)]
    Ma, Mb = Ma[Ma.sample_id.isin(common)], Mb[Mb.sample_id.isin(common)]
    point = {k: metrics(Ia, Ma)[k] - metrics(Ib, Mb)[k] for k in keys}
    layouts = Ma.layout_id.unique()
    g = [dict(tuple(x.groupby("layout_id"))) for x in (Ia, Ma, Ib, Mb)]
    vals = {k: [] for k in keys}
    for _ in range(N_BOOT):
        pick = rng.choice(layouts, len(layouts), replace=True)
        ma = metrics(pd.concat([g[0][l] for l in pick]), pd.concat([g[1][l] for l in pick]))
        mb = metrics(pd.concat([g[2][l] for l in pick]), pd.concat([g[3][l] for l in pick]))
        for k in keys:
            vals[k].append(ma[k] - mb[k])
    return {k: {"diff": float(point[k]), "ci95": [float(np.nanpercentile(vals[k], 2.5)),
                                                 float(np.nanpercentile(vals[k], 97.5))]} for k in keys}, len(common)


def grouped(I, M, by_inst, by_img=None):
    rows = []
    for key, Ig in I.groupby(by_inst):
        Mg = M[M.sample_id.isin(Ig.sample_id)] if by_img is None else M[M[by_img] == key]
        r = {by_inst: key, "n_instances": len(Ig)}
        r["official_instance_success"] = Ig.official_success.mean()
        r["official_miou"] = Ig.official_miou.mean()
        r["loc_iou"] = Ig.loc_iou.mean()
        r["o2o_presence"] = Ig.o2o_present.mean()
        r["o2o_instance_success"] = Ig.o2o_success.mean()
        r["fail_missing"] = (Ig.failure == "missing").mean()
        r["fail_mislocated"] = (Ig.failure == "mislocated").mean()
        r["fail_colour_wrong"] = (Ig.failure == "colour_wrong").mean()
        if by_img is not None:
            r["official_image_success"] = Mg.official_image_success.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--runs", nargs="+", required=True, help="name=gen_dir")
    ap.add_argument("--ref", default=None, help="reference run for paired differences")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no_ci", action="store_true")
    args = ap.parse_args()
    rows = read_jsonl(args.manifest)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    Is, Ms, summary = [], [], {"manifest": args.manifest, "runs": {}, "paired_vs_ref": {}}
    data = {}
    for spec in args.runs:
        name, gd = spec.split("=", 1)
        I, M, missing = load_run(name, gd, rows)
        data[name] = (I, M)
        Is.append(I)
        Ms.append(M)
        cfgs = sorted(Path(gd).glob("run_config_*.json"))
        s = {"gen_dir": gd, "missing_samples": missing, "run_configs": [str(c) for c in cfgs],
             "overall": metrics(I, M)}
        if not args.no_ci and len(M):
            s["ci95"] = bootstrap(I, M, rng)
        summary["runs"][name] = s
    if args.ref and not args.no_ci:
        Ir, Mr = data[args.ref]
        for name, (I, M) in data.items():
            if name != args.ref:
                d, n = paired_diff(I, M, Ir, Mr, rng)
                summary["paired_vs_ref"][f"{name}-{args.ref}"] = {"n_common_images": n, **d}
    I = pd.concat(Is)
    M = pd.concat(Ms)
    I.to_csv(out / "instances.csv", index=False)
    M.to_csv(out / "images.csv", index=False)
    for name, (Ii, Mi) in data.items():
        for col, img_col in [("level", "level"), ("overlap_bucket", None), ("same_cat_diff_color", None)]:
            g = grouped(Ii, Mi, col, img_col)
            g.insert(0, "method", name)
            g.to_csv(out / f"by_{col}_{name}.csv", index=False)
    dump_json(out / "summary.json", summary)

    # markdown overview
    keys = ["n_images", "official_instance_success", "official_image_success", "official_miou", "loc_iou",
            "presence_any", "o2o_presence", "o2o_instance_success", "fail_missing", "fail_mislocated",
            "fail_colour_wrong", "crosstalk_share_of_colour_wrong", "eval_errors", "clipiqa", "sec_per_image",
            "peak_mem_allocated_gb_max"]
    lines = ["| metric | " + " | ".join(summary["runs"]) + " |", "|---|" + "---|" * len(summary["runs"])]
    for k in keys:
        cells = []
        for name, s in summary["runs"].items():
            v = s["overall"][k]
            cell = f"{v:.3f}" if isinstance(v, float) else str(v)
            if "ci95" in s and k in s["ci95"]:
                lo, hi = s["ci95"][k]
                cell += f" [{lo:.3f}, {hi:.3f}]"
            cells.append(cell)
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if summary["paired_vs_ref"]:
        lines += ["", "Paired differences (same samples), 95% layout-bootstrap CI:", "",
                  "| comparison | metric | diff | CI95 |", "|---|---|---|---|"]
        for comp, d in summary["paired_vs_ref"].items():
            for k in CI_KEYS:
                lines.append(f"| {comp} | {k} | {d[k]['diff']:+.3f} | [{d[k]['ci95'][0]:+.3f}, {d[k]['ci95'][1]:+.3f}] |")
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
