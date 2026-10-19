"""Per-object difficulty annotations for the fixed COCO-MIG manifests (side file; manifests untouched).

Adds the missing "small object" condition next to the existing same-category-different-colour
and box-overlap annotations. Two definitions are written so the report can state which one a
table uses:

  size_bucket_coco   : COCO convention evaluated at the 512x512 generation resolution
                       small  < 32^2 px, medium < 96^2 px, large otherwise
  relative_size_q    : quartile of normalised box area within the split (Q1 = smallest 25 %)

Also records box aspect ratio and whether the box touches the image border. Everything is
derived from the manifest inputs only (no generation results), so it is fixed before any
model output is inspected.

Usage:  python data_tools/annotate_difficulty.py
Writes: manifests/cocomig_difficulty.csv, manifests/cocomig_difficulty_summary.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import PROJECT_ROOT, read_jsonl  # noqa: E402

RES = 512


def coco_bucket(area_px):
    if area_px < 32 ** 2:
        return "small"
    if area_px < 96 ** 2:
        return "medium"
    return "large"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    args = ap.parse_args()
    mdir = args.root.resolve() / "manifests"
    rows = []
    for split in ["pilot", "val", "test"]:
        for s in read_jsonl(mdir / f"cocomig_{split}.jsonl"):
            for i, o in enumerate(s["objects"]):
                x0, y0, x1, y1 = o["box"]
                w, h = max(x1 - x0, 0), max(y1 - y0, 0)
                area = w * h
                rows.append({"split": split, "sample_id": s["sample_id"], "layout_id": s["layout_id"], "seed": s["seed"],
                             "idx": i, "level": s["level"], "n_objects": s["n_objects"], "category": o["category"],
                             "color": o["color"], "normalised_area": area, "area_px_512": area * RES * RES,
                             "side_px_512": np.sqrt(area) * RES, "aspect_ratio": (w / h) if h > 0 else np.inf,
                             "touches_border": bool(x0 <= 1e-3 or y0 <= 1e-3 or x1 >= 1 - 1e-3 or y1 >= 1 - 1e-3),
                             "overlap_bucket": o["overlap_bucket"], "max_iou_other": o["max_iou_other"],
                             "same_cat_diff_color": bool(o["same_cat_diff_color"])})
    df = pd.DataFrame(rows)
    df["size_bucket_coco"] = df.area_px_512.map(coco_bucket)
    # quartiles over unique (layout, idx) so the two seeds do not double-count
    df["relative_size_q"] = ""
    cutoffs = {}
    for split, g in df.groupby("split"):
        uniq = g.drop_duplicates(["layout_id", "idx"]).normalised_area
        q = uniq.quantile([.25, .5, .75]).to_numpy()
        cutoffs[split] = {"q25": float(q[0]), "q50": float(q[1]), "q75": float(q[2])}
        df.loc[g.index, "relative_size_q"] = pd.cut(g.normalised_area, [-np.inf, *q, np.inf],
                                                    labels=["Q1_smallest", "Q2", "Q3", "Q4_largest"]).astype(str)
    df.to_csv(mdir / "cocomig_difficulty.csv", index=False)
    summary = {"resolution": RES, "definitions": {"size_bucket_coco": "COCO small/medium/large thresholds (32^2, 96^2 px) at 512x512",
                                                  "relative_size_q": "quartiles of normalised box area within each split (unique layouts)"},
               "quartile_cutoffs_normalised_area": cutoffs, "splits": {}}
    for split, g in df.groupby("split"):
        u = g.drop_duplicates(["layout_id", "idx"])
        summary["splits"][split] = {"objects_unique": len(u), "objects_with_seeds": len(g),
                                    "size_bucket_coco": u.size_bucket_coco.value_counts().to_dict(),
                                    "touches_border": int(u.touches_border.sum()),
                                    "median_side_px_512": float(u.side_px_512.median()),
                                    "min_side_px_512": float(u.side_px_512.min())}
    (mdir / "cocomig_difficulty_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
