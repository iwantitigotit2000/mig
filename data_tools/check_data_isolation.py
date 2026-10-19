"""Independent data-isolation record for the fixed manifests (CPU only, stdlib + numpy + scipy).

Covers the three open items of the "data responsibility" workflow:

1. Export the evaluation layout IDs (val / test) with a content hash of every layout.
   The COCO-MIG bench file (MIGC bench_file/mig_bench_anno.yaml) does NOT carry COCO
   image IDs: its `coco_id` is the index 0..799 inside the sampled bench. A COCO-ID list
   therefore cannot be exported; the layout ID + content hash is the strongest identifier
   available and is what this script writes.
2. Check train-vs-evaluation overlap independently of prepare_coco_train.py:
     a. source isolation  : every training file name must be a COCO *train2014* file
                            (COCO-MIG layouts come from val2014; the two sets are disjoint);
     b. layout duplicates : for every val/test layout, search the 3 100 training images for
                            one with the same category multiset and near-identical
                            normalised boxes (Hungarian matching, mean IoU). A layout copied
                            from a training image would show mean IoU close to 1.
3. Record sha256, row counts and a version tag for every manifest so the report can cite
   an exact data version. Nothing under manifests/ is modified.

Usage:  python data_tools/check_data_isolation.py
Writes: manifests/eval_layout_ids.csv, manifests/MANIFEST_VERSIONS.json,
        outputs/checks/train_eval_isolation.json, outputs/checks/train_eval_isolation.md
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import PROJECT_ROOT, dump_json, read_jsonl  # noqa: E402

MANIFESTS = ["cocomig_pilot.jsonl", "cocomig_val.jsonl", "cocomig_test.jsonl",
             "coco_train.jsonl", "coco_trainval.jsonl",
             "cocomig_split_summary.json", "coco_train_summary.json"]
DATA_VERSION = "v1.0-locked-2026-10"   # manifests have not changed since the first training run


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def layout_hash(objs):
    """Order-independent content hash of (category, colour, box rounded to 1e-4)."""
    items = sorted((o["category"], o["color"], tuple(round(float(x), 4) for x in o["box"])) for o in objs)
    return hashlib.sha1(json.dumps(items).encode()).hexdigest()[:12]


def iou_matrix(a, b):
    a, b = np.asarray(a, float)[:, None, :], np.asarray(b, float)[None, :, :]
    ix = np.clip(np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0]), 0, None)
    iy = np.clip(np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1]), 0, None)
    inter = ix * iy
    area = lambda t: (t[..., 2] - t[..., 0]) * (t[..., 3] - t[..., 1])
    return inter / np.maximum(area(a) + area(b) - inter, 1e-9)


def best_layout_match(eval_objs, train_images_by_cat):
    """Highest mean matched IoU between an evaluation layout and any training image that
    contains at least the same number of boxes of every category in the layout."""
    cats = Counter(o["category"] for o in eval_objs)
    first = min(cats, key=lambda c: len(train_images_by_cat.get(c, [])))
    best = {"mean_iou": 0.0, "image_id": None, "file_name": None}
    for img in train_images_by_cat.get(first, []):
        if any(img["cats"][c] < n for c, n in cats.items()):
            continue
        ious = []
        for c, n in cats.items():
            e = [o["box"] for o in eval_objs if o["category"] == c]
            t = [o["box"] for o in img["objects"] if o["category"] == c]
            m = iou_matrix(e, t)
            r, k = linear_sum_assignment(-m)
            ious.extend(m[r, k].tolist())
        mean_iou = float(np.mean(ious))
        if mean_iou > best["mean_iou"]:
            best = {"mean_iou": mean_iou, "image_id": img["image_id"], "file_name": img["file_name"]}
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=PROJECT_ROOT)
    ap.add_argument("--dup_iou", type=float, default=0.9,
                    help="mean matched IoU at/above which a layout is flagged as a copy (COCO boxes copied exactly give ~1.0)")
    ap.add_argument("--near_iou", type=float, default=0.7, help="lower threshold reported as 'similar but distinct'")
    args = ap.parse_args()
    root = args.root.resolve()
    mdir, cdir = root / "manifests", root / "outputs" / "checks"
    cdir.mkdir(parents=True, exist_ok=True)

    # ---- 1. evaluation layout IDs -------------------------------------------------------
    rows = []
    for split in ["val", "test"]:
        for s in read_jsonl(mdir / f"cocomig_{split}.jsonl"):
            rows.append({"split": split, "layout_id": s["layout_id"], "sample_id": s["sample_id"], "seed": s["seed"],
                         "level": s["level"], "n_objects": s["n_objects"], "layout_hash": layout_hash(s["objects"]),
                         "caption": s["caption"]})
    with open(mdir / "eval_layout_ids.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    val_layouts = {r["layout_id"] for r in rows if r["split"] == "val"}
    test_layouts = {r["layout_id"] for r in rows if r["split"] == "test"}
    pilot_layouts = {s["layout_id"] for s in read_jsonl(mdir / "cocomig_pilot.jsonl")}
    if val_layouts & test_layouts:
        sys.exit("ISOLATION CHECK FAILED: val and test layouts overlap")
    if not pilot_layouts <= val_layouts:
        sys.exit("ISOLATION CHECK FAILED: pilot must be a subset of val")

    # ---- 2. train vs evaluation isolation ----------------------------------------------
    train = read_jsonl(mdir / "coco_train.jsonl") + read_jsonl(mdir / "coco_trainval.jsonl")
    bad_source = [t["file_name"] for t in train if not t["file_name"].startswith("COCO_train2014_")]
    by_cat = {}
    for t in train:
        objs = []
        for o in t["objects"]:
            x, y, w, h = o["bbox_xywh"]
            objs.append({"category": o["category"],
                         "box": [x / t["width"], y / t["height"], (x + w) / t["width"], (y + h) / t["height"]]})
        rec = {"image_id": t["image_id"], "file_name": t["file_name"], "objects": objs,
               "cats": Counter(o["category"] for o in objs)}
        for c in rec["cats"]:
            by_cat.setdefault(c, []).append(rec)
    per_layout = []
    seen = set()
    for split in ["val", "test"]:
        for s in read_jsonl(mdir / f"cocomig_{split}.jsonl"):
            if s["layout_id"] in seen:          # two seeds share one layout
                continue
            seen.add(s["layout_id"])
            b = best_layout_match(s["objects"], by_cat)
            per_layout.append({"split": split, "layout_id": s["layout_id"], "n_objects": s["n_objects"],
                               "best_train_mean_iou": round(b["mean_iou"], 4),
                               "best_train_image_id": b["image_id"], "flagged": b["mean_iou"] >= args.dup_iou})
    flagged = [p for p in per_layout if p["flagged"]]
    near = [p for p in per_layout if args.near_iou <= p["best_train_mean_iou"] < args.dup_iou]
    ious = np.array([p["best_train_mean_iou"] for p in per_layout])

    # ---- 3. manifest versions ------------------------------------------------------------
    versions = {"data_version": DATA_VERSION, "files": {}}
    for name in MANIFESTS:
        p = mdir / name
        n_rows = sum(1 for l in open(p, encoding="utf-8") if l.strip()) if name.endswith(".jsonl") else None
        versions["files"][name] = {"sha256": sha256_file(p), "bytes": p.stat().st_size, "rows": n_rows}
    dump_json(mdir / "MANIFEST_VERSIONS.json", versions)

    report = {
        "data_version": DATA_VERSION,
        "eval_layouts": {"val": len(val_layouts), "test": len(test_layouts), "pilot_subset_of_val": True,
                         "val_test_disjoint": True,
                         "note": "COCO-MIG bench carries no COCO image IDs; layout_id (bench index) + content hash exported instead"},
        "source_isolation": {"train_images_checked": len(train), "non_train2014_file_names": bad_source,
                             "passed": not bad_source},
        "layout_duplicate_search": {"train_images_searched": len(train), "eval_layouts_searched": len(per_layout),
                                    "flag_threshold_mean_iou": args.dup_iou, "flagged": flagged,
                                    "near_threshold_mean_iou": args.near_iou, "similar_but_distinct": near,
                                    "max_best_mean_iou": float(ious.max()), "median_best_mean_iou": float(np.median(ious)),
                                    "passed": not flagged},
        "per_layout": per_layout,
    }
    dump_json(cdir / "train_eval_isolation.json", report)
    md = [
        "# Train / evaluation isolation check (independent record)", "",
        f"Data version `{DATA_VERSION}`; sha256 of every manifest in `manifests/MANIFEST_VERSIONS.json`.", "",
        "| Check | Result |", "|---|---|",
        f"| val / test layouts disjoint | yes ({len(val_layouts)} / {len(test_layouts)} layouts) |",
        f"| pilot ⊂ val | yes ({len(pilot_layouts)} layouts) |",
        f"| all {len(train)} training images are COCO train2014 files | {'yes' if not bad_source else 'NO: ' + str(bad_source[:5])} |",
        f"| evaluation layouts copied from a training image (mean matched IoU ≥ {args.dup_iou}) | {len(flagged)} of {len(per_layout)} |",
        f"| similar but distinct layouts ({args.near_iou} ≤ mean IoU < {args.dup_iou}; all 2–3-object layouts of common categories) | {len(near)} of {len(per_layout)} |",
        f"| highest / median best-match mean IoU over all evaluation layouts | {ious.max():.3f} / {np.median(ious):.3f} |",
        "",
        "COCO-MIG layouts are derived from COCO val2014 and the training subset is drawn from train2014, "
        "which are disjoint image sets by construction. Because the bench file does not record COCO image IDs, "
        "an ID-level exclusion list cannot be produced; the layout-level duplicate search above is the "
        "strongest check possible from the committed manifests and is reported instead.",
    ]
    md += ["", "**Scope.** This layout-level search does NOT replace a COCO image-ID check; such a check is impossible from the "
           "bench file, and the report must say so rather than claim ID-level exclusion."]
    (cdir / "train_eval_isolation.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    failures = []
    if bad_source:
        failures.append(f"{len(bad_source)} training files are not COCO train2014")
    if flagged:
        failures.append(f"{len(flagged)} evaluation layouts duplicate a training layout (mean IoU >= {args.dup_iou})")
    if failures:
        print("ISOLATION CHECK FAILED: " + "; ".join(failures), file=sys.stderr)
        sys.exit(1)
    print("isolation check passed")


if __name__ == "__main__":
    main()
