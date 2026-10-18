"""Build fixed COCO-MIG sample manifests (pilot / val / test) from the official bench file.

Source: limuloo/MIGC bench_file/mig_bench_anno.yaml (800 layouts, boxes are normalised
[x0, y0, x1, y1]). The yaml groups boxes by phrase, so object order is restored from the
caption "a photo of <p1> and <p2> ...". `layout_id` is the bench's own `coco_id` field,
which is the index in the sampled bench (0..799), not a COCO image id.

Splits are disjoint at the layout level and stratified by object count (L2..L6):
  val   : 6 layouts / level  (30)  -> hyper-parameter selection only
  pilot : first 2 val layouts / level (10) -> stage-A pipeline check
  test  : 20 layouts / level (100) -> never used for selection
Each layout is paired with the same two seeds for every method.
"""
import argparse
import itertools
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import PROJECT_ROOT, dump_json, load_paths, write_jsonl  # noqa: E402

COLORS = ["red", "blue", "green", "yellow", "black", "white", "brown"]  # MIG_Bench color_list


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ios(a, b):
    """intersection over the smaller box (captures containment that IoU underrates)."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    sa = (a[2] - a[0]) * (a[3] - a[1])
    sb = (b[2] - b[0]) * (b[3] - b[1])
    return ix * iy / max(min(sa, sb), 1e-9)


def overlap_bucket(v):
    # thresholds on max pairwise IoU, fixed before looking at any generation result
    if v <= 0.0:
        return "none"
    if v < 0.1:
        return "low"
    if v < 0.3:
        return "mid"
    return "high"


def parse_layouts(anno):
    layouts = []
    for caption, entry in anno.items():
        phrases = caption[len("a photo of "):].split(" and ")
        pools = {k: list(v) for k, v in entry.items() if k != "coco_id"}
        objs = []
        for p in phrases:
            box = pools[p].pop(0)
            words = p.split(" ")
            assert words[0] == "a" and words[1] in COLORS, p
            objs.append({"desc": p, "color": words[1], "category": " ".join(words[2:]),
                         "box": [float(x) for x in box]})
        assert all(len(v) == 0 for v in pools.values()), caption
        n = len(objs)
        pair_iou = [[0.0] * n for _ in range(n)]
        for i, j in itertools.combinations(range(n), 2):
            pair_iou[i][j] = pair_iou[j][i] = iou(objs[i]["box"], objs[j]["box"])
        for i, o in enumerate(objs):
            others = [j for j in range(n) if j != i]
            o["max_iou_other"] = max(pair_iou[i][j] for j in others)
            o["max_ios_other"] = max(ios(o["box"], objs[j]["box"]) for j in others)
            o["overlap_bucket"] = overlap_bucket(o["max_iou_other"])
            o["same_cat_diff_color"] = any(objs[j]["category"] == o["category"] and objs[j]["color"] != o["color"]
                                           for j in others)
            o["same_cat_any"] = any(objs[j]["category"] == o["category"] for j in others)
        max_iou = max(o["max_iou_other"] for o in objs)
        layouts.append({
            "layout_id": str(entry["coco_id"]),
            "caption": caption,
            "objects": objs,
            "n_objects": n,
            "level": f"L{n}",
            "max_pair_iou": max_iou,
            "overlap_bucket": overlap_bucket(max_iou),
            "has_same_cat_diff_color": any(o["same_cat_diff_color"] for o in objs),
        })
    return layouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno", default=None)
    ap.add_argument("--split_seed", type=int, default=2024)
    ap.add_argument("--n_val_per_level", type=int, default=6)
    ap.add_argument("--n_pilot_per_level", type=int, default=2)
    ap.add_argument("--n_test_per_level", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 2025])
    ap.add_argument("--out", default=str(PROJECT_ROOT / "manifests"))
    args = ap.parse_args()

    paths = load_paths()
    anno_path = args.anno or str(Path(paths["repos"]["migc"]) / "bench_file" / "mig_bench_anno.yaml")
    anno = yaml.safe_load(open(anno_path, encoding="utf-8"))
    layouts = parse_layouts(anno)
    assert len({l["layout_id"] for l in layouts}) == len(layouts)

    rng = random.Random(args.split_seed)
    by_level = {}
    for l in sorted(layouts, key=lambda x: int(x["layout_id"])):
        by_level.setdefault(l["level"], []).append(l)
    val, pilot, test = [], [], []
    for lvl in sorted(by_level):
        pool = by_level[lvl][:]
        rng.shuffle(pool)
        v = pool[:args.n_val_per_level]
        t = pool[args.n_val_per_level:args.n_val_per_level + args.n_test_per_level]
        val += v
        pilot += v[:args.n_pilot_per_level]
        test += t
    assert not ({l["layout_id"] for l in val} & {l["layout_id"] for l in test})

    def expand(ls, split):
        rows = []
        for l in ls:
            for s in args.seeds:
                rows.append({"sample_id": f"mig{int(l['layout_id']):03d}_s{s}", "split": split, "seed": s, **l})
        return rows

    out = Path(args.out)
    for name, ls in [("pilot", pilot), ("val", val), ("test", test)]:
        write_jsonl(out / f"cocomig_{name}.jsonl", expand(ls, name))

    def stats(ls):
        inst = [o for l in ls for o in l["objects"]]
        return {
            "layouts": len(ls),
            "instances": len(inst),
            "levels": dict(sorted(Counter(l["level"] for l in ls).items())),
            "layout_overlap_bucket": dict(Counter(l["overlap_bucket"] for l in ls)),
            "instance_overlap_bucket": dict(Counter(o["overlap_bucket"] for o in inst)),
            "layouts_with_same_cat_diff_color": sum(l["has_same_cat_diff_color"] for l in ls),
            "instances_same_cat_diff_color": sum(o["same_cat_diff_color"] for o in inst),
            "colors": dict(Counter(o["color"] for o in inst)),
        }

    summary = {
        "source": anno_path,
        "split_seed": args.split_seed,
        "seeds": args.seeds,
        "all_800": stats(layouts),
        "pilot": stats(pilot), "val": stats(val), "test": stats(test),
        "overlap_bucket_rule": "max IoU with any other box: none=0, low<0.1, mid<0.3, high>=0.3",
    }
    dump_json(out / "cocomig_split_summary.json", summary)
    print(yaml.safe_dump({k: summary[k] for k in ("pilot", "val", "test")}, sort_keys=False))


if __name__ == "__main__":
    main()
