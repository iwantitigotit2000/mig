"""Select and download a small COCO train2014 subset for GLIGEN fine-tuning.

Source isolation: COCO-MIG layouts (val / test) are sampled from COCO *val2014*
(MIG_Bench/bench_resample.md), so fine-tuning data is drawn only from *train2014*,
which is disjoint from val2014 by construction. The script also asserts that no selected
file name belongs to val2014.

Annotations used per image (all from the real image, nothing synthesised):
  caption : the 5 human COCO captions (one is sampled per step during training)
  objects : every non-crowd instance box with its COCO *category name* as the phrase.
COCO has no colour labels, so this data carries NO colour/attribute supervision.

Image selection (to resemble COCO-MIG layouts; decided before training):
  * no 'person' instance (COCO-MIG removes person layouts)
  * 2..6 instances whose normalised w and h are both >= 1/8 (the bench's size filter)
Split: train / train_val (held-out loss + sample images), disjoint, fixed seed.
"""
import argparse
import json
import random
import sys
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import PROJECT_ROOT, dump_json, load_paths, write_jsonl  # noqa: E402

URL = "http://images.cocodataset.org/train2014/{}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_train", type=int, default=3000)
    ap.add_argument("--n_trainval", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--no_download", action="store_true")
    args = ap.parse_args()

    paths = load_paths()
    ann_dir = Path(paths["coco"]["root"]) / "annotations"
    inst = json.load(open(ann_dir / "instances_train2014.json"))
    caps = json.load(open(ann_dir / "captions_train2014.json"))
    val_names = {im["file_name"] for im in json.load(open(ann_dir / "instances_val2014.json"))["images"]}

    cats = {c["id"]: c["name"] for c in inst["categories"]}
    images = {im["id"]: im for im in inst["images"]}
    anns = defaultdict(list)
    for a in inst["annotations"]:
        anns[a["image_id"]].append(a)
    captions = defaultdict(list)
    for c in caps["annotations"]:
        captions[c["image_id"]].append(c["caption"].strip())

    eligible = []
    for iid, im in images.items():
        a = anns.get(iid, [])
        if not a or any(cats[x["category_id"]] == "person" for x in a):
            continue
        W, H = im["width"], im["height"]
        big = [x for x in a if not x["iscrowd"] and x["bbox"][2] / W >= 1 / 8 and x["bbox"][3] / H >= 1 / 8]
        if 2 <= len(big) <= 6 and captions.get(iid):
            eligible.append(iid)
    eligible.sort()
    rng = random.Random(args.seed)
    rng.shuffle(eligible)
    chosen = eligible[:args.n_train + args.n_trainval]

    rows = []
    for k, iid in enumerate(chosen):
        im = images[iid]
        assert im["file_name"] not in val_names
        objs = [{"category": cats[x["category_id"]], "phrase": cats[x["category_id"]],
                 "bbox_xywh": x["bbox"]} for x in anns[iid] if not x["iscrowd"]]
        rows.append({"image_id": iid, "file_name": im["file_name"], "width": im["width"], "height": im["height"],
                     "captions": captions[iid], "objects": objs,
                     "split": "train" if k < args.n_train else "train_val"})

    out = PROJECT_ROOT / "manifests"
    write_jsonl(out / "coco_train.jsonl", [r for r in rows if r["split"] == "train"])
    write_jsonl(out / "coco_trainval.jsonl", [r for r in rows if r["split"] == "train_val"])
    dump_json(out / "coco_train_summary.json", {
        "source": "COCO train2014 (instances_train2014.json, captions_train2014.json)",
        "eligible_images": len(eligible), "n_train": args.n_train, "n_trainval": args.n_trainval,
        "seed": args.seed, "overlap_with_val2014_file_names": 0,
        "mean_objects_per_image": sum(len(r["objects"]) for r in rows) / len(rows),
        "attribute_supervision": "none (phrases are COCO category names only)",
    })
    print(f"eligible={len(eligible)} selected={len(rows)}")

    if args.no_download:
        return
    img_dir = Path(paths["coco"]["train_images"])
    img_dir.mkdir(parents=True, exist_ok=True)

    def fetch(r):
        dst = img_dir / r["file_name"]
        if dst.exists() and dst.stat().st_size > 0:
            return 0
        tmp = dst.with_suffix(".part")
        urllib.request.urlretrieve(URL.format(r["file_name"]), tmp)
        tmp.replace(dst)
        return 1

    with ThreadPoolExecutor(args.workers) as ex:
        n = sum(ex.map(fetch, rows))
    missing = [r["file_name"] for r in rows if not (img_dir / r["file_name"]).exists()]
    print(f"downloaded={n} missing={len(missing)}")
    assert not missing


if __name__ == "__main__":
    main()
