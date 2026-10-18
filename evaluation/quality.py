"""CLIP-IQA (pyiqa 'clipiqa', zero-shot CLIP antonym prompts) as an auxiliary image-quality score.

It says nothing about whether objects, positions or colours are correct; it is reported
separately from the task metrics. Scores are cached in <gen_dir>/eval_cache/clipiqa.json.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import dump_json  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, nargs="+")
    args = ap.parse_args()
    import pyiqa
    metric = None
    for gd in args.gen_dir:
        gd = Path(gd)
        cache_p = gd / "eval_cache" / "clipiqa.json"
        cache = json.load(open(cache_p)) if cache_p.exists() else {}
        imgs = sorted((gd / "images").glob("*.png"))
        todo = [p for p in imgs if p.stem not in cache]
        if not todo:
            continue
        if metric is None:
            metric = pyiqa.create_metric("clipiqa", device="cuda")
        with torch.no_grad():
            for p in todo:
                cache[p.stem] = float(metric(str(p)).item())
        dump_json(cache_p, cache)
        print(f"{gd}: {len(todo)} scored, mean={sum(cache.values()) / len(cache):.4f}")


if __name__ == "__main__":
    main()
