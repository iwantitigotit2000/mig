"""COCO train2014 box-grounding dataset producing the same batch fields as the official
GLIGEN TSVDataset (image, caption, boxes, masks, text_embeddings, ...).

The image/box processing calls the official functions `center_crop_arr` and
`recalculate_box_and_verify_if_valid` (GLIGEN/dataset/base_dataset.py); box selection
(largest `max_boxes_per_data`, min_box_size, random flip, caption dropout with
`prob_use_caption`) follows TSVDataset.__getitem__.
Phrase embeddings are the CLIP ViT-L/14 text pooler ("before" projection) of the COCO
category name, precomputed once with the official `get_clip_feature`.

Randomness (flip, caption choice, caption dropout) is drawn from a per-(epoch, index)
seeded RNG so a resumed run sees exactly the same data stream.
"""
import random
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import load_paths, read_jsonl  # noqa: E402

PATHS = load_paths()
if PATHS["repos"]["gligen"] not in sys.path:
    sys.path.insert(0, PATHS["repos"]["gligen"])
from dataset.base_dataset import center_crop_arr, recalculate_box_and_verify_if_valid  # noqa: E402


class CocoGroundingDataset(torch.utils.data.Dataset):
    def __init__(self, manifest, image_root, phrase_embeddings, image_size=512, max_boxes_per_data=30,
                 min_box_size=0.01, prob_use_caption=0.5, random_flip=True, seed=0):
        self.rows = read_jsonl(manifest)
        self.image_root = Path(image_root)
        self.emb = phrase_embeddings  # {phrase: tensor(768)}
        self.image_size = image_size
        self.max_boxes = max_boxes_per_data
        self.min_box_size = min_box_size
        self.prob_use_caption = prob_use_caption
        self.random_flip = random_flip
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        return len(self.rows)

    def set_epoch(self, e):
        self.epoch = e

    def __getitem__(self, key):
        # key = (epoch, index) from the resumable sampler, or a plain index
        epoch, index = key if isinstance(key, tuple) else (self.epoch, key)
        rng = random.Random(self.seed * 1_000_003 + epoch * 10_007 + index)
        r = self.rows[index]
        pil = Image.open(self.image_root / r["file_name"]).convert("RGB")
        arr, info = center_crop_arr(pil, self.image_size)
        info["performed_flip"] = False
        if self.random_flip and rng.random() < 0.5:
            arr = arr[:, ::-1]
            info["performed_flip"] = True
        img = torch.tensor(np.transpose(arr.astype(np.float32) / 127.5 - 1, [2, 0, 1]))

        areas, boxes, embs, names = [], [], [], []
        for o in r["objects"]:
            x, y, w, h = o["bbox_xywh"]
            valid, (x0, y0, x1, y1) = recalculate_box_and_verify_if_valid(
                x, y, w, h, info, self.image_size, self.min_box_size)
            if valid:
                areas.append((x1 - x0) * (y1 - y0))
                boxes.append(torch.tensor([x0, y0, x1, y1]) / self.image_size)
                embs.append(self.emb[o["phrase"]])
                names.append(o["phrase"])
        order = torch.tensor(areas).sort(descending=True)[1][:self.max_boxes] if areas else []
        B = torch.zeros(self.max_boxes, 4)
        M = torch.zeros(self.max_boxes)
        E = torch.zeros(self.max_boxes, 768)
        for i, j in enumerate(order):
            B[i], M[i], E[i] = boxes[j], 1, embs[j]
        caption = rng.choice(r["captions"]) if rng.random() < self.prob_use_caption else ""
        return {"id": r["image_id"], "image": img, "caption": caption, "boxes": B, "masks": M,
                "text_masks": M.clone(), "image_masks": torch.zeros_like(M), "text_embeddings": E,
                "n_boxes": int(M.sum())}


class ResumableSampler(torch.utils.data.Sampler):
    """Infinite stream of (epoch, index) keys; epoch e uses permutation seeded by (seed, e).
    `start` = number of keys already consumed, so resuming skips exactly the seen samples."""

    def __init__(self, n, seed, start=0):
        self.n, self.seed, self.start = n, seed, start

    def __iter__(self):
        e, off = divmod(self.start, self.n)
        while True:
            perm = list(range(self.n))
            random.Random(self.seed + 7919 * e).shuffle(perm)
            for i in perm[off:]:
                yield (e, i)
            e, off = e + 1, 0

    def __len__(self):
        return 2**31
