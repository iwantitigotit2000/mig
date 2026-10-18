"""COCO-MIG automatic evaluation (Grounding DINO + SAM + HSV colour), with caching.

OFFICIAL rule (LeyRio/MIG_Bench @ 063ec45, eval_mig.py::check_on_image), reproduced exactly:
  * query Grounding DINO (SwinT-OGC) with the class name only, box_t = text_t = 0.25, NMS 0.8
  * position ok  <=> max IoU(GT box, any detection) >= 0.5   (no one-to-one constraint)
  * SAM (ViT-H) prompted with the *GT box*, multimask, best-score mask
  * colour ok    <=> |mask & inRange(HSV, colour)| / |mask| >= 0.2   (official colour table)
  * official mIoU contribution: 0 if no detection; max IoU if position fails; 0 if colour
    fails; max IoU if both pass (i.e. colour errors are zeroed, position errors are not)
  * instance success = position ok & colour ok ; image success = all instances succeed

ADDED diagnostics (named differently, never mixed with the official numbers):
  * loc_iou        : max IoU with any detection of the class (0 if none), no colour zeroing
  * present_any    : >= 1 detection of the class anywhere in the image
  * o2o_*          : one-to-one Hungarian matching between GT instances and detections of the
                     same class (maximising IoU); o2o_present = got its own detection,
                     o2o_pos = matched IoU >= 0.5, colour measured in the SAM mask of the
                     *matched detection box*; o2o_success = o2o_pos & colour ok
  * colour_ratios  : ratio of every bench colour inside the official (GT-box) mask
  * failure        : missing | mislocated | colour_wrong | success  (official rule order)
  * crosstalk      : colour_wrong and the dominant colour in the mask is the target colour of
                     another instance in the same prompt (an observable proxy, not a cause)

Implementation notes (documented deviations, no effect on the decision rules):
  * MIG_Bench's vendored GroundingDINO hard-codes a BERT path; we point it to a local copy.
  * The compiled MultiScaleDeformableAttention CUDA op is not built on this machine; we use the
    same file's pure-PyTorch `multi_scale_deformable_attn_pytorch` reference implementation.
  * Detections are computed once per (image, class) and SAM image embeddings once per image;
    the official script recomputes them per instance with identical inputs (same result).
"""
import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import config_hash, dump_json, env_info, git_commit, load_paths, read_jsonl  # noqa: E402

PATHS = load_paths()
BENCH = Path(PATHS["repos"]["mig_bench"])

# official colour table (eval_mig.py, active version)
COLOR_DICT = {
    'red': [{'Lower': np.array([0, 50, 70]), 'Upper': np.array([9, 255, 255])},
            {'Lower': np.array([159, 50, 70]), 'Upper': np.array([180, 255, 255])}],
    'blue': {'Lower': np.array([90, 50, 70]), 'Upper': np.array([128, 255, 255])},
    'yellow': {'Lower': np.array([25, 50, 70]), 'Upper': np.array([35, 255, 255])},
    'green': {'Lower': np.array([36, 50, 70]), 'Upper': np.array([89, 255, 255])},
    'black': {'Lower': np.array([0, 0, 0]), 'Upper': np.array([180, 255, 30])},
    'white': {'Lower': np.array([0, 0, 221]), 'Upper': np.array([180, 43, 255])},
    'brown': {'Lower': np.array([6, 43, 35]), 'Upper': np.array([25, 255, 255])},
}
EVAL_CFG = {"box_threshold": 0.25, "text_threshold": 0.25, "nms_threshold": 0.8, "iou_threshold": 0.5,
            "color_ratio_threshold": 0.2, "image_size": 512, "detector": "GroundingDINO SwinT-OGC",
            "segmenter": "SAM ViT-H", "version": 2}


# ------------------------------------------------------------------------------------ models
def load_groundingdino():
    sys.path.insert(0, str(BENCH / "GroundingDINO"))
    import groundingdino.models.GroundingDINO.ms_deform_attn as msda
    from groundingdino.util import get_tokenlizer

    class _PytorchMSDA:  # same reference implementation shipped in ms_deform_attn.py
        @staticmethod
        def apply(value, spatial_shapes, level_start_index, sampling_locations, attention_weights, im2col_step):
            return msda.multi_scale_deformable_attn_pytorch(value, spatial_shapes, sampling_locations, attention_weights)

    msda.MultiScaleDeformableAttnFunction = _PytorchMSDA
    from transformers import AutoTokenizer, BertModel
    get_tokenlizer.get_tokenlizer = lambda _t: AutoTokenizer.from_pretrained(PATHS["weights"]["bert"])
    get_tokenlizer.get_pretrained_language_model = lambda _t: BertModel.from_pretrained(PATHS["weights"]["bert"])
    from groundingdino.util.inference import Model
    cfg = str(BENCH / "GroundingDINO" / "groundingdino" / "config" / "GroundingDINO_SwinT_OGC.py")
    return Model(model_config_path=cfg, model_checkpoint_path=PATHS["weights"]["groundingdino"])


def load_sam():
    sys.path.insert(0, str(BENCH / "segment_anything"))
    from segment_anything import SamPredictor, sam_model_registry
    sam = sam_model_registry["vit_h"](checkpoint=PATHS["weights"]["sam_vit_h"]).to("cuda")
    return SamPredictor(sam)


# ------------------------------------------------------------------------------------ helpers
def iou_vec(pred, gt):
    """official IoU of one GT box against N predicted boxes (pixel xyxy)."""
    min_x = np.maximum(pred[:, 0], gt[0]); max_x = np.minimum(pred[:, 2], gt[2])
    min_y = np.maximum(pred[:, 1], gt[1]); max_y = np.minimum(pred[:, 3], gt[3])
    inter = np.maximum(max_x - min_x, 0.) * np.maximum(max_y - min_y, 0.)
    union = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
    return inter / union


def color_mask(image_bgr, color):
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    spec = COLOR_DICT[color]
    if isinstance(spec, list):
        m = np.zeros(hsv.shape[:2], bool)
        for s in spec:
            m |= cv2.inRange(hsv, s["Lower"], s["Upper"]) > 0
        return m
    return cv2.inRange(hsv, spec["Lower"], spec["Upper"]) > 0


def color_ratios(image_bgr, mask):
    """official: background replaced by gray 127, HSV inRange, fraction of mask pixels in range."""
    img = image_bgr * mask[..., None] + (1 - mask[..., None]) * 127
    img = img.astype(np.uint8)
    n = mask.sum()
    return {c: (float(np.logical_and(mask, color_mask(img, c)).sum() / n) if n > 0 else 0.0) for c in COLOR_DICT}


def sam_mask(predictor, box):
    masks, scores, _ = predictor.predict(box=np.asarray(box, dtype=np.float32), multimask_output=True)
    return masks[int(np.argmax(scores))].astype(np.uint8), float(np.max(scores))


def rle(mask):
    from pycocotools import mask as mask_utils
    r = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
    r["counts"] = r["counts"].decode("utf-8")
    return r


def hungarian(iou_mat):
    from scipy.optimize import linear_sum_assignment
    rows, cols = linear_sum_assignment(-iou_mat)
    return list(zip(rows.tolist(), cols.tolist()))


# ------------------------------------------------------------------------------------ main eval
def detect_all(model, sample, image_bgr):
    S = EVAL_CFG["image_size"]
    out = {}
    for cat in sorted({o["category"] for o in sample["objects"]}):
        det = model.predict_with_classes(image=image_bgr, classes=[cat], box_threshold=EVAL_CFG["box_threshold"],
                                         text_threshold=EVAL_CFG["text_threshold"])
        keep = torchvision.ops.nms(torch.from_numpy(det.xyxy), torch.from_numpy(det.confidence),
                                   EVAL_CFG["nms_threshold"]).numpy().tolist()
        out[cat] = {"xyxy": det.xyxy[keep].tolist(), "conf": det.confidence[keep].tolist()}
    return out


def eval_sample(sample, image_bgr, dets, predictor):
    S = EVAL_CFG["image_size"]
    objs = sample["objects"]
    predictor.set_image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
    inst = []
    # one-to-one assignment per category
    o2o = {}
    for cat in {o["category"] for o in objs}:
        gt_idx = [i for i, o in enumerate(objs) if o["category"] == cat]
        P = np.array(dets[cat]["xyxy"]).reshape(-1, 4)
        if len(P) == 0:
            continue
        M = np.stack([iou_vec(P, np.array(objs[i]["box"]) * S) for i in gt_idx])  # (n_gt, n_det)
        for r, c in hungarian(M):
            o2o[gt_idx[r]] = (c, float(M[r, c]))
    for i, o in enumerate(objs):
        rec = {"idx": i, "desc": o["desc"], "category": o["category"], "color": o["color"],
               "box": o["box"], "status": "ok"}
        try:
            gt = np.array(o["box"]) * S
            P = np.array(dets[o["category"]]["xyxy"]).reshape(-1, 4)
            rec["n_det"] = int(len(P))
            rec["present_any"] = bool(len(P) > 0)
            ov = float(iou_vec(P, gt).max()) if len(P) else 0.0
            rec["loc_iou"] = ov
            # ---- official
            if len(P) == 0:
                pos, attr, miou = 0, 0, 0.0
            elif ov < EVAL_CFG["iou_threshold"]:
                pos, attr, miou = 0, 0, ov
            else:
                pos = 1
                m, sc = sam_mask(predictor, gt)
                ratios = color_ratios(image_bgr, m)
                rec.update(mask_gt_box=rle(m), sam_score_gt_box=sc, mask_area=int(m.sum()), color_ratios=ratios,
                           empty_mask=bool(m.sum() == 0))
                attr = int(m.sum() > 0 and ratios[o["color"]] >= EVAL_CFG["color_ratio_threshold"])
                miou = ov if attr else 0.0
            rec.update(official_pos=pos, official_color=attr, official_success=int(pos and attr), official_miou=miou)
            # ---- failure category + crosstalk (based on official rule)
            if len(P) == 0:
                rec["failure"] = "missing"
            elif not pos:
                rec["failure"] = "mislocated"
            elif not attr:
                rec["failure"] = "colour_wrong"
                r = rec["color_ratios"]
                dom = max(r, key=r.get)
                others = {objs[j]["color"] for j in range(len(objs)) if j != i}
                rec["dominant_color"] = dom
                rec["crosstalk"] = bool(dom != o["color"] and dom in others and r[dom] >= EVAL_CFG["color_ratio_threshold"])
            else:
                rec["failure"] = "success"
            # ---- one-to-one diagnostic
            if i in o2o:
                c, v = o2o[i]
                rec.update(o2o_present=1, o2o_iou=v, o2o_det=P[c].tolist())
                if v >= EVAL_CFG["iou_threshold"]:
                    m2, _ = sam_mask(predictor, P[c])
                    r2 = color_ratios(image_bgr, m2)
                    ok = int(m2.sum() > 0 and r2[o["color"]] >= EVAL_CFG["color_ratio_threshold"])
                    rec.update(o2o_pos=1, o2o_color=ok, o2o_success=ok, o2o_color_ratio=r2[o["color"]])
                else:
                    rec.update(o2o_pos=0, o2o_color=0, o2o_success=0)
            else:
                rec.update(o2o_present=0, o2o_iou=0.0, o2o_pos=0, o2o_color=0, o2o_success=0)
        except Exception as e:  # never silently dropped: counted as failure and listed
            rec.update(status="error", error=repr(e), trace=traceback.format_exc(), official_pos=0,
                       official_color=0, official_success=0, official_miou=0.0, loc_iou=0.0, failure="eval_error",
                       o2o_present=0, o2o_iou=0.0, o2o_pos=0, o2o_color=0, o2o_success=0, present_any=False)
        inst.append(rec)
    return inst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen_dir", required=True, nargs="+")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    samples = {r["sample_id"]: r for r in read_jsonl(args.manifest)}
    chash = config_hash(EVAL_CFG)

    jobs = []
    for gd in args.gen_dir:
        gd = Path(gd)
        cache = gd / "eval_cache"
        cache.mkdir(exist_ok=True)
        dump_json(gd / "eval_config.json", {"eval_cfg": EVAL_CFG, "config_hash": chash, "env": env_info(),
                                            "mig_bench_commit": git_commit(BENCH)})
        for sid, s in samples.items():
            img = gd / "images" / f"{sid}.png"
            if not img.exists():
                continue
            out = cache / f"{sid}.json"
            if out.exists() and not args.force:
                c = json.load(open(out))
                if c.get("config_hash") == chash and c.get("image_sha1") == hashlib.sha1(img.read_bytes()).hexdigest():
                    continue
            jobs.append((gd, sid, img, out))
    print(f"{len(jobs)} images to evaluate")
    if not jobs:
        return

    # stage 1: detection (Grounding DINO only on GPU)
    t0 = time.time()
    gdino = load_groundingdino()
    det_cache = {}
    for gd, sid, img, out in jobs:
        bgr = cv2.imread(str(img))
        if bgr.shape[0] != 512:
            bgr = cv2.resize(bgr, (512, 512))
        try:
            det_cache[(gd, sid)] = detect_all(gdino, samples[sid], bgr)
        except Exception as e:
            det_cache[(gd, sid)] = {"__error__": repr(e)}
    del gdino
    torch.cuda.empty_cache()
    t_det = time.time() - t0

    # stage 2: segmentation + colour (SAM only on GPU)
    t0 = time.time()
    predictor = load_sam()
    for gd, sid, img, out in jobs:
        bgr = cv2.imread(str(img))
        if bgr.shape[0] != 512:
            bgr = cv2.resize(bgr, (512, 512))
        s = samples[sid]
        dets = det_cache[(gd, sid)]
        if "__error__" in dets:
            inst = [{"idx": i, "desc": o["desc"], "status": "error", "error": "detection: " + dets["__error__"],
                     "official_pos": 0, "official_color": 0, "official_success": 0, "official_miou": 0.0,
                     "loc_iou": 0.0, "failure": "eval_error", "present_any": False, "o2o_present": 0,
                     "o2o_iou": 0.0, "o2o_pos": 0, "o2o_color": 0, "o2o_success": 0}
                    for i, o in enumerate(s["objects"])]
        else:
            inst = eval_sample(s, bgr, dets, predictor)
        dump_json(out, {"sample_id": sid, "config_hash": chash, "image_sha1": hashlib.sha1(img.read_bytes()).hexdigest(),
                        "detections": dets, "instances": inst})
    t_seg = time.time() - t0
    print(f"detection {t_det:.0f}s, segmentation+colour {t_seg:.0f}s for {len(jobs)} images")


if __name__ == "__main__":
    main()
