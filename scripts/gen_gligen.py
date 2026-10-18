"""Generate COCO-MIG samples with GLIGEN (G0 = released checkpoint; G1/G2 = + fine-tuned params).

Example:
  python scripts/gen_gligen.py --manifest manifests/cocomig_pilot.jsonl --out outputs/gen/pilot/G0
  python scripts/gen_gligen.py --manifest ... --out ... --trainable D:/mig_assets/runs/g2/ckpt_002000.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.gen_runner import run_generation  # noqa: E402
from common.utils import PROJECT_ROOT, ckpt_fingerprint, git_commit, load_paths, load_yaml  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--trainable", default=None, help="fine-tuned position_net/fuser state (G1/G2)")
    ap.add_argument("--alpha_type", type=float, nargs=3, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no_sdpa", action="store_true")
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    from adapters.gligen_adapter import GLIGEN
    paths = load_paths()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "gen.yaml")
    common, g = cfg["common"], cfg["gligen"]
    alpha_type = args.alpha_type or g["alpha_type"]
    sdpa = g["sdpa"] and not args.no_sdpa
    autocast = common["autocast_fp16"] and not args.fp32
    ckpt = paths["weights"][g["checkpoint"]]
    model = GLIGEN(ckpt, sdpa=sdpa, trainable_ckpt=args.trainable)

    run_config = {
        "method": "GLIGEN", "checkpoint": ckpt_fingerprint(ckpt), "checkpoint_extra_keys": model.extra_ckpt_keys,
        "trainable_ckpt": args.trainable, "repo_commit": git_commit(paths["repos"]["gligen"]),
        "sampler": "PLMS", "steps": common["steps"], "guidance_scale": common["guidance_scale"],
        "negative_prompt": common["negative_prompt"], "alpha_type": list(alpha_type),
        "resolution": [common["height"], common["width"]], "sdpa": sdpa, "autocast_fp16": autocast,
        "phrase_source": "object desc, e.g. 'a red carrot'", "max_objs": 30,
    }

    def gen(r, noise):
        return model.generate(r["caption"], [o["desc"] for o in r["objects"]], [o["box"] for o in r["objects"]],
                              noise, alpha_type=alpha_type, steps=common["steps"],
                              guidance_scale=common["guidance_scale"], negative_prompt=common["negative_prompt"],
                              autocast=autocast)

    run_generation(args.manifest, args.out, "GLIGEN", run_config, gen, limit=args.limit)


if __name__ == "__main__":
    main()
