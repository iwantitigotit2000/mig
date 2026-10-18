"""Generate COCO-MIG samples with MIGC (public MIGC_SD14.ckpt, inference only).

Example:
  python scripts/gen_migc.py --manifest manifests/cocomig_pilot.jsonl --out outputs/gen/pilot/MIGC
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
    ap.add_argument("--migc_steps", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--fp32", action="store_true")
    args = ap.parse_args()

    from adapters.migc_adapter import MIGCModel
    paths = load_paths()
    cfg = load_yaml(PROJECT_ROOT / "configs" / "gen.yaml")
    common, m = cfg["common"], cfg["migc"]
    migc_steps = args.migc_steps if args.migc_steps is not None else m["MIGCsteps"]
    autocast = common["autocast_fp16"] and not args.fp32
    ckpt = paths["weights"][m["checkpoint"]]
    model = MIGCModel(ckpt, paths["weights"]["sd14_diffusers"])

    run_config = {
        "method": "MIGC", "checkpoint": ckpt_fingerprint(ckpt), "base_model": paths["weights"]["sd14_diffusers"],
        "repo_commit": git_commit(paths["repos"]["migc"]), "scheduler": "EulerDiscreteScheduler",
        "steps": common["steps"], "guidance_scale": common["guidance_scale"],
        "negative_prompt": common["negative_prompt"], "MIGCsteps": migc_steps,
        "NaiveFuserSteps": m["NaiveFuserSteps"], "aug_phase_with_and": m["aug_phase_with_and"],
        "resolution": [common["height"], common["width"]], "autocast_fp16": autocast,
    }

    def gen(r, noise):
        return model.generate(r["caption"], [o["desc"] for o in r["objects"]], [o["box"] for o in r["objects"]],
                              noise, seed=r["seed"], steps=common["steps"], guidance_scale=common["guidance_scale"],
                              negative_prompt=common["negative_prompt"], MIGCsteps=migc_steps,
                              NaiveFuserSteps=m["NaiveFuserSteps"], aug_phase_with_and=m["aug_phase_with_and"],
                              autocast=autocast)

    run_generation(args.manifest, args.out, "MIGC", run_config, gen, limit=args.limit)


if __name__ == "__main__":
    main()
