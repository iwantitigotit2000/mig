"""Record real tensor shapes inside MIGC with forward hooks (one full sampling call, first step).

Output: outputs/analysis/migc_probe.json
  which UNet cross-attention layers carry a MIGC module (others use the naive/vanilla path)
  per MIGC call at the first denoising step: ca_x (per-instance cross-attn outputs), guidance
  masks, image tokens, box tokens from its PositionNet, EA output, LA output, SAC weights.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.migc_adapter import MIGCModel  # noqa: E402
from common.utils import PROJECT_ROOT, dump_json, initial_noise, read_jsonl  # noqa: E402
from migc.migc_arch import MIGC  # noqa: E402


def shp(t):
    return list(t.shape) if torch.is_tensor(t) else None


def main():
    m = MIGCModel()
    r = read_jsonl(PROJECT_ROOT / "manifests" / "cocomig_pilot.jsonl")[2]
    rec = {"sample": r["sample_id"], "caption": r["caption"], "n_objects": r["n_objects"], "migc_layers": [],
           "calls": []}
    for name, proc in m.pipe.unet.attn_processors.items():
        if getattr(proc, "migc", None) is not None:
            rec["migc_layers"].append({"processor": name, "C": proc.migc.norm.normalized_shape[0],
                                       "n_params": sum(p.numel() for p in proc.migc.parameters())})
    rec["n_params_migc_total"] = sum(x["n_params"] for x in rec["migc_layers"])
    rec["n_params_unet"] = sum(p.numel() for p in m.pipe.unet.parameters())
    seen = {"n": 0}
    hooks = []
    for name, proc in m.pipe.unet.attn_processors.items():
        mod = getattr(proc, "migc", None)
        if mod is None:
            continue

        def make(name):
            def hook(module, args, kwargs, out):
                if seen["n"] >= len(rec["migc_layers"]):
                    return  # only the first denoising step
                seen["n"] += 1
                ca_x, gmask = args[0], args[1]
                info = kwargs["other_info"]
                o, fuser = out
                rec["calls"].append({
                    "processor": name, "ca_x(B,1+inst,HW,C)": shp(ca_x), "guidance_mask_in": shp(gmask),
                    "image_token": shp(info["image_token"]), "context_pooler(inst,1,768)": shp(info["context_pooler"]),
                    "box(B,inst,4)": shp(info["box"]), "supplement_mask": shp(info["supplement_mask"]),
                    "out": shp(o), "sac_scale(B,inst+2,H,W)": shp(fuser["sac_scale"]),
                    "ea_attn(B,inst,H,W,2)": shp(fuser["ea_attn"])})
            return hook
        hooks.append(mod.register_forward_hook(make(name), with_kwargs=True))
    # extra hooks on sub-modules of the first MIGC layer for EA / LA / pos_net shapes
    first = next(p.migc for p in m.pipe.unet.attn_processors.values() if getattr(p, "migc", None) is not None)
    sub = {}
    for nm in ("ea", "la", "pos_net", "sac"):
        def mk(nm):
            def h(module, args, kwargs, out):
                if nm not in sub:
                    a = list(args) + list(kwargs.values())
                    sub[nm] = {"inputs": [shp(x) for x in a if torch.is_tensor(x)],
                               "output": shp(out[0] if isinstance(out, tuple) else out)}
            return h
        hooks.append(getattr(first, nm).register_forward_hook(mk(nm), with_kwargs=True))
    m.generate(r["caption"], [o["desc"] for o in r["objects"]], [o["box"] for o in r["objects"]],
               initial_noise(r["seed"]), seed=r["seed"], steps=2, MIGCsteps=25)
    rec["first_layer_submodules"] = sub
    for h in hooks:
        h.remove()
    dump_json(PROJECT_ROOT / "outputs" / "analysis" / "migc_probe.json", rec)
    print(len(rec["migc_layers"]), rec["n_params_migc_total"], sub)


if __name__ == "__main__":
    main()
