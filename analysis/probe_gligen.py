"""Record real tensor shapes / gate values inside GLIGEN with forward hooks (one grounded UNet call).

Output: outputs/analysis/gligen_probe.json
  position_net: input boxes/masks/embeddings and output grounding tokens
  every GatedSelfAttentionDense (fuser): visual tokens x, projected objs, concat length,
      tanh(alpha_attn), tanh(alpha_dense) (learned gates of the released checkpoint)
  every text cross-attention (attn2): query and context shapes
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.gligen_adapter import GLIGEN  # noqa: E402
import ldm.modules.attention as A  # noqa: E402
from common.utils import PROJECT_ROOT, dump_json, initial_noise, read_jsonl  # noqa: E402


@torch.no_grad()
def main():
    g = GLIGEN(sdpa=True)
    r = read_jsonl(PROJECT_ROOT / "manifests" / "cocomig_pilot.jsonl")[2]
    batch = g.grounding_batch([o["desc"] for o in r["objects"]], [o["box"] for o in r["objects"]])
    rec = {"sample": r["sample_id"], "caption": r["caption"], "n_objects": r["n_objects"], "fusers": [], "cross_attn": []}
    names = {m: n for n, m in g.model.named_modules()}

    def pos_hook(m, args, kwargs, out):
        rec["position_net"] = {"boxes": list(kwargs["boxes"].shape), "masks": list(kwargs["masks"].shape),
                               "positive_embeddings": list(kwargs["positive_embeddings"].shape),
                               "fourier_xyxy_dim": m.position_dim, "out_tokens": list(out.shape),
                               "mlp": str(m.linears)}

    def fuser_hook(m, args, out):
        x, objs = args
        rec["fusers"].append({"name": names[m], "x_visual_tokens": list(x.shape), "objs_in": list(objs.shape),
                              "objs_projected": [objs.shape[0], objs.shape[1], m.linear.out_features],
                              "concat_len": x.shape[1] + objs.shape[1], "heads": m.attn.heads,
                              "tanh_alpha_attn": float(torch.tanh(m.alpha_attn)),
                              "tanh_alpha_dense": float(torch.tanh(m.alpha_dense)),
                              "n_params": sum(p.numel() for p in m.parameters())})

    def ca_hook(m, args, out):
        x, key = args[0], args[1]
        rec["cross_attn"].append({"name": names[m], "query": list(x.shape), "context": list(key.shape)})

    hs = [g.model.position_net.register_forward_hook(pos_hook, with_kwargs=True)]
    for n, m in g.model.named_modules():
        if isinstance(m, A.GatedSelfAttentionDense):
            hs.append(m.register_forward_hook(fuser_hook))
        if n.endswith("attn2"):
            hs.append(m.register_forward_hook(ca_hook))
    inp = dict(x=initial_noise(0).cuda(), timesteps=torch.tensor([500], device="cuda"),
               context=g.text_encoder.encode([r["caption"]]),
               grounding_input=g.grounding_tokenizer_input.prepare(batch),
               inpainting_extra_input=None, grounding_extra_input=None)
    eps = g.model(inp)
    rec["unet_out"] = list(eps.shape)
    rec["n_params"] = {"unet_total": sum(p.numel() for p in g.model.parameters()),
                       "position_net": sum(p.numel() for p in g.model.position_net.parameters()),
                       "fusers_total": sum(f["n_params"] for f in rec["fusers"])}
    for h in hs:
        h.remove()
    dump_json(PROJECT_ROOT / "outputs" / "analysis" / "gligen_probe.json", rec)
    print(rec["position_net"], rec["n_params"], len(rec["fusers"]))


if __name__ == "__main__":
    main()
