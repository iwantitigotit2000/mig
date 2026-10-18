"""Check that the SDPA attention swap reproduces the official einsum attention.

Runs one grounded UNet forward (fp32, no autocast) with both implementations on the same
inputs and reports the max/mean absolute difference of the predicted noise.
Writes outputs/checks/gligen_sdpa_check.json.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.gligen_adapter import GLIGEN, set_sdpa  # noqa: E402
from common.utils import PROJECT_ROOT, dump_json, initial_noise, read_jsonl  # noqa: E402


@torch.no_grad()
def main():
    g = GLIGEN(sdpa=False)
    r = read_jsonl(PROJECT_ROOT / "manifests" / "cocomig_pilot.jsonl")[0]
    batch = g.grounding_batch([o["desc"] for o in r["objects"]], [o["box"] for o in r["objects"]])
    context = g.text_encoder.encode([r["caption"]])
    x = initial_noise(0).cuda()
    res = {}
    for t in (999, 500, 50):
        inp = dict(x=x, timesteps=torch.tensor([t], device="cuda"), context=context,
                   grounding_input=g.grounding_tokenizer_input.prepare(batch),
                   inpainting_extra_input=None, grounding_extra_input=None)
        set_sdpa(False)
        a = g.model(inp)
        set_sdpa(True)
        b = g.model(inp)
        res[str(t)] = {"max_abs_diff": (a - b).abs().max().item(), "mean_abs_diff": (a - b).abs().mean().item(),
                       "ref_mean_abs": a.abs().mean().item()}
    print(res)
    dump_json(PROJECT_ROOT / "outputs" / "checks" / "gligen_sdpa_check.json", res)


if __name__ == "__main__":
    main()
