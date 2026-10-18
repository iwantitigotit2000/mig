"""Stage-A fidelity check for GLIGEN: official gligen_inference.run() vs. our adapter.

The official function is executed unmodified from a scratch working directory that contains
  openai/clip-vit-large-patch14 -> junction to the local CLIP copy (the official code calls
                                  from_pretrained("openai/clip-vit-large-patch14"))
  SD_input_conv_weight_bias.pth  (cwd-relative file read by the official sampler)
Example = the official 'generation_box_text' meta in gligen_inference.py (teddy bear + bird),
batch 1, explicit starting noise. The adapter is run with the same inputs in fp32 with the
original einsum attention, and the images are compared pixel-wise.
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import PROJECT_ROOT, dump_json, initial_noise, load_paths  # noqa: E402

PATHS = load_paths()
OFFICIAL_NEG = ('longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, '
                'cropped, worst quality, low quality')
META = dict(prompt="a teddy bear sitting next to a bird", phrases=['a teddy bear', 'a bird'],
            locations=[[0.0, 0.09, 0.33, 0.76], [0.55, 0.11, 1.0, 0.8]], alpha_type=[0.3, 0.0, 0.7],
            save_folder_name="generation_box_text")


def main():
    out = PROJECT_ROOT / "outputs" / "checks"
    work = Path(PATHS["assets_root"]) / "scratch_gligen_official"
    (work / "openai").mkdir(parents=True, exist_ok=True)
    link = work / "openai" / "clip-vit-large-patch14"
    if not link.exists():
        subprocess.check_call(["cmd", "/c", "mklink", "/J", str(link), str(Path(PATHS["weights"]["clip_text"]))])
    shutil.copy(Path(PATHS["repos"]["gligen"]) / "SD_input_conv_weight_bias.pth", work)
    os.chdir(work)
    sys.path.insert(0, PATHS["repos"]["gligen"])
    import gligen_inference as gi

    noise = initial_noise(1234).cuda()
    gi.args = argparse.Namespace(folder=str(work / "official_out"), batch_size=1, no_plms=False,
                                 guidance_scale=7.5, negative_prompt=OFFICIAL_NEG)
    shutil.rmtree(work / "official_out", ignore_errors=True)
    meta = dict(META, ckpt=PATHS["weights"]["gligen_box_text"])
    gi.run(meta, gi.args, starting_noise=noise.clone())
    img_off = Image.open(work / "official_out" / META["save_folder_name"] / "0.png")
    img_off.save(out / "gligen_official_example.png")
    torch.cuda.empty_cache()

    os.chdir(PROJECT_ROOT)
    from adapters.gligen_adapter import GLIGEN
    g = GLIGEN(sdpa=False)
    img_ad = g.generate(META["prompt"], META["phrases"], META["locations"], noise.clone(),
                        alpha_type=META["alpha_type"], negative_prompt=OFFICIAL_NEG, autocast=False)
    img_ad.save(out / "gligen_adapter_example.png")
    d = np.abs(np.asarray(img_off, np.int16) - np.asarray(img_ad, np.int16))
    res = {"max_pixel_diff": int(d.max()), "mean_pixel_diff": float(d.mean())}
    dump_json(out / "gligen_official_check.json", res)
    print(res)


if __name__ == "__main__":
    main()
