"""Stage-A fidelity check for MIGC: official example vs. our adapter.

1) Runs the exact code path of MIGC/inference_single_image.py (fp32, official seed, the
   pipeline draws its own latent) -> outputs/checks/migc_official_example.png
2) Runs our MIGCModel adapter in fp32 on the same inputs, passing the latent the official path
   would have drawn (torch.randn on CUDA right after seed_everything) -> must be identical.
   (SAC's torch.randperm uses the CPU RNG, latents use the CUDA RNG, so passing latents does not
   shift any other random stream.)
The official README figure figures/MIGC_SD14_out.png is copied next to it for visual reference
(bit-equality with that figure is not expected: different GPU/library versions).
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from adapters.migc_adapter import MIGCModel, seed_everything  # noqa: E402
from common.utils import PROJECT_ROOT, dump_json, load_paths  # noqa: E402

PROMPT = [['masterpiece, best quality,black colored ball,gray colored cat,white colored  bed,\
                     green colored plant,red colored teddy bear,blue colored wall,brown colored vase,orange colored book,\
                     yellow colored hat', 'black colored ball', 'gray colored cat', 'white colored  bed', 'green colored plant', \
           'red colored teddy bear', 'blue colored wall', 'brown colored vase', 'orange colored book', 'yellow colored hat']]
BOXES = [[[0.3125, 0.609375, 0.625, 0.875], [0.5625, 0.171875, 0.984375, 0.6875],
          [0.0, 0.265625, 0.984375, 0.984375], [0.0, 0.015625, 0.21875, 0.328125],
          [0.171875, 0.109375, 0.546875, 0.515625], [0.234375, 0.0, 1.0, 0.3125],
          [0.71875, 0.625, 0.953125, 0.921875], [0.0625, 0.484375, 0.359375, 0.8125],
          [0.609375, 0.09375, 0.90625, 0.28125]]]
NEG = 'worst quality, low quality, bad anatomy, watermark, text, blurry'
SEED = 7351007268695528845


def main():
    out = PROJECT_ROOT / "outputs" / "checks"
    out.mkdir(parents=True, exist_ok=True)
    m = MIGCModel()
    pipe = m.pipe
    # 1) official path
    seed_everything(SEED)
    img_off = pipe([list(PROMPT[0])], BOXES, num_inference_steps=50, guidance_scale=7.5,
                   MIGCsteps=25, aug_phase_with_and=False, negative_prompt=NEG).images[0]
    img_off.save(out / "migc_official_example.png")
    # draw_box_desc draws on the PIL image in place -> annotate a copy
    pipe.draw_box_desc(img_off.copy(), BOXES[0], PROMPT[0][1:]).save(out / "migc_official_example_anno.png")
    # 2) adapter path with the latent the official path drew
    seed_everything(SEED)
    # diffusers' prepare_latents scales a given latent by init_noise_sigma exactly like a drawn one
    lat = torch.randn((1, 4, 64, 64), device="cuda", dtype=torch.float32).cpu()
    img_ad = m.generate(PROMPT[0][0], PROMPT[0][1:], BOXES[0], lat, seed=SEED, negative_prompt=NEG,
                        MIGCsteps=25, aug_phase_with_and=False, autocast=False)
    img_ad.save(out / "migc_adapter_example.png")
    d = np.abs(np.asarray(img_off, np.int16) - np.asarray(img_ad, np.int16))
    res = {"max_pixel_diff": int(d.max()), "mean_pixel_diff": float(d.mean())}
    shutil.copy(Path(load_paths()["repos"]["migc"]) / "figures" / "MIGC_SD14_out.png", out / "migc_readme_reference.png")
    dump_json(out / "migc_official_check.json", res)
    print(res)


if __name__ == "__main__":
    main()
