"""Thin wrapper around the official MIGC pipeline (limuloo/MIGC @ effd6cc), public MIGC_SD14.ckpt.

Construction follows inference_mig_benchmark.py: StableDiffusionMIGCPipeline.from_pretrained(SD1.4),
AttentionStore, load_migc(..., MIGCProcessor), EulerDiscreteScheduler. Differences from that
script (recorded in run configs): the starting latent is passed explicitly (shared with GLIGEN),
the safety checker is not loaded (the MIGC pipeline never calls it), and sampling runs under
fp16 autocast when enabled.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import load_paths  # noqa: E402

PATHS = load_paths()
if PATHS["repos"]["migc"] not in sys.path:
    sys.path.insert(0, PATHS["repos"]["migc"])

from diffusers import EulerDiscreteScheduler  # noqa: E402
from migc.migc_pipeline import AttentionStore, MIGCProcessor, StableDiffusionMIGCPipeline  # noqa: E402
from migc.migc_utils import load_migc, seed_everything  # noqa: E402


class MIGCModel:
    def __init__(self, ckpt_path=None, sd_path=None, device="cuda"):
        self.ckpt_path = ckpt_path or PATHS["weights"]["migc_sd14"]
        self.sd_path = sd_path or PATHS["weights"]["sd14_diffusers"]
        pipe = StableDiffusionMIGCPipeline.from_pretrained(self.sd_path, safety_checker=None,
                                                           requires_safety_checker=False)
        pipe.attention_store = AttentionStore()
        load_migc(pipe.unet, pipe.attention_store, self.ckpt_path, attn_processor=MIGCProcessor)
        pipe = pipe.to(device)
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
        pipe.set_progress_bar_config(disable=True)
        self.pipe = pipe
        self.device = device

    @torch.no_grad()
    def generate(self, prompt, phrases, boxes, noise, seed, steps=50, guidance_scale=7.5, negative_prompt=None,
                 MIGCsteps=25, NaiveFuserSteps=-1, aug_phase_with_and=True, autocast=True):
        prompt_final = [[prompt] + list(phrases)]
        bboxes = [[list(b) for b in boxes]]
        seed_everything(seed)  # SAC uses torch.randperm on the global RNG
        ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else torch.autocast("cuda", enabled=False)
        with ctx:
            out = self.pipe(prompt_final, bboxes, num_inference_steps=steps, guidance_scale=guidance_scale,
                            negative_prompt=negative_prompt, latents=noise.to(self.device),
                            MIGCsteps=MIGCsteps, NaiveFuserSteps=NaiveFuserSteps,
                            aug_phase_with_and=aug_phase_with_and)
        return out.images[0]
