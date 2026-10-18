"""Thin wrapper around the official GLIGEN code (gligen/GLIGEN @ f9dccb9).

Everything model-related comes from the official repo: model/VAE/text-encoder/diffusion
instantiation from the checkpoint's own config, `PLMSSampler`, `alpha_generator`,
`set_alpha_scale`, `get_clip_feature` and the GroundingNetInput tokenizer input.
Only orchestration (batching one layout, shared initial noise, timing) lives here.

Optional `sdpa=True` swaps the einsum attention of ldm.modules.attention.{Self,Cross}Attention
for torch.nn.functional.scaled_dot_product_attention (same weights, same maths). It is
verified against the original implementation in scripts/check_gligen_sdpa.py before use.
"""
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import load_paths  # noqa: E402

PATHS = load_paths()
GLIGEN_REPO = PATHS["repos"]["gligen"]
if GLIGEN_REPO not in sys.path:
    sys.path.insert(0, GLIGEN_REPO)

from ldm.util import instantiate_from_config  # noqa: E402
from ldm.models.diffusion.plms import PLMSSampler  # noqa: E402
import ldm.modules.attention as gl_attn  # noqa: E402

MAX_OBJS = 30  # official prepare_batch default


# --------------------------------------------------------------------------- attention swap
def _sdpa_self(self, x):
    B, N, _ = x.shape
    H = self.heads
    q, k, v = (t.view(B, N, H, -1).transpose(1, 2) for t in (self.to_q(x), self.to_k(x), self.to_v(x)))
    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    return self.to_out(out.transpose(1, 2).reshape(B, N, -1))


def _sdpa_cross(self, x, key, value, mask=None):
    assert mask is None
    B, N, _ = x.shape
    M = key.shape[1]
    H = self.heads
    q = self.to_q(x).view(B, N, H, -1).transpose(1, 2)
    k = self.to_k(key).view(B, M, H, -1).transpose(1, 2)
    v = self.to_v(value).view(B, M, H, -1).transpose(1, 2)
    out = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
    return self.to_out(out.transpose(1, 2).reshape(B, N, -1))


_ORIG = {"self": gl_attn.SelfAttention.forward, "cross": gl_attn.CrossAttention.forward}


def set_sdpa(enabled):
    gl_attn.SelfAttention.forward = _sdpa_self if enabled else _ORIG["self"]
    gl_attn.CrossAttention.forward = _sdpa_cross if enabled else _ORIG["cross"]


# --------------------------------------------------------------------------- loading
def load_official_ckpt(ckpt_path, device="cuda"):
    """Mirror of gligen_inference.load_ckpt (kept identical, plus mmap to save host RAM)."""
    try:
        saved = torch.load(ckpt_path, map_location="cpu", mmap=True)
    except Exception:
        saved = torch.load(ckpt_path, map_location="cpu")
    config = saved["config_dict"]["_content"]
    # FrozenCLIPEmbedder(version=...) is the official constructor argument; point it at the
    # local copy of openai/clip-vit-large-patch14 (transformers 4.19 cannot reach today's hub).
    config["text_encoder"]["params"] = {"version": PATHS["weights"]["clip_text"]}
    model = instantiate_from_config(config["model"]).to(device).eval()
    autoencoder = instantiate_from_config(config["autoencoder"]).to(device).eval()
    text_encoder = instantiate_from_config(config["text_encoder"]).to(device).eval()
    diffusion = instantiate_from_config(config["diffusion"]).to(device)
    model.load_state_dict(saved["model"])
    autoencoder.load_state_dict(saved["autoencoder"])
    text_encoder.load_state_dict(saved["text_encoder"])
    diffusion.load_state_dict(saved["diffusion"])
    extra_keys = sorted(set(saved.keys()) - {"model", "autoencoder", "text_encoder", "diffusion", "config_dict"})
    del saved
    return model, autoencoder, text_encoder, diffusion, config, extra_keys


def trainable_filter(name, mode):
    """Parameter selection for G1/G2; names follow the official trainer's rules."""
    is_pos = "position_net" in name
    is_fuser = ("transformer_blocks" in name) and ("fuser" in name)
    if mode == "g1":
        return is_pos
    if mode == "g2":
        return is_pos or is_fuser
    raise ValueError(mode)


class GLIGEN:
    def __init__(self, ckpt_path=None, device="cuda", sdpa=False, trainable_ckpt=None):
        import gligen_inference as gi  # official helpers (alpha_generator, set_alpha_scale, get_clip_feature)
        from transformers import CLIPModel, CLIPProcessor
        self.gi = gi
        self.device = device
        set_sdpa(sdpa)
        self.sdpa = sdpa
        self.ckpt_path = ckpt_path or PATHS["weights"]["gligen_box_text"]
        (self.model, self.autoencoder, self.text_encoder, self.diffusion,
         self.config, self.extra_ckpt_keys) = load_official_ckpt(self.ckpt_path, device)
        self.grounding_tokenizer_input = instantiate_from_config(self.config["grounding_tokenizer_input"])
        self.model.grounding_tokenizer_input = self.grounding_tokenizer_input
        self._guard_first_conv_restore()
        self.trainable_ckpt = None
        if trainable_ckpt:
            self.load_trainable(trainable_ckpt)
        # phrase encoder exactly as official prepare_batch: CLIPModel text tower, pooler ("before" projection)
        version = PATHS["weights"]["clip_text"]
        self.clip_model = CLIPModel.from_pretrained(version).to(device).eval()
        self.clip_processor = CLIPProcessor.from_pretrained(version)
        self._phrase_cache = {}

    def _guard_first_conv_restore(self):
        """PLMSSampler calls model.restore_first_conv_from_SD() at every alpha==0 step. It loads
        GLIGEN/SD_input_conv_weight_bias.pth (cwd-relative) and swaps in a new Conv2d. For the
        box-text model this conv is bit-identical to SD's, so the call is a value no-op; we assert
        that once and replace it by a real no-op (keeps module identity / frozen flags intact)."""
        sd_conv = torch.load(Path(GLIGEN_REPO) / "SD_input_conv_weight_bias.pth", map_location="cpu")
        conv = self.model.input_blocks[0][0]
        assert torch.equal(conv.weight.detach().cpu(), sd_conv["weight"]) and \
            torch.equal(conv.bias.detach().cpu(), sd_conv["bias"]), "first conv differs from SD; keep official restore"
        self.model.restore_first_conv_from_SD = lambda: None

    def load_trainable(self, path):
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("trainable_state", sd)
        own = dict(self.model.named_parameters())
        missing = [k for k in sd if k not in own]
        assert not missing, missing[:5]
        with torch.no_grad():
            for k, v in sd.items():
                own[k].copy_(v.to(own[k].device, own[k].dtype))
        self.trainable_ckpt = str(path)
        return len(sd)

    def release_clip(self):
        """Free the phrase encoder after all needed phrases are cached (training saves ~1.7 GB)."""
        self.clip_model = None
        torch.cuda.empty_cache()

    @torch.no_grad()
    def phrase_embedding(self, phrase):
        if phrase not in self._phrase_cache:
            assert self.clip_model is not None, f"phrase '{phrase}' not cached and CLIP released"
            self._phrase_cache[phrase] = self.gi.get_clip_feature(
                self.clip_model, self.clip_processor, phrase, is_image=False).float().cpu()
        return self._phrase_cache[phrase]

    @torch.no_grad()
    def grounding_batch(self, phrases, boxes, batch=1):
        """Same tensors as official gligen_inference.prepare_batch (text-only grounding)."""
        B = torch.zeros(MAX_OBJS, 4)
        M = torch.zeros(MAX_OBJS)
        E = torch.zeros(MAX_OBJS, 768)
        for i, (p, b) in enumerate(zip(phrases, boxes)):
            B[i] = torch.tensor(b)
            M[i] = 1
            E[i] = self.phrase_embedding(p)
        out = {"boxes": B[None].repeat(batch, 1, 1), "masks": M[None].repeat(batch, 1),
               "text_embeddings": E[None].repeat(batch, 1, 1)}
        return {k: v.to(self.device) for k, v in out.items()}

    @torch.no_grad()
    def generate(self, prompt, phrases, boxes, noise, alpha_type=(0.3, 0.0, 0.7), steps=50,
                 guidance_scale=7.5, negative_prompt="", autocast=True):
        assert len(phrases) == len(boxes) <= MAX_OBJS
        batch = self.grounding_batch(phrases, boxes)
        ctx = torch.autocast("cuda", dtype=torch.float16) if autocast else torch.autocast("cuda", enabled=False)
        with ctx:
            context = self.text_encoder.encode([prompt])
            uc = self.text_encoder.encode([negative_prompt])
            alpha_fn = lambda n: self.gi.alpha_generator(n, type=list(alpha_type))  # noqa: E731
            sampler = PLMSSampler(self.diffusion, self.model, alpha_generator_func=alpha_fn,
                                  set_alpha_scale=self.gi.set_alpha_scale)
            inp = dict(x=noise.to(self.device), timesteps=None, context=context,
                       grounding_input=self.grounding_tokenizer_input.prepare(batch),
                       inpainting_extra_input=None, grounding_extra_input=None)
            shape = (1, self.model.in_channels, self.model.image_size, self.model.image_size)
            z = sampler.sample(S=steps, shape=shape, input=inp, uc=uc, guidance_scale=guidance_scale)
            x = self.autoencoder.decode(z.float() if not autocast else z)
        # restore full grounding strength for the next call (sampler leaves the last alpha set)
        self.gi.set_alpha_scale(self.model, 1)
        x = torch.clamp(x.float(), -1, 1) * 0.5 + 0.5
        arr = (x[0].cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        return Image.fromarray(arr)
