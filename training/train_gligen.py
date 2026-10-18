"""Local fine-tuning of the released GLIGEN box-text model (architecture unchanged).

  G1: update position_net only
  G2: update position_net + every transformer_blocks.*.fuser (gated self-attention)

Adapted from the official trainer.py (gligen/GLIGEN @ f9dccb9):
  * same objective: eps-prediction MSE, t ~ U{0..999}, q_sample from the official LatentDiffusion
  * same 10% grounding-token drop (inside the official UNetModel.forward when model.training)
  * same caption dropout (prob_use_caption) and box processing (see coco_grounding_dataset.py)
  * same optimiser family: AdamW(weight_decay=0) + constant schedule with warm-up
Differences: initial weights = released GLIGEN checkpoint (no re-initialisation of any module),
only the selected subset is trainable, optional bf16 autocast, gradient accumulation, and the
checkpoint stores only trainable tensors + optimiser/scheduler/RNG state for exact resumption.

Usage:
  python training/train_gligen.py --config configs/train/g2_lr5e-5.yaml [--max_steps 100] [--max_hours 5]
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common.utils import (PROJECT_ROOT, append_jsonl, ckpt_fingerprint, dump_json, env_info, git_commit,  # noqa: E402
                          initial_noise, load_paths, load_yaml, read_jsonl)
from adapters.gligen_adapter import GLIGEN, trainable_filter  # noqa: E402
from training.coco_grounding_dataset import CocoGroundingDataset, ResumableSampler  # noqa: E402

PATHS = load_paths()


def param_fingerprint(params):
    """Exact float64 checksums per tensor (sum, abs-sum) -- equal iff unchanged with overwhelming probability."""
    out = {}
    with torch.no_grad():
        for n, p in params:
            d = p.detach().double()
            out[n] = (d.sum().item(), d.abs().sum().item())
    return out


def collate(items):
    out = {}
    for k in items[0]:
        v = [it[k] for it in items]
        out[k] = torch.stack(v) if torch.is_tensor(v[0]) else v
    return out


class Trainer:
    def __init__(self, cfg, args):
        self.cfg = cfg
        self.args = args
        self.device = "cuda"
        self.run_dir = Path(PATHS["runs_root"]) / cfg["name"]
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "samples").mkdir(exist_ok=True)
        self.log_path = self.run_dir / "train_log.jsonl"
        self.val_log_path = self.run_dir / "val_log.jsonl"

        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])
        np.random.seed(cfg["seed"])

        # ---------------- model: released GLIGEN checkpoint, nothing re-initialised
        self.g = GLIGEN(PATHS["weights"]["gligen_box_text"], sdpa=cfg.get("sdpa", True))
        self.model, self.ae, self.te, self.diffusion = self.g.model, self.g.autoencoder, self.g.text_encoder, self.g.diffusion
        for m in (self.model, self.ae, self.te):
            for p in m.parameters():
                p.requires_grad_(False)
        self.trainable = [(n, p) for n, p in self.model.named_parameters() if trainable_filter(n, cfg["mode"])]
        for _, p in self.trainable:
            p.requires_grad_(True)
        self.frozen = [(n, p) for n, p in self.model.named_parameters() if not p.requires_grad]
        n_tr = sum(p.numel() for _, p in self.trainable)
        n_all = sum(p.numel() for _, p in self.model.named_parameters())
        groups = {}
        for n, p in self.trainable:
            key = "position_net" if "position_net" in n else "fuser"
            groups[key] = groups.get(key, 0) + p.numel()
        dump_json(self.run_dir / "trainable_params.json", {
            "mode": cfg["mode"], "n_trainable": n_tr, "n_unet_total": n_all, "fraction": n_tr / n_all,
            "by_group": groups, "n_tensors": len(self.trainable), "names": [n for n, _ in self.trainable]})
        print(f"trainable params: {n_tr:,} / {n_all:,} ({100 * n_tr / n_all:.2f}%) groups={groups}")

        # ---------------- optimiser / scheduler (official choices)
        from transformers import get_constant_schedule_with_warmup
        self.opt = torch.optim.AdamW([p for _, p in self.trainable], lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        self.sched = get_constant_schedule_with_warmup(self.opt, num_warmup_steps=cfg["warmup_steps"])

        # ---------------- data
        self.phrase_emb = self.load_phrase_embeddings()
        for r in read_jsonl(PROJECT_ROOT / cfg["val"]["sample_manifest"])[: cfg["val"]["n_samples"]]:
            for o in r["objects"]:
                self.g.phrase_embedding(o["desc"])
        self.g.release_clip()
        dcfg = cfg["data"]
        self.ds = CocoGroundingDataset(PROJECT_ROOT / dcfg["train_manifest"], PATHS["coco"]["train_images"],
                                       self.phrase_emb, prob_use_caption=dcfg["prob_use_caption"],
                                       max_boxes_per_data=dcfg["max_boxes_per_data"], seed=cfg["seed"])
        self.val_ds = CocoGroundingDataset(PROJECT_ROOT / dcfg["val_manifest"], PATHS["coco"]["train_images"],
                                           self.phrase_emb, prob_use_caption=1.0, random_flip=False,
                                           max_boxes_per_data=dcfg["max_boxes_per_data"], seed=12345)

        self.step = 0
        self.init_trainable = {n: p.detach().cpu().clone() for n, p in self.trainable}
        self.resume()
        if self.step == 0:
            self.frozen_fp0 = param_fingerprint(self.frozen)
            torch.save({"frozen_fp": self.frozen_fp0, "trainable_init": self.init_trainable}, self.run_dir / "init_state.pt")
        else:
            st = torch.load(self.run_dir / "init_state.pt")
            self.frozen_fp0, self.init_trainable = st["frozen_fp"], st["trainable_init"]

        sampler = ResumableSampler(len(self.ds), cfg["seed"], start=self.step * cfg["batch_size"] * cfg["grad_accum"])
        self.loader = iter(torch.utils.data.DataLoader(self.ds, batch_size=cfg["batch_size"], sampler=sampler,
                                                       num_workers=cfg.get("workers", 2), collate_fn=collate,
                                                       pin_memory=True, persistent_workers=cfg.get("workers", 2) > 0))
        if self.step == 0:
            dump_json(self.run_dir / "run_info.json", {
                "config": cfg, "env": env_info(), "gligen_commit": git_commit(PATHS["repos"]["gligen"]),
                "init_checkpoint": ckpt_fingerprint(PATHS["weights"]["gligen_box_text"]),
                "n_train_images": len(self.ds), "n_val_images": len(self.val_ds)})

    # ------------------------------------------------------------------ helpers
    def load_phrase_embeddings(self):
        cache = Path(PATHS["coco"]["root"]) / "phrase_embeddings_clipL14_before.pt"
        if cache.exists():
            return torch.load(cache)
        names = sorted({o["phrase"] for f in ("coco_train.jsonl", "coco_trainval.jsonl")
                        for r in read_jsonl(PROJECT_ROOT / "manifests" / f) for o in r["objects"]})
        emb = {n: self.g.phrase_embedding(n) for n in names}
        torch.save(emb, cache)
        return emb

    def autocast(self):
        prec = self.cfg.get("precision", "bf16")
        if prec == "fp32":
            return torch.autocast("cuda", enabled=False)
        return torch.autocast("cuda", dtype=torch.bfloat16 if prec == "bf16" else torch.float16)

    def loss_on(self, batch, t=None, noise=None, gen=None):
        """Official run_one_step: z = E(x), context = CLIP(caption), x_t = q_sample, MSE(eps_hat, eps)."""
        batch = {k: (v.to(self.device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad(), self.autocast():
            z = self.ae.encode(batch["image"]).float()
            context = self.te.encode(batch["caption"])
        if t is None:
            _t = torch.rand(z.shape[0], device=z.device, generator=gen)
            t = (_t * 1000).long()
            t = torch.where(t != 1000, t, 999)
        if noise is None:
            noise = torch.randn(z.shape, device=z.device, generator=gen)
        x_noisy = self.diffusion.q_sample(x_start=z, t=t, noise=noise)
        grounding_input = self.g.grounding_tokenizer_input.prepare(batch)
        inp = dict(x=x_noisy, timesteps=t, context=context, inpainting_extra_input=None,
                   grounding_extra_input=None, grounding_input=grounding_input)
        with self.autocast():
            out = self.model(inp)
        return torch.nn.functional.mse_loss(out.float(), noise)

    # ------------------------------------------------------------------ checkpointing
    def save(self, tag=None):
        state = {
            "trainable_state": {n: p.detach().cpu() for n, p in self.trainable},
            "opt": self.opt.state_dict(), "sched": self.sched.state_dict(), "step": self.step,
            "rng": {"py": random.getstate(), "np": np.random.get_state(), "torch": torch.get_rng_state(),
                    "cuda": torch.cuda.get_rng_state(), "gen": self.gen.get_state()},
            "config": self.cfg,
        }
        tmp = self.run_dir / "ckpt_latest.pt.tmp"
        torch.save(state, tmp)
        os.replace(tmp, self.run_dir / "ckpt_latest.pt")
        if tag:
            torch.save({"trainable_state": state["trainable_state"], "step": self.step, "config": self.cfg},
                       self.run_dir / f"ckpt_{self.step:06d}.pt")

    def resume(self):
        self.gen = torch.Generator(device=self.device).manual_seed(self.cfg["seed"] + 1)
        p = self.run_dir / "ckpt_latest.pt"
        if not p.exists():
            return
        st = torch.load(p, map_location="cpu")
        own = dict(self.trainable)
        with torch.no_grad():
            for n, v in st["trainable_state"].items():
                own[n].copy_(v.to(own[n].device))
        self.opt.load_state_dict(st["opt"])
        self.sched.load_state_dict(st["sched"])
        self.step = st["step"]
        random.setstate(st["rng"]["py"])
        np.random.set_state(st["rng"]["np"])
        torch.set_rng_state(st["rng"]["torch"])
        torch.cuda.set_rng_state(st["rng"]["cuda"])
        self.gen.set_state(st["rng"]["gen"])
        print(f"resumed from {p} at step {self.step}")

    # ------------------------------------------------------------------ checks
    def gradient_check(self):
        no_grad = [n for n, p in self.trainable if p.grad is None]
        zero_grad = [n for n, p in self.trainable if p.grad is not None and p.grad.abs().sum().item() == 0]
        frozen_with_grad = [n for n, p in self.frozen if p.grad is not None]
        res = {"step": self.step, "trainable_without_grad": no_grad, "trainable_zero_grad": zero_grad,
               "frozen_with_grad": frozen_with_grad,
               "grad_norm_by_group": self.grad_norms()}
        dump_json(self.run_dir / "gradient_check.json", res)
        print(f"gradient check: {len(no_grad)} trainable w/o grad, {len(zero_grad)} zero-grad, "
              f"{len(frozen_with_grad)} frozen with grad")
        return res

    def grad_norms(self):
        g = {}
        for n, p in self.trainable:
            if p.grad is None:
                continue
            key = "position_net" if "position_net" in n else ("fuser_gates" if "alpha_" in n else "fuser")
            g[key] = g.get(key, 0.0) + p.grad.float().pow(2).sum().item()
        return {k: math.sqrt(v) for k, v in g.items()}

    def weight_check(self):
        fp = param_fingerprint(self.frozen)
        changed_frozen = [n for n in fp if fp[n] != self.frozen_fp0[n]]
        delta = {}
        unchanged_trainable = []
        for n, p in self.trainable:
            d = (p.detach().cpu().float() - self.init_trainable[n].float()).norm().item()
            key = "position_net" if "position_net" in n else ("fuser_gates" if "alpha_" in n else "fuser")
            delta[key] = delta.get(key, 0.0) + d * d
            if d == 0:
                unchanged_trainable.append(n)
        gates = {n: math.tanh(p.item()) for n, p in self.trainable if n.endswith("alpha_attn")}
        res = {"step": self.step, "n_frozen_checked": len(fp), "frozen_changed": changed_frozen,
               "trainable_unchanged": unchanged_trainable,
               "trainable_delta_l2_by_group": {k: math.sqrt(v) for k, v in delta.items()},
               "tanh_alpha_attn": gates}
        dump_json(self.run_dir / f"weight_check_{self.step:06d}.json", res)
        print(f"weight check @ {self.step}: frozen changed={len(changed_frozen)}, "
              f"trainable unchanged={len(unchanged_trainable)}, delta={res['trainable_delta_l2_by_group']}")
        return res

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self):
        self.model.eval()  # eval: no random grounding drop
        n = min(self.cfg["val"]["n_images"], len(self.val_ds))
        ts = self.cfg["val"]["timesteps"]
        losses = {t: [] for t in ts}
        for i in range(n):
            b = collate([self.val_ds[(0, i)]])
            for t in ts:
                noise = initial_noise(10_000 + i).to(self.device)  # fixed per image
                tt = torch.full((1,), t, device=self.device, dtype=torch.long)
                losses[t].append(self.loss_on(b, t=tt, noise=noise).item())
        res = {"step": self.step, "val_loss": float(np.mean([v for l in losses.values() for v in l])),
               "val_loss_by_t": {str(t): float(np.mean(l)) for t, l in losses.items()}, "n_images": n}
        append_jsonl(self.val_log_path, res)
        print(f"val @ {self.step}: {res['val_loss']:.5f} {res['val_loss_by_t']}")
        self.model.train()
        return res

    @torch.no_grad()
    def sample_images(self):
        self.model.eval()
        rows = read_jsonl(PROJECT_ROOT / self.cfg["val"]["sample_manifest"])[: self.cfg["val"]["n_samples"]]
        tiles = []
        for r in rows:
            img = self.g.generate(r["caption"], [o["desc"] for o in r["objects"]], [o["box"] for o in r["objects"]],
                                  initial_noise(r["seed"]), alpha_type=(0.3, 0.0, 0.7), steps=50,
                                  negative_prompt="worst quality, low quality, bad anatomy")
            tiles.append(np.array(img))
        grid = np.concatenate(tiles, axis=1)
        from PIL import Image
        Image.fromarray(grid).save(self.run_dir / "samples" / f"step_{self.step:06d}.png")
        self.model.train()

    # ------------------------------------------------------------------ loop
    def train(self):
        cfg = self.cfg
        total = self.args.max_steps or cfg["total_steps"]
        deadline = time.time() + 3600 * self.args.max_hours if self.args.max_hours else None
        self.model.train()
        if self.step == 0:
            self.validate()
            if cfg["val"]["n_samples"]:
                self.sample_images()
        torch.cuda.reset_peak_memory_stats()
        while self.step < total:
            t0 = time.perf_counter()
            self.opt.zero_grad(set_to_none=True)
            loss_acc = 0.0
            for _ in range(cfg["grad_accum"]):
                batch = next(self.loader)
                loss = self.loss_on(batch, gen=self.gen) / cfg["grad_accum"]
                loss.backward()
                loss_acc += loss.item()
            if self.step == 0:
                self.gradient_check()
            gn = self.grad_norms() if (self.step % cfg["log_every"] == 0) else None
            self.opt.step()
            self.sched.step()
            self.step += 1
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            rec = {"step": self.step, "loss": loss_acc, "lr": self.sched.get_last_lr()[0], "sec_per_update": round(dt, 3),
                   "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3)}
            if gn:
                rec["grad_norm"] = gn
            append_jsonl(self.log_path, rec)
            if self.step % cfg["log_every"] == 0 or self.step == 1:
                print(json.dumps(rec), flush=True)
            if self.step % cfg["val"]["every"] == 0 or self.step == total:
                self.validate()
            keep = self.step in cfg.get("keep_steps", []) or self.step == total
            if self.step % cfg["save_every"] == 0 or keep:
                self.save(tag=keep)
                self.weight_check()
                if keep and cfg["val"]["n_samples"]:
                    self.sample_images()
            if deadline and time.time() > deadline:
                self.save()
                print(f"time budget reached at step {self.step}; saved for resume")
                return
        print("training finished")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--max_steps", type=int, default=None, help="override total steps (pilot runs)")
    ap.add_argument("--max_hours", type=float, default=None, help="stop+save before a job time limit")
    ap.add_argument("--name", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    if args.name:
        cfg["name"] = args.name
    Trainer(cfg, args).train()


if __name__ == "__main__":
    main()
