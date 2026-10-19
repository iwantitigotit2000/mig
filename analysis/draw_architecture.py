"""Schematic diagrams of the GLIGEN and MIGC control paths, annotated with probed tensor shapes.

Shapes come from outputs/analysis/{gligen,migc}_probe.json (one real forward pass, 2-object
layout mig277_s42, 512x512). Output: outputs/final_analysis/figures/arch_{gligen,migc}.png
"""
import json
import sys
from pathlib import Path

from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.plot_style import apply_style, plt, COLORS, TEXT, MUTED  # noqa: E402

FROZEN, GLIGEN_C, MIGC_C, IO = "#e6e6e6", COLORS["G0"], COLORS["MIGC"], "#f7f1dc"


def box(ax, x, y, w, h, text, fc=FROZEN, ec=MUTED, fs=7.5, bold=False, tc=TEXT):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=ec, lw=1))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=tc, weight="bold" if bold else "normal")


def arrow(ax, p, q, text=None, color=MUTED, ls="-", fs=6.5, off=(0, 0.04)):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, color=color, lw=1, ls=ls, shrinkA=1, shrinkB=1))
    if text:
        ax.text((p[0] + q[0]) / 2 + off[0], (p[1] + q[1]) / 2 + off[1], text, fontsize=fs, color=MUTED, ha="center")


def gligen(out):
    g = json.load(open(ROOT / "outputs/analysis/gligen_probe.json"))
    pn = g["position_net"]
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")
    ax.set_title("GLIGEN: grounding tokens injected by a gated self-attention layer in every transformer block (16 blocks)", fontsize=9)
    # inputs
    box(ax, 0.2, 3.9, 1.9, 0.7, "boxes (30×4)\nFourier enc. → 64-d", fc=IO)
    box(ax, 0.2, 2.9, 1.9, 0.7, "phrase CLIP text\nfeature (30×768)", fc=IO)
    box(ax, 2.6, 3.2, 2.3, 1.0, "position_net\nMLP 832→512→512→768\n1.08 M params\n(trainable in G1 and G2)", fc="#d6e6f9", ec=GLIGEN_C, fs=7)
    arrow(ax, (2.1, 4.25), (2.6, 3.9))
    arrow(ax, (2.1, 3.25), (2.6, 3.5))
    box(ax, 5.3, 3.3, 1.9, 0.8, "30 grounding tokens\n(B,30,768) + mask", fc=IO)
    arrow(ax, (4.9, 3.7), (5.3, 3.7))
    # transformer block
    box(ax, 0.2, 0.2, 9.6, 2.4, "", fc="#fafafa", ec="#bbbbbb")
    ax.text(0.35, 2.4, "one SD1.4 transformer block (×16, resolutions 64²→8²; C = 320 / 640 / 1280)", fontsize=7.5, color=MUTED)
    box(ax, 0.4, 0.9, 1.5, 0.9, "self-attn\n(frozen)")
    box(ax, 2.3, 0.8, 3.0, 1.1, "gated self-attn (fuser)\nattn over [x ; proj(objs)], HW+30 tokens\nx + tanh(α)·Attn(·)\nα per layer, |tanh α| < 0.06", fc="#d6e6f9", ec=GLIGEN_C, fs=7)
    box(ax, 5.7, 0.9, 1.5, 0.9, "cross-attn\ncaption 77×768\n(frozen)")
    box(ax, 7.6, 0.9, 1.3, 0.9, "FFN\n(frozen)")
    arrow(ax, (1.9, 1.35), (2.3, 1.35))
    arrow(ax, (5.3, 1.35), (5.7, 1.35))
    arrow(ax, (7.2, 1.35), (7.6, 1.35))
    arrow(ax, (6.2, 3.3), (3.8, 1.9), color=GLIGEN_C)
    ax.text(5.6, 2.55, "proj 768→C; the 16 fusers\n(208 M params) are trainable in G2", fontsize=6.5, color=GLIGEN_C, ha="left")
    ax.text(0.35, 0.4, f"probe: x_visual {g['fusers'][0]['x_visual_tokens']} → concat length {g['fusers'][0]['concat_len']} at 64²;  "
                       f"α_type=[0.3,0,0.7]: gate active only in the first 30 % of the 50 PLMS steps", fontsize=6.5, color=MUTED)
    fig.savefig(out / "arch_gligen.png")
    plt.close(fig)


def migc(out):
    m = json.load(open(ROOT / "outputs/analysis/migc_probe.json"))
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.3, 5.4)
    ax.axis("off")
    ax.set_title("MIGC: divide (per-instance shading) → conquer (SAC aggregation), replacing cross-attention in 4 low-resolution layers", fontsize=9)
    box(ax, 0.2, 4.3, 2.0, 0.7, "image tokens x\n(B, HW=64 or 256, 1280)", fc=IO)
    box(ax, 2.6, 4.3, 2.0, 0.7, "instance phrases\nCLIP (inst, 77, 768)", fc=IO)
    box(ax, 5.0, 4.3, 1.6, 0.7, "boxes (B, inst, 4)\n+ box masks", fc=IO)
    box(ax, 7.0, 4.3, 2.8, 0.7, "global caption cross-attn\n(vanilla SD1.4, frozen)", fc=FROZEN)
    # divide
    box(ax, 0.2, 2.6, 3.0, 1.1, "instance shading (per instance)\nmasked cross-attn x ↔ phrase\ninside its box → ca_x (B, inst, HW, C)", fc="#fbe3d6", ec=MIGC_C, fs=6.8)
    box(ax, 3.5, 2.6, 2.9, 1.1, "Enhancement Attention (EA)\ncontext = [pooled phrase ; pos_net(box)]\n(inst, HW, 1280) × (inst, 2, 768), residual", fc="#fbe3d6", ec=MIGC_C, fs=6.8)
    box(ax, 6.7, 2.6, 3.1, 1.1, "Layout Attention (LA)\nself-attn of x with layout mask\n(1, inst+1, H, W) → shading template", fc="#fbe3d6", ec=MIGC_C, fs=6.8)
    arrow(ax, (1.2, 4.3), (1.2, 3.7))
    arrow(ax, (3.6, 4.3), (2.0, 3.7))
    arrow(ax, (3.6, 4.3), (4.6, 3.7))
    arrow(ax, (5.8, 4.3), (5.4, 3.7))
    arrow(ax, (5.8, 4.3), (7.6, 3.7))
    arrow(ax, (2.2, 4.65), (8.3, 3.7), color="#bbbbbb")
    # conquer
    box(ax, 2.4, 0.9, 5.2, 1.0, "Shading Aggregation Controller (SAC)\nstack (B, inst+2, HW, C): inst shadings + global CA + LA template\nper-pixel softmax weights sac_scale (B, inst+2, H, W) → weighted sum → (B, 1, HW, C)",
        fc="#fbe3d6", ec=MIGC_C, fs=7)
    arrow(ax, (1.7, 2.6), (3.6, 1.9))
    arrow(ax, (5.0, 2.6), (5.0, 1.9))
    arrow(ax, (8.2, 2.6), (6.4, 1.9))
    arrow(ax, (9.3, 4.3), (9.3, 1.4), color="#bbbbbb")
    arrow(ax, (9.3, 1.4), (7.6, 1.4), color="#bbbbbb")
    box(ax, 3.4, 0.15, 3.2, 0.5, "output replaces the block's cross-attn result", fc=IO, fs=7)
    arrow(ax, (5.0, 0.9), (5.0, 0.65))
    layers = ", ".join(l["processor"].split(".transformer")[0] for l in m["migc_layers"])
    ax.text(0.2, -0.2, f"MIGC layers: {layers} (C=1280, 8²/16² tokens); {m['n_params_migc_total']/1e6:.1f} M params total;\n"
                       f"active for MIGCsteps = 25 of 50 Euler steps, then vanilla cross-attn", fontsize=6.6, color=MUTED, va="bottom")
    fig.savefig(out / "arch_migc.png")
    plt.close(fig)


if __name__ == "__main__":
    apply_style()
    out = ROOT / "outputs/final_analysis/figures"
    out.mkdir(parents=True, exist_ok=True)
    gligen(out)
    migc(out)
    print("written", out)
