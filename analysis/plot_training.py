"""Training curves and fine-tuning analysis from committed logs (no GPU).

Inputs : outputs/runs/<run>/{train_log.jsonl,val_log.jsonl,weight_check_*.json,run_info.json}
         outputs/results/val_finetune/{instances.csv,images.csv}
Outputs: outputs/final_analysis/figures/train_*.png, finetune_*.png
         outputs/final_analysis/finetune_tables.md  (+ json)
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.final_statistics import cluster_estimate, paired_estimate  # noqa: E402
from analysis.metric_names import add_failure_columns  # noqa: E402
from analysis.plot_style import apply_style, color_of, label_of, plt, MUTED, TEXT  # noqa: E402

RUNS = ["g1_lr5e-5", "g2_lr5e-5", "g2_lr1e-5"]


def smooth(y, k=50):
    y = np.asarray(y, float)
    if len(y) < k:
        return y
    c = np.cumsum(np.insert(y, 0, 0))
    out = (c[k:] - c[:-k]) / k
    return np.concatenate([np.full(k - 1, np.nan), out])


def load_runs(root):
    runs = {}
    for r in RUNS:
        d = root / "outputs/runs" / r
        if not d.exists():
            continue
        tl = pd.DataFrame([json.loads(l) for l in open(d / "train_log.jsonl") if l.strip()])
        vl = pd.DataFrame([json.loads(l) for l in open(d / "val_log.jsonl") if l.strip()])
        wc = []
        for p in sorted(d.glob("weight_check_*.json")):
            j = json.load(open(p))
            wc.append({"step": j["step"], **{f"delta_{k}": v for k, v in j["trainable_delta_l2_by_group"].items()}})
        gn = []
        for _, row in tl.iterrows():
            if isinstance(row.get("grad_norm"), dict):
                gn.append({"step": row["step"], **row["grad_norm"]})
        runs[r] = {"train": tl, "val": vl, "weights": pd.DataFrame(wc), "grad": pd.DataFrame(gn),
                   "info": json.load(open(d / "run_info.json"))}
    return runs


def fig_training(runs, out):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    ax = axes[0]
    ref = runs["g1_lr5e-5"]["train"] if "g1_lr5e-5" in runs else None
    for r, d in runs.items():
        ax.plot(d["train"].step, smooth(d["train"].loss), color=color_of(r), label=label_of(r),
                lw=2.2 if r == "g1_lr5e-5" else 1.0, alpha=1 if r == "g1_lr5e-5" else 0.9)
    ax.set(title="(a) Training ε-MSE, 50-step MA\n(same batches for all runs: curves overlap)", xlabel="optimiser step", ylabel="ε-MSE")
    ax.legend()
    ax = axes[1]
    if ref is not None:
        for r, d in runs.items():
            if r == "g1_lr5e-5":
                continue
            m = d["train"][["step", "loss"]].merge(ref[["step", "loss"]], on="step", suffixes=("", "_g1"))
            ax.plot(m.step, smooth((m.loss - m.loss_g1) * 1e3, 100), color=color_of(r), label=f"{label_of(r)} − G1")
        ax.axhline(0, color=MUTED, lw=0.8)
    ax.set(title="(b) Same-batch loss, G2 − G1\n(×10⁻³, 100-step MA)", xlabel="optimiser step", ylabel="Δ loss ×10⁻³")
    ax.legend()
    ax = axes[2]
    for r, d in runs.items():
        v = d["val"]
        ax.plot(v.step, (v.val_loss - v.val_loss.iloc[0]) * 1e3, color=color_of(r), marker="o", ms=4, label=label_of(r))
    ax.axhline(0, color=MUTED, lw=0.8)
    n_val = int(next(iter(runs.values()))["val"].n_images.iloc[0]) if runs else 0
    ax.set(title=f"(c) Held-out loss vs step 0\n(×10⁻³, {n_val} COCO train2014 images)", xlabel="optimiser step", ylabel="Δ val loss ×10⁻³")
    ax.legend()
    fig.savefig(out / "train_curves.png")
    plt.close(fig)


def fig_weights(runs, out):
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for r, d in runs.items():
        w = d["weights"]
        axes[0].plot(w.step, w["delta_position_net"], color=color_of(r), marker="o", ms=4, label=label_of(r))
        if "delta_fuser" in w:
            axes[1].plot(w.step, w["delta_fuser"], color=color_of(r), marker="o", ms=4, label=label_of(r))
    axes[0].set(title="position_net: ‖θ_t − θ_0‖₂", xlabel="optimiser step", ylabel="L2 change")
    axes[1].set(title="fuser (gated self-attention): ‖θ_t − θ_0‖₂", xlabel="optimiser step")
    for a in axes:
        a.legend(fontsize=7)
    fig.savefig(out / "train_weight_drift.png")
    plt.close(fig)


def fig_gates(runs, root, out):
    """Per-layer tanh(alpha_attn) of the GLIGEN gates before/after G2 fine-tuning."""
    before = {f["name"] + ".alpha_attn": f["tanh_alpha_attn"] for f in json.load(open(root / "outputs/analysis/gligen_probe.json"))["fusers"]}
    fig, ax = plt.subplots(figsize=(8, 3))
    names = list(before)
    x = np.arange(len(names))
    ax.bar(x - 0.27, [before[n] for n in names], width=0.27, color="#9e9e9e", label="official checkpoint")
    for k, r in enumerate(["g2_lr1e-5", "g2_lr5e-5"]):
        if r not in runs:
            continue
        after = json.load(open(root / "outputs/runs" / r / "weight_check_002000.json"))["tanh_alpha_attn"]
        ax.bar(x + 0.27 * k, [after.get(n, np.nan) for n in names], width=0.27, color=color_of(r), label=f"{label_of(r)} @2000")
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace(".transformer_blocks.0.fuser.alpha_attn", "").replace("_blocks", "") for n in names], rotation=60, ha="right", fontsize=7)
    ax.set(title="GLIGEN gate values tanh(α_attn) per fuser layer", ylabel="tanh(α)")
    ax.legend()
    fig.savefig(out / "train_gate_values.png")
    plt.close(fig)


def finetune_tables(root, out, n_boot):
    V = add_failure_columns(pd.read_csv(root / "outputs/results/val_finetune/instances.csv"))
    methods = ["G0"] + [m for m in sorted(V.method.unique()) if m != "G0"]
    metrics = [("Attribute Success", "official_success"), ("Position Success", "official_pos"),
               ("Object Recall", "o2o_present"), ("Localisation IoU", "loc_iou"),
               ("Colour Error Rate", "fail_colour_wrong"), ("Color Leakage Rate", "crosstalk")]
    table, paired = {}, {}
    for name, col in metrics:
        table[name] = {m: cluster_estimate(V[V.method == m], col, n_boot=n_boot) for m in methods}
        paired[name] = {m: paired_estimate(V[V.method == m], V[V.method == "G0"], col, n_boot=n_boot) for m in methods if m != "G0"}
    # forest plot of paired differences vs G0
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), sharey=True)
    for ax, name in zip(axes, ["Attribute Success", "Localisation IoU", "Colour Error Rate"]):
        ys = []
        for i, m in enumerate([m for m in methods if m != "G0"]):
            d = paired[name][m]
            ax.errorbar(d["estimate"], i, xerr=[[d["estimate"] - d["ci95"][0]], [d["ci95"][1] - d["estimate"]]],
                        fmt="o", color=color_of(m), ms=6, capsize=3, lw=1.4)
            ys.append(label_of(m))
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels(ys)
        ax.set_title(f"Δ {name} vs GLIGEN (val, paired)")
        ax.set_xlabel("difference (95% layout-bootstrap CI)")
    fig.savefig(out / "finetune_paired_vs_g0.png")
    plt.close(fig)
    # by level
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    levels = ["L2", "L3", "L4", "L5", "L6"]
    for m in methods:
        f = V[V.method == m]
        y = [f[f.level == L].official_success.mean() for L in levels]
        ls = "--" if m.endswith("s000500") else "-"
        ax.plot(levels, y, color=color_of(m), ls=ls, marker="o", ms=4, label=label_of(m), lw=1.5 if m != "G0" else 2.4)
    ax.set(title="Attribute Success by object count (validation, 30 layouts × 2 seeds)", ylabel="Attribute Success", xlabel="objects per image")
    ax.legend(ncol=2, fontsize=7)
    fig.savefig(out / "finetune_by_level.png")
    plt.close(fig)
    lines = ["# Fine-tuning results on the validation set (r = 0.5, 30 layouts × 2 seeds)", "",
             "Layout bootstrap 95% CI; differences are paired on identical (sample, object). Exploratory: 30 layouts only.", "",
             "| Metric | " + " | ".join(label_of(m) for m in methods) + " |", "|---|" + "---:|" * len(methods)]
    for name, _ in metrics:
        lines.append(f"| {name} | " + " | ".join(f"{table[name][m]['estimate']:.3f}" for m in methods) + " |")
    lines += ["", "| Paired difference vs GLIGEN official | " + " | ".join(label_of(m) for m in methods if m != "G0") + " |",
              "|---|" + "---:|" * (len(methods) - 1)]
    for name, _ in metrics:
        lines.append(f"| {name} | " + " | ".join(f"{paired[name][m]['estimate']:+.3f} [{paired[name][m]['ci95'][0]:+.3f}, {paired[name][m]['ci95'][1]:+.3f}]" for m in methods if m != "G0") + " |")
    (out.parent / "finetune_tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out.parent / "finetune_tables.json").write_text(json.dumps({"point": table, "paired_vs_G0": paired}, indent=2) + "\n")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()
    root = args.root.resolve()
    out = (args.out or root / "outputs/final_analysis/figures")
    out.mkdir(parents=True, exist_ok=True)
    apply_style()
    runs = load_runs(root)
    fig_training(runs, out)
    fig_weights(runs, out)
    fig_gates(runs, root, out)
    finetune_tables(root, out, args.n_boot)
    # compact run facts for the report
    facts = {r: {"trainable_params": json.load(open(root / "outputs/runs" / r / "trainable_params.json"))["n_trainable"],
                 "trainable_fraction": json.load(open(root / "outputs/runs" / r / "trainable_params.json"))["fraction"],
                 "sec_per_update_median": float(d["train"].sec_per_update.median()),
                 "peak_mem_gb": float(d["train"].peak_mem_gb.max()),
                 "final_train_loss_ma50": float(np.nanmean(d["train"].loss.tail(50))),
                 "val_loss_step0": float(d["val"].val_loss.iloc[0]), "val_loss_final": float(d["val"].val_loss.iloc[-1]),
                 "weight_delta_final": d["weights"].iloc[-1].drop("step").to_dict()} for r, d in runs.items()}
    (out.parent / "training_facts.json").write_text(json.dumps(facts, indent=2) + "\n")
    print(json.dumps(facts, indent=2))


if __name__ == "__main__":
    main()
