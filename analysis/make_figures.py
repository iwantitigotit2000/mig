"""Result tables and figures for the test-set comparison GLIGEN (G0) vs MIGC. CPU only.

Reads  outputs/results/test_original/{instances,images}.csv, manifests/cocomig_difficulty.csv,
       outputs/gen/test/*/meta/*.json (provenance only)
Writes outputs/final_analysis/
         PROVENANCE.md                       where the numbers come from (machine, checkpoint path, timing)
         results_table.md / results_table.csv   report-named metrics with layout-bootstrap CIs and paired deltas
         subgroup_table.md                   Attribute Success by difficulty condition
         colour_threshold_curve.csv
         figures/test_*.png

Everything is recomputed from instances.csv, so rerunning after replacing the results directory
(e.g. with the cluster run) regenerates all tables and figures consistently.
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
from analysis.metric_names import INSTANCE_METRICS, IMAGE_METRICS, add_failure_columns  # noqa: E402
from analysis.plot_style import apply_style, color_of, label_of, plt, MUTED, TEXT  # noqa: E402

COLOUR_HEX = {"red": "#d62728", "blue": "#1f77b4", "green": "#2ca02c", "yellow": "#e8c100", "black": "#222222",
              "white": "#ffffff", "brown": "#8c564b"}
LEVELS = ["L2", "L3", "L4", "L5", "L6"]
OVERLAP = ["none", "low", "mid", "high"]
SIZEQ = ["Q1_smallest", "Q2", "Q3", "Q4_largest"]


def provenance(root, results, methods):
    """Describe where the numbers come from. The status is FINAL only when the results directory
    carries an explicit, human-written provenance.json; everything inferred from checkpoint paths is
    reported as evidence, never as a verdict."""
    try:
        display_results = results.relative_to(root)
    except ValueError:
        display_results = results
    lines = ["# Provenance of the test-set numbers", "", f"instances.csv: `{display_results / 'instances.csv'}`", ""]
    decl_path = results / "provenance.json"
    decl = json.load(open(decl_path)) if decl_path.exists() else None
    evidence = []
    for m in methods:
        gen = root / "outputs/gen/test" / m / "meta"
        metas = sorted(gen.glob("*.json"))
        if not metas:
            lines.append(f"- {m}: no generation metadata found under `{gen}` → hardware unknown")
            evidence.append("no-metadata")
            continue
        d = json.load(open(metas[0]))
        path = str(d["config"].get("checkpoint", {}).get("path", ""))
        # Keep the artifact identity needed for provenance without publishing the
        # originating workstation's absolute filesystem layout.
        artifact = Path(path.replace("\\", "/")).name if path else "not recorded"
        secs = [json.load(open(p))["seconds"] for p in metas]
        gpu = d.get("gpu") or d.get("env", {}).get("gpu")   # written by newer gen_runner versions only
        hint = "cluster-like path (/home/...)" if path.startswith("/home/") else \
               "Windows drive path (local workstation)" if len(path) > 2 and path[1:3] == ":/" else "path gives no hint"
        evidence.append(hint)
        lines.append(f"- {m}: {len(metas)} images; recorded GPU: {gpu or 'not recorded'}; checkpoint artifact `{artifact}` "
                     f"(absolute source path redacted; {hint}); "
                     f"mean {np.mean(secs):.2f} s/image; peak {d.get('peak_mem_allocated_gb', float('nan')):.2f} GB")
    if decl and decl.get("machine") and decl.get("declared_by"):
        status = f"FINAL — declared by {decl['declared_by']}: {decl['machine']}" + (f" ({decl.get('note')})" if decl.get("note") else "")
    elif any("Windows" in e for e in evidence):
        status = "PROVISIONAL — metadata points to a local-workstation run, but no explicit provenance.json confirms the machine and run owner"
    else:
        status = "UNVERIFIED — no provenance.json in the results directory and metadata is insufficient to identify the machine"
    lines += ["", f"**Status: {status}**", "",
              "To mark a results directory as final, add `provenance.json` next to instances.csv, e.g. "
              '`{"machine": "CCDS TC2 cluster, NVIDIA A40", "declared_by": "<name>", "job_id": "...", "note": "..."}`.',
              "", "Earlier group notes mention an A40 result of 0.319 / 0.669, while the committed run gives 0.321 / 0.665. "
              "The A40 raw files are not in the repository, so that result is neither verified nor used as the report's canonical result. "
              "If the files become available, add explicit provenance, select one run, and regenerate every Stage D artifact."]
    return "\n".join(lines) + "\n", status


def results_table(I, M, n_boot):
    rows = []
    for name, (tbl, col, hib, desc) in {**INSTANCE_METRICS, **IMAGE_METRICS}.items():
        if name == "Instance Success (MIG-Bench)":
            continue
        f = I if tbl == "instances" else M
        if col not in f or f[col].isna().all():
            continue
        a, b = f[f.method == "MIGC"], f[f.method == "G0"]
        g0, mg = cluster_estimate(b, col, n_boot=n_boot), cluster_estimate(a, col, n_boot=n_boot)
        d = paired_estimate(a, b, col, n_boot=n_boot)
        rows.append({"metric": name, "higher_is_better": hib, "G0": g0["estimate"], "G0_lo": g0["ci95"][0], "G0_hi": g0["ci95"][1],
                     "MIGC": mg["estimate"], "MIGC_lo": mg["ci95"][0], "MIGC_hi": mg["ci95"][1],
                     "delta": d["estimate"], "delta_lo": d["ci95"][0], "delta_hi": d["ci95"][1], "definition": desc})
    # conditional metrics
    for name, col, cond in [("Attribute Success | Position", "official_color", "official_pos == 1"),
                            ("Leakage share of colour errors", "crosstalk", "failure == 'colour_wrong'")]:
        vals = {}
        for m in ["G0", "MIGC"]:
            f = I[I.method == m]
            vals[m] = cluster_estimate(f.query(cond), col, n_boot=n_boot, universe=sorted(f.layout_id.unique()))
        rows.append({"metric": name, "higher_is_better": name.startswith("Attribute"), "G0": vals["G0"]["estimate"], "G0_lo": vals["G0"]["ci95"][0],
                     "G0_hi": vals["G0"]["ci95"][1], "MIGC": vals["MIGC"]["estimate"], "MIGC_lo": vals["MIGC"]["ci95"][0],
                     "MIGC_hi": vals["MIGC"]["ci95"][1], "delta": vals["MIGC"]["estimate"] - vals["G0"]["estimate"],
                     "delta_lo": np.nan, "delta_hi": np.nan, "definition": "conditional; unpaired difference"})
    df = pd.DataFrame(rows)
    md = ["| Metric | GLIGEN (G0) | MIGC | MIGC − G0 [95% CI] |", "|---|---:|---:|---|"]
    for r in df.itertuples():
        ci = f"{r.delta:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}]" if np.isfinite(r.delta_lo) else f"{r.delta:+.3f} (unpaired)"
        arrow = "↑" if r.higher_is_better else "↓"
        md.append(f"| {r.metric} {arrow} | {r.G0:.3f} [{r.G0_lo:.3f}, {r.G0_hi:.3f}] | {r.MIGC:.3f} [{r.MIGC_lo:.3f}, {r.MIGC_hi:.3f}] | {ci} |")
    return df, "\n".join(md) + "\n"


def subgroup(I, n_boot):
    recs = []
    for field, order in [("level", LEVELS), ("overlap_bucket", OVERLAP), ("same_cat_diff_color", [False, True]), ("relative_size_q", SIZEQ)]:
        for g in order:
            sub = I[I[field] == g]
            if sub.empty:
                continue
            a, b = sub[sub.method == "MIGC"], sub[sub.method == "G0"]
            d = paired_estimate(a, b, "official_success", n_boot=n_boot)
            recs.append({"group_by": field, "group": str(g), "n_instances_per_method": len(b),
                         **{f"G0_{k}": v for k, v in zip(["est", "lo", "hi"], [cluster_estimate(b, "official_success", n_boot=n_boot)[x] for x in ["estimate"]] + cluster_estimate(b, "official_success", n_boot=n_boot)["ci95"])},
                         **{f"MIGC_{k}": v for k, v in zip(["est", "lo", "hi"], [cluster_estimate(a, "official_success", n_boot=n_boot)["estimate"]] + cluster_estimate(a, "official_success", n_boot=n_boot)["ci95"])},
                         "delta": d["estimate"], "delta_lo": d["ci95"][0], "delta_hi": d["ci95"][1],
                         "G0_colour_wrong": b.fail_colour_wrong.mean(), "MIGC_colour_wrong": a.fail_colour_wrong.mean(),
                         "G0_loc_iou": b.loc_iou.mean(), "MIGC_loc_iou": a.loc_iou.mean()})
    df = pd.DataFrame(recs)
    md = ["| Condition | Group | n/method | GLIGEN Attr. Success | MIGC Attr. Success | MIGC − G0 [95% CI] | GLIGEN colour-wrong | MIGC colour-wrong |",
          "|---|---|---:|---:|---:|---|---:|---:|"]
    for r in df.itertuples():
        md.append(f"| {r.group_by} | {r.group} | {r.n_instances_per_method} | {r.G0_est:.3f} | {r.MIGC_est:.3f} | {r.delta:+.3f} [{r.delta_lo:+.3f}, {r.delta_hi:+.3f}] | {r.G0_colour_wrong:.3f} | {r.MIGC_colour_wrong:.3f} |")
    return df, "\n".join(md) + "\n"


def fig_subgroups(S, out):
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.2), sharey=True)
    titles = {"level": "objects per image", "overlap_bucket": "box overlap (max IoU with another box)",
              "same_cat_diff_color": "same category, different colour in image", "relative_size_q": "target size quartile (Q1 smallest)"}
    for ax, field in zip(axes, ["level", "overlap_bucket", "same_cat_diff_color", "relative_size_q"]):
        s = S[S.group_by == field]
        x = np.arange(len(s))
        for m in ["G0", "MIGC"]:
            ax.errorbar(x, s[f"{m}_est"], yerr=[s[f"{m}_est"] - s[f"{m}_lo"], s[f"{m}_hi"] - s[f"{m}_est"]],
                        color=color_of(m), marker="o", ms=5, capsize=2.5, lw=1.6, label=label_of(m))
        ax.set_xticks(x)
        ax.set_xticklabels([g.replace("_smallest", "").replace("_largest", "") for g in s.group], fontsize=8)
        ax.set_title(titles[field], fontsize=9)
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Attribute Success (MIG-Bench instance success)")
    axes[0].legend(loc="upper right")
    fig.suptitle("Attribute Success by difficulty condition, test set (100 layouts × 2 seeds; 95% layout-bootstrap CI)", fontsize=9.5, y=1.02)
    fig.savefig(out / "test_subgroups.png")
    plt.close(fig)


def fig_paired(R, I, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [1.1, 1]})
    ax = axes[0]
    rr = R[np.isfinite(R.delta_lo)].iloc[::-1]
    y = np.arange(len(rr))
    ax.barh(y, rr.delta, color=[color_of("MIGC") if (d > 0) == h else color_of("G0") for d, h in zip(rr.delta, rr.higher_is_better)], height=0.55, alpha=0.9)
    ax.errorbar(rr.delta, y, xerr=[rr.delta - rr.delta_lo, rr.delta_hi - rr.delta], fmt="none", ecolor=TEXT, capsize=3, lw=1)
    ax.axvline(0, color=MUTED, lw=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{m} {'↑' if h else '↓'}" for m, h in zip(rr.metric, rr.higher_is_better)])
    ax.set_xlabel("MIGC − GLIGEN, paired on identical (layout, seed, object); 95% layout-bootstrap CI")
    ax.set_title("Paired differences on every metric (orange = MIGC better)", fontsize=9.5)
    ax = axes[1]
    per = I.groupby(["method", "layout_id"]).official_success.mean().unstack(0)
    lv = I.drop_duplicates("layout_id").set_index("layout_id").level.reindex(per.index)
    for L in LEVELS:
        sel = lv == L
        ax.scatter(per.G0[sel], per.MIGC[sel], s=22, alpha=0.8, label=L, edgecolor="white", lw=0.4,
                   color=plt.cm.Blues(0.35 + 0.15 * LEVELS.index(L)))
    ax.plot([0, 1], [0, 1], color=MUTED, lw=0.9, ls="--")
    ax.set(xlabel="GLIGEN Attribute Success per layout (mean of 2 seeds)", ylabel="MIGC Attribute Success per layout",
           title=f"Per-layout pairing: MIGC better on {(per.MIGC > per.G0).sum()} / {len(per)} layouts, tie {(per.MIGC == per.G0).sum()}", xlim=(-0.03, 1.03), ylim=(-0.03, 1.03))
    ax.set_aspect("equal")
    ax.legend(title="objects", fontsize=7, title_fontsize=7, loc="lower right")
    fig.savefig(out / "test_paired.png")
    plt.close(fig)


def fig_failures(I, out):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    cats = [("success", "Attribute Success"), ("colour_wrong", "colour wrong (leakage)"), ("colour_wrong_other", "colour wrong (other)"),
            ("mislocated", "mislocated"), ("missing", "missing")]
    shades = {"success": "#bfbfbf", "colour_wrong": None, "colour_wrong_other": None, "mislocated": "#7f7f7f", "missing": "#3f3f3f"}
    for i, m in enumerate(["G0", "MIGC"]):
        f = I[I.method == m]
        n = len(f)
        parts = {"success": (f.failure == "success").sum() / n,
                 "colour_wrong": ((f.failure == "colour_wrong") & (f.crosstalk == 1)).sum() / n,
                 "colour_wrong_other": ((f.failure == "colour_wrong") & (f.crosstalk == 0)).sum() / n,
                 "mislocated": (f.failure == "mislocated").sum() / n, "missing": (f.failure == "missing").sum() / n}
        left = 0
        for key, lab in cats:
            c = {"colour_wrong": "#7a5195", "colour_wrong_other": "#cdb9dc"}.get(key, shades[key])
            ax.barh(i, parts[key], left=left, color=c, edgecolor="white", lw=1.5, label=lab if i == 0 else None)
            if parts[key] > 0.05:
                ax.text(left + parts[key] / 2, i, f"{parts[key]:.0%}", ha="center", va="center", fontsize=7.5, color="white" if key in ("missing", "mislocated", "colour_wrong") else TEXT)
            left += parts[key]
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["GLIGEN", "MIGC"])
    ax.set_xlim(0, 1)
    ax.set_title("Outcome of every target instance (n = 800 per method)", fontsize=9.5)
    ax.legend(ncol=3, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18))
    ax.grid(False)
    # colour confusion: target colour vs dominant colour among colour-wrong instances
    ax = axes[1]
    colours = list(COLOUR_HEX)
    mats = {}
    for m in ["G0", "MIGC"]:
        f = I[(I.method == m) & (I.failure == "colour_wrong")]
        mat = pd.crosstab(f.color, f.dominant_color).reindex(index=colours, columns=colours, fill_value=0)
        mats[m] = mat
    both = np.concatenate([mats["G0"].to_numpy(), np.full((7, 1), np.nan), mats["MIGC"].to_numpy()], axis=1)
    im = ax.imshow(both, cmap="Blues", vmin=0)
    ax.set_xticks(list(range(7)) + list(range(8, 15)))
    ax.set_xticklabels(colours + colours, rotation=60, fontsize=7)
    ax.set_yticks(range(7))
    ax.set_yticklabels(colours, fontsize=7)
    ax.set_xlabel("dominant colour in the SAM mask          |          dominant colour in the SAM mask")
    ax.set_ylabel("target colour")
    ax.set_title(f"Colour-wrong instances: GLIGEN (left, n={int(mats['G0'].to_numpy().sum())})  vs  MIGC (right, n={int(mats['MIGC'].to_numpy().sum())})", fontsize=9)
    for k, m in enumerate(["G0", "MIGC"]):
        for i in range(7):
            for j in range(7):
                v = mats[m].iat[i, j]
                if v:
                    ax.text(j + 8 * k, i, str(v), ha="center", va="center", fontsize=6.5, color="white" if v > both[np.isfinite(both)].max() * 0.6 else TEXT)
    ax.grid(False)
    fig.savefig(out / "test_failure_modes.png")
    plt.close(fig)


def colour_threshold_curve(I, out_dir, out_fig, n_boot):
    ths = np.round(np.arange(0.05, 0.51, 0.025), 3)
    rows = []
    for t in ths:
        f = I.copy()
        f["s"] = (f.official_pos.eq(1) & f.target_color_ratio.ge(t) & f.empty_mask.eq(0)).astype(float)
        a, b = f[f.method == "MIGC"], f[f.method == "G0"]
        d = paired_estimate(a, b, "s", n_boot=max(1000, n_boot // 5))
        rows.append({"threshold": t, "G0": b.s.mean(), "MIGC": a.s.mean(), "delta": d["estimate"], "delta_lo": d["ci95"][0], "delta_hi": d["ci95"][1]})
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "colour_threshold_curve.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    for m in ["G0", "MIGC"]:
        axes[0].plot(df.threshold, df[m], color=color_of(m), marker="o", ms=3, label=label_of(m))
    axes[0].axvline(0.2, color=MUTED, ls=":", lw=1)
    axes[0].text(0.205, 0.95, "official 0.2", fontsize=7, color=MUTED)
    axes[0].set(xlabel="colour-coverage threshold (fraction of SAM mask)", ylabel="Attribute Success", title="Score vs evaluator colour threshold", ylim=(0, 1))
    axes[0].legend()
    axes[1].plot(df.threshold, df.delta, color=TEXT, marker="o", ms=3)
    axes[1].fill_between(df.threshold, df.delta_lo, df.delta_hi, color="#cccccc", alpha=0.5, label="95% CI (paired)")
    axes[1].axvline(0.2, color=MUTED, ls=":", lw=1)
    axes[1].set(xlabel="colour-coverage threshold", ylabel="MIGC − GLIGEN", title="Gap is stable across thresholds", ylim=(0, 0.5))
    axes[1].legend()
    fig.savefig(out_fig / "test_colour_threshold.png")
    plt.close(fig)
    return df


def fig_cost(root, out):
    recs = []
    for m in ["G0", "MIGC"]:
        for p in sorted((root / "outputs/gen/test" / m / "meta").glob("*.json")):
            d = json.load(open(p))
            recs.append({"method": m, "seconds": d["seconds"], "mem": d["peak_mem_allocated_gb"], "n": len(d["objects"])})
    if not recs:
        return
    df = pd.DataFrame(recs)
    fig, ax = plt.subplots(figsize=(5, 3))
    for m in ["G0", "MIGC"]:
        g = df[df.method == m].groupby("n").seconds.agg(["mean", "std"]).reset_index()
        ax.errorbar(g.n, g["mean"], yerr=g["std"], color=color_of(m), marker="o", ms=5, capsize=3, label=f"{label_of(m)} (peak {df[df.method == m].mem.max():.1f} GB)")
    ax.set(xlabel="objects per image", ylabel="seconds per image (50 steps, 512², fp16)", title="Generation cost vs number of instances")
    ax.legend()
    fig.savefig(out / "test_cost.png")
    plt.close(fig)
    df.to_csv(out.parent / "generation_cost.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--results", type=Path, default=None, help="default outputs/results/test_original")
    ap.add_argument("--out", type=Path, default=None, help="default outputs/final_analysis")
    ap.add_argument("--n_boot", type=int, default=5000)
    args = ap.parse_args()
    root = args.root.resolve()
    results = (args.results or root / "outputs/results/test_original").resolve()
    out = (args.out or root / "outputs/final_analysis").resolve()
    figs = out / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    apply_style()

    I = add_failure_columns(pd.read_csv(results / "instances.csv"))
    M = pd.read_csv(results / "images.csv")
    D = pd.read_csv(root / "manifests/cocomig_difficulty.csv")
    D = D[D.split == "test"][["sample_id", "idx", "relative_size_q", "size_bucket_coco", "normalised_area"]]
    I = I.merge(D, on=["sample_id", "idx"], how="left", validate="many_to_one")
    assert I.relative_size_q.notna().all(), "difficulty annotations missing; run data_tools/annotate_difficulty.py"

    prov, status = provenance(root, results, ["G0", "MIGC"])
    (out / "PROVENANCE.md").write_text(prov, encoding="utf-8")
    R, md = results_table(I, M, args.n_boot)
    R.to_csv(out / "results_table.csv", index=False)
    (out / "results_table.md").write_text(f"# Test-set results ({status})\n\n" + md, encoding="utf-8")
    S, smd = subgroup(I, args.n_boot)
    S.to_csv(out / "subgroup_table.csv", index=False)
    (out / "subgroup_table.md").write_text(f"# Attribute Success by difficulty condition ({status})\n\n" + smd, encoding="utf-8")
    fig_subgroups(S, figs)
    fig_paired(R, I, figs)
    fig_failures(I, figs)
    colour_threshold_curve(I, out, figs, args.n_boot)
    fig_cost(root, figs)
    print(prov)
    print(md)
    print(smd)


if __name__ == "__main__":
    main()
