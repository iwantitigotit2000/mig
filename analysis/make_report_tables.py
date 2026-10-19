"""Generate every number-bearing piece of report/report.tex from the analysis outputs.

Writes report/tables/
  main_table.tex     body of the test-set results table   (from outputs/final_analysis/results_table.csv)
  ft_table.tex       body of the fine-tuning table        (from outputs/final_analysis/finetune_tables.json + val instances)
  numbers.tex        \\newcommand macros for every number quoted in the prose
                     (from results_table.csv, subgroup_table.csv, finetune_tables.json, statistics.json,
                      colour_threshold_curve.csv, instances.csv, training logs)
report.tex \\input{}s these files, so no result number is typed by hand. analysis/check_report_numbers.py
verifies that the committed .tex files equal a fresh regeneration and that no stray hand-typed numbers remain.
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.metric_names import add_failure_columns  # noqa: E402

FA = ROOT / "outputs/final_analysis"
OUT = ROOT / "report/tables"


def f3(x):
    return f"{x:.3f}"


def sgn(x, nd=3):
    s = f"{x:+.{nd}f}"
    return s.replace("-", "$-$")


def ci(lo, hi, nd=3):
    return f"[{sgn(lo, nd)}, {sgn(hi, nd)}]"


def pct(x, nd=0):
    return f"{100 * x:.{nd}f}\\%"


def main_table(R):
    order = [("Object Recall", True), ("Position Success", True), ("Localisation IoU", True), ("Attribute Success", True),
             ("Attribute Success | Position", True), ("Official mIoU", True), ("Image Success", True),
             ("Color Leakage Rate", False), ("Missing Rate", False), ("Mislocation Rate", False), ("CLIP-IQA", True)]
    lines = []
    for name, hib in order:
        r = R.loc[name]
        arrow = "$\\uparrow$" if hib else "$\\downarrow$"
        if name == "Attribute Success":
            label = "\\textbf{Attribute Success} " + arrow
            delta = f"\\textbf{{{sgn(r.delta)} {ci(r.delta_lo, r.delta_hi)}}}"
        elif name == "Attribute Success | Position":
            label = "~~given position OK"
            delta = f"{sgn(r.delta)} (unpaired)"
        else:
            label = f"{name} {arrow}"
            delta = f"{sgn(r.delta)} {ci(r.delta_lo, r.delta_hi)}"
        lines.append(f"{label} & {f3(r.G0)} & {f3(r.MIGC)} & {delta} \\\\")
    return "\n".join(lines) + "\n\\bottomrule%\n"


def ft_table(F, V):
    runs = [("official", "G0"), ("G1 5e-5 @500", "g1_lr5e-5_s000500"), ("G1 5e-5 @2000", "g1_lr5e-5_s002000"),
            ("G2 1e-5 @500", "g2_lr1e-5_s000500"), ("G2 1e-5 @2000", "g2_lr1e-5_s002000"),
            ("G2 5e-5 @500", "g2_lr5e-5_s000500"), ("G2 5e-5 @2000", "g2_lr5e-5_s002000")]
    lines = []
    for label, m in runs:
        att = F["point"]["Attribute Success"][m]["estimate"]
        iou = F["point"]["Localisation IoU"][m]["estimate"]
        sub = V[(V.method == m) & (V.official_pos == 1)]
        colpos = sub.official_color.mean()
        if m == "G0":
            d = "--"
        else:
            p = F["paired_vs_G0"]["Attribute Success"][m]
            d = f"{sgn(p['estimate'])} {ci(*p['ci95'])}"
            if p["ci95"][0] > 0:
                d = f"\\textbf{{{d}}}"
        lines.append(f"{label} & {f3(att)} & {d} & {f3(colpos)} & {f3(iou)} \\\\")
    return "\n".join(lines) + "\n\\bottomrule%\n"


def numbers(R, S, F, V, I, C, stats):
    g = lambda name, col: R.loc[name, col]
    N = {}
    # headline
    N["AttrG"], N["AttrM"] = f3(g("Attribute Success", "G0")), f3(g("Attribute Success", "MIGC"))
    N["AttrDelta"] = sgn(g("Attribute Success", "delta"))
    N["AttrDeltaCI"] = ci(g("Attribute Success", "delta_lo"), g("Attribute Success", "delta_hi"))
    N["AttrGap"] = f"{g('Attribute Success', 'delta'):.2f}"
    N["RecallG"], N["RecallM"] = f"{g('Object Recall', 'G0'):.2f}", f"{g('Object Recall', 'MIGC'):.2f}"
    N["PosDelta"] = f"{g('Position Success', 'delta'):+.2f}"
    N["IoUDelta"] = f"{g('Localisation IoU', 'delta'):+.2f}"
    N["ColPosG"] = pct(g("Attribute Success | Position", "G0"))
    N["ColPosM"] = pct(g("Attribute Success | Position", "MIGC"))
    N["ImgG"], N["ImgM"] = f3(g("Image Success", "G0")), f3(g("Image Success", "MIGC"))
    N["IQADelta"] = sgn(g("CLIP-IQA", "delta"))
    N["LeakG"], N["LeakM"] = pct(g("Color Leakage Rate", "G0")), pct(g("Color Leakage Rate", "MIGC"))
    N["ColWrongG"], N["ColWrongM"] = pct(g("Colour Error Rate", "G0")), pct(g("Colour Error Rate", "MIGC"))
    N["MissG"], N["MissM"] = pct(g("Missing Rate", "G0")), pct(g("Missing Rate", "MIGC"))
    N["MislocG"], N["MislocM"] = pct(g("Mislocation Rate", "G0")), pct(g("Mislocation Rate", "MIGC"))
    N["LeakShareG"] = pct(g("Leakage share of colour errors", "G0"))
    # per-layout pairing
    per = I.groupby(["method", "layout_id"]).official_success.mean().unstack(0)
    N["LayoutsBetter"], N["LayoutsTotal"] = str(int((per.MIGC > per.G0).sum())), str(len(per))
    N["InstPerImage"] = f"{len(I[I.method == 'G0']) / I[I.method == 'G0'].sample_id.nunique():.0f}"
    N["InstPerMethod"] = str(len(I[I.method == "G0"]))
    # colour confusion among MIGC colour-wrong
    cw = I[(I.method == "MIGC") & (I.failure == "colour_wrong")]
    pairs = {("yellow", "brown"): "YB", ("brown", "black"): "BrBl", ("yellow", "black"): "YBl"}
    tot = 0
    for (a, b), k in pairs.items():
        n = int(((cw.color == a) & (cw.dominant_color == b)).sum())
        N[f"Conf{k}"] = str(n)
        tot += n
    N["ConfThree"], N["ConfTotal"] = str(tot), str(len(cw))
    N["ConfShare"] = pct(tot / len(cw)) if len(cw) else "n/a"
    # subgroups
    sg = lambda f, grp, col: S[(S.group_by == f) & (S.group == grp)].iloc[0][col]
    N["LvlTwoG"], N["LvlSixG"] = f"{sg('level', 'L2', 'G0_est'):.2f}", f"{sg('level', 'L6', 'G0_est'):.2f}"
    N["LvlTwoM"], N["LvlSixM"] = f"{sg('level', 'L2', 'MIGC_est'):.2f}", f"{sg('level', 'L6', 'MIGC_est'):.2f}"
    N["OvNoneM"], N["OvHighM"] = f"{sg('overlap_bucket', 'none', 'MIGC_est'):.2f}", f"{sg('overlap_bucket', 'high', 'MIGC_est'):.2f}"
    N["OvNoneDelta"] = f"{sg('overlap_bucket', 'none', 'delta'):+.2f}"
    N["OvHighDelta"] = f"{sg('overlap_bucket', 'high', 'delta'):+.2f}"
    N["OvHighDeltaCI"] = ci(sg("overlap_bucket", "high", "delta_lo"), sg("overlap_bucket", "high", "delta_hi"), 2)
    N["SameCatMaxChange"] = f"{max(abs(sg('same_cat_diff_color', 'False', c) - sg('same_cat_diff_color', 'True', c)) for c in ['G0_est', 'MIGC_est']):.2f}"
    N["SizeQoneG"], N["SizeQfourG"] = f"{sg('relative_size_q', 'Q1_smallest', 'G0_est'):.2f}", f"{sg('relative_size_q', 'Q4_largest', 'G0_est'):.2f}"
    N["SizeQoneM"], N["SizeQfourM"] = f"{sg('relative_size_q', 'Q1_smallest', 'MIGC_est'):.2f}", f"{sg('relative_size_q', 'Q4_largest', 'MIGC_est'):.2f}"
    # threshold curve
    N["ThrGLo"], N["ThrGHi"] = f"{C.G0.iloc[-1]:.2f}", f"{C.G0.iloc[0]:.2f}"
    N["ThrMLo"], N["ThrMHi"] = f"{C.MIGC.iloc[-1]:.2f}", f"{C.MIGC.iloc[0]:.2f}"
    N["ThrGapLo"], N["ThrGapHi"] = f"{C.delta.min():.2f}", f"{C.delta.max():.2f}"
    N["ThrGapAllPositive"] = "never include zero" if (C.delta_lo > 0).all() else "include zero at some thresholds"
    # fine-tuning
    p = F["paired_vs_G0"]["Attribute Success"]
    N["FtGoneFive"] = sgn(p["g1_lr5e-5_s000500"]["estimate"])
    N["FtGtwoFive"] = sgn(p["g2_lr5e-5_s000500"]["estimate"])
    N["FtMaxGain"] = f"{max(v['estimate'] for v in p.values()):+.2f}"
    N["FtMinGain"] = f"{min(v['estimate'] for k, v in p.items() if k.startswith('g1') or 'lr5e-5' in k):+.2f}"
    vp = stats["validation_paired"]
    d = vp["g2_lr5e-5_s000500 minus g2_lr1e-5_s000500"]
    N["LrContrast"] = f"{sgn(d['estimate'])} {ci(*d['ci95'])}"
    d = vp["g2_lr5e-5_s000500 minus g1_lr5e-5_s000500"]
    N["GoneGtwoContrast"] = f"{sgn(d['estimate'])} {ci(*d['ci95'])}"
    g1 = F["point"]["Attribute Success"]
    N["GoneFiveHundred"], N["GoneTwoThousand"] = f3(g1["g1_lr5e-5_s000500"]["estimate"]), f3(g1["g1_lr5e-5_s002000"]["estimate"])
    colpos = {m: V[(V.method == m) & (V.official_pos == 1)].official_color.mean() for m in V.method.unique()}
    N["ColPosValG"] = f"{colpos['G0']:.2f}"
    N["ColPosValLo"] = f"{min(v for k, v in colpos.items() if k != 'G0' and '5e-5' in k):.2f}"
    N["ColPosValHi"] = f"{max(v for k, v in colpos.items() if k != 'G0'):.2f}"
    lk = F["paired_vs_G0"]["Color Leakage Rate"]["g2_lr5e-5_s000500"]
    N["LeakValG"] = f"{F['point']['Color Leakage Rate']['G0']['estimate']:.2f}"
    N["LeakValGtwo"] = f"{F['point']['Color Leakage Rate']['g2_lr5e-5_s000500']['estimate']:.2f}"
    N["LeakValCI"] = ci(*lk["ci95"], 2)
    N["RecallValG"] = f"{F['point']['Object Recall']['G0']['estimate']:.2f}"
    # training facts
    T = json.load(open(FA / "training_facts.json"))
    N["ValImages"] = str(int(pd.read_json(ROOT / "outputs/runs/g1_lr5e-5/val_log.jsonl", lines=True).n_images.iloc[0]))
    N["PosNetDrift"] = f"{T['g1_lr5e-5']['weight_delta_final']['delta_position_net']:.1f}"
    gates = json.load(open(ROOT / "outputs/runs/g2_lr5e-5/weight_check_002000.json"))["tanh_alpha_attn"]
    before = {f["name"] + ".alpha_attn": f["tanh_alpha_attn"] for f in json.load(open(ROOT / "outputs/analysis/gligen_probe.json"))["fusers"]}
    N["GateMaxChange"] = f"{max(abs(gates[k] - before[k]) for k in gates):.3f}"
    N["GateMaxAbs"] = f"{max(abs(v) for v in before.values()):.2f}"
    # control steps
    N["GligenSteps"] = str(int(round(0.3 * 50)))
    N["MigcSteps"] = "25"
    # architecture facts from probes
    gp = json.load(open(ROOT / "outputs/analysis/gligen_probe.json"))["n_params"]
    mp = json.load(open(ROOT / "outputs/analysis/migc_probe.json"))
    N["PosNetParamsM"] = f"{gp['position_net'] / 1e6:.2f}"
    N["FuserParamsM"] = f"{gp['fusers_total'] / 1e6:.0f}"
    N["FuserFrac"] = f"{100 * gp['fusers_total'] / gp['unet_total']:.1f}\\%"
    N["GligenAddedM"] = f"{(gp['fusers_total'] + gp['position_net']) / 1e6:.0f}"
    N["GligenAddedFrac"] = f"{100 * (gp['fusers_total'] + gp['position_net']) / gp['unet_total']:.1f}\\%"
    N["PosNetFrac"] = f"{100 * gp['position_net'] / gp['unet_total']:.2f}\\%"
    N["MigcParamsM"] = f"{mp['n_params_migc_total'] / 1e6:.1f}"
    N["MigcFrac"] = f"{100 * mp['n_params_migc_total'] / (mp['n_params_unet'] - mp['n_params_migc_total']):.1f}\\%"
    N["SdUnetM"] = f"{(mp['n_params_unet'] - mp['n_params_migc_total']) / 1e6:.0f}"
    N["FuserCount"] = str(len(json.load(open(ROOT / "outputs/analysis/gligen_probe.json"))["fusers"]))
    # generation cost (from committed generation metadata)
    G = pd.read_csv(FA / "generation_cost.csv")
    for m, k in [("G0", "G"), ("MIGC", "M")]:
        N[f"Sec{k}"] = f"{G[G.method == m].seconds.mean():.1f}"
        N[f"Mem{k}"] = f"{G[G.method == m].mem.max():.1f}"
    N["TrainMemGone"] = f"{T['g1_lr5e-5']['peak_mem_gb']:.1f}"
    N["TrainMemGtwo"] = f"{T['g2_lr5e-5']['peak_mem_gb']:.1f}"
    N["ParamRatio"] = str(int(round((gp['fusers_total'] + gp['position_net']) / gp['position_net'])))
    lines = ["% generated by analysis/make_report_tables.py -- do not edit"]
    for k, v in N.items():
        assert re.fullmatch(r"[A-Za-z]+", k), k
        lines.append(f"\\newcommand{{\\{k}}}{{{v}}}")
    return "\n".join(lines) + "\n"


def build():
    R = pd.read_csv(FA / "results_table.csv").set_index("metric")
    S = pd.read_csv(FA / "subgroup_table.csv")
    F = json.load(open(FA / "finetune_tables.json"))
    V = add_failure_columns(pd.read_csv(ROOT / "outputs/results/val_finetune/instances.csv"))
    I = add_failure_columns(pd.read_csv(ROOT / "outputs/results/test_original/instances.csv"))
    C = pd.read_csv(FA / "colour_threshold_curve.csv")
    stats = json.load(open(FA / "statistics.json"))
    return {"main_table.tex": main_table(R), "ft_table.tex": ft_table(F, V), "numbers.tex": numbers(R, S, F, V, I, C, stats)}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in build().items():
        (OUT / name).write_text(text, encoding="utf-8")
    print("wrote", ", ".join(f"report/tables/{n}" for n in build()))


if __name__ == "__main__":
    main()
