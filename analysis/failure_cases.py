"""Side-by-side failure / success cases from the committed test images (CPU only).

Selection is rule-based and deterministic from instances.csv so the same cases are chosen
whenever the script runs on the same results; it does not presuppose that every category has
examples (empty categories are reported as such).

Categories
  A  GLIGEN colour leakage, MIGC all correct            (the dominant failure mode)
  B  GLIGEN misses or mislocates an object, MIGC correct
  C  MIGC fails where GLIGEN succeeds                    (counter-examples)
  D  both fail on a high-overlap layout
  E  MIGC "colour wrong" with yellow/brown or black/brown dominant colour (possible evaluator artefact)

Outputs  outputs/final_analysis/failure_cases/cases.csv  (case id, sample, per-object status, auto description,
         empty `human_observation` column to fill after looking at the images)
         outputs/final_analysis/figures/cases_<category>.png  contact sheets with GT boxes
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.plot_style import apply_style, plt, TEXT, MUTED  # noqa: E402

COLOUR_RGB = {"red": (214, 39, 40), "blue": (31, 119, 180), "green": (44, 160, 44), "yellow": (232, 193, 0),
              "black": (20, 20, 20), "white": (245, 245, 245), "brown": (140, 86, 75)}
STATUS_MARK = {"success": "✓", "colour_wrong": "✗colour", "mislocated": "✗pos", "missing": "✗missing", "eval_error": "?"}


def draw(img_path, objs, size=512):
    im = Image.open(img_path).convert("RGB").resize((size, size))
    d = ImageDraw.Draw(im)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 13)
    except OSError:
        font = ImageFont.load_default()
    for o in objs:
        x0, y0, x1, y1 = [v * size for v in o["box"]]
        c = COLOUR_RGB[o["color"]]
        d.rectangle([x0, y0, x1, y1], outline=c, width=3)
        d.rectangle([x0, y0, x1, y1], outline=(255, 255, 255) if o["color"] != "white" else (0, 0, 0), width=1)
        tag = f"{o['idx']}:{STATUS_MARK.get(o['failure'], o['failure'])}"
        tw = d.textlength(tag, font=font)
        d.rectangle([x0, max(0, y0 - 16), x0 + tw + 6, max(16, y0)], fill=(0, 0, 0))
        d.text((x0 + 3, max(0, y0 - 15)), tag, fill=(255, 255, 255), font=font)
    return im


def select(I):
    piv = {}
    for sid, g in I.groupby("sample_id"):
        a, b = g[g.method == "MIGC"].sort_values("idx"), g[g.method == "G0"].sort_values("idx")
        mx = g.max_iou_other.max()
        piv[sid] = {"level": g.level.iloc[0], "layout_id": g.layout_id.iloc[0], "max_iou": mx,
                    "overlap": "none" if mx <= 0 else "low" if mx < 0.1 else "mid" if mx < 0.3 else "high",
                    "g0_fail": b.failure.tolist(), "migc_fail": a.failure.tolist(), "g0_leak": int(b.crosstalk.sum()),
                    "g0_succ": b.official_success.mean(), "migc_succ": a.official_success.mean(),
                    "migc_dom": a.dominant_color.tolist(), "migc_color": a.color.tolist()}
    P = pd.DataFrame(piv).T
    sel = {}
    A = P[(P.migc_succ == 1) & (P.g0_leak >= 1)].copy()
    A["score"] = A.g0_leak - A.g0_succ
    sel["A"] = A.sort_values(["score", "level"], ascending=[False, False]).drop_duplicates("layout_id").head(3).index.tolist()
    B = P[(P.migc_succ == 1) & P.g0_fail.map(lambda f: any(x in ("missing", "mislocated") for x in f))]
    sel["B"] = B.sort_values("level").head(2).index.tolist()
    C = P[(P.g0_succ == 1) & (P.migc_succ < 1)]
    sel["C"] = C.sort_values("migc_succ").head(3).index.tolist()
    D = P[(P.g0_succ < 0.5) & (P.migc_succ < 0.5) & (P.max_iou >= 0.3)]
    sel["D"] = D.sort_values("max_iou", ascending=False).drop_duplicates("layout_id").head(2).index.tolist()
    E = P[P.apply(lambda r: any(f == "colour_wrong" and ((c, d) in {("yellow", "brown"), ("brown", "black"), ("brown", "yellow")})
                                for f, c, d in zip(r.migc_fail, r.migc_color, r.migc_dom)), axis=1)]
    sel["E"] = E.sort_values("g0_succ", ascending=False).head(2).index.tolist()
    return P, sel


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--results", type=Path, default=None)
    ap.add_argument("--gen", type=Path, default=None, help="default outputs/gen/test")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root.resolve()
    results = (args.results or root / "outputs/results/test_original").resolve()
    gen = (args.gen or root / "outputs/gen/test").resolve()
    out = (args.out or root / "outputs/final_analysis").resolve()
    (out / "failure_cases").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    apply_style()
    I = pd.read_csv(results / "instances.csv")
    P, sel = select(I)
    titles = {"A": "GLIGEN colour leakage, MIGC correct", "B": "GLIGEN misses / mislocates an object, MIGC correct",
              "C": "MIGC fails where GLIGEN succeeds", "D": "Both fail on a high-overlap layout",
              "E": "MIGC flagged colour-wrong with yellow↔brown / brown→black dominant colour"}
    rows = []
    case_no = 0
    for cat, sids in sel.items():
        if not sids:
            rows.append({"case": f"{cat}-none", "category": titles[cat], "sample_id": "", "note": "no example in this results set"})
            continue
        S, gap, head, foot = 420, 16, 26, 44
        sheet = Image.new("RGB", (2 * S + 3 * gap, 34 + len(sids) * (S + head + foot + gap)), (255, 255, 255))
        dr = ImageDraw.Draw(sheet)
        try:
            f_title = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 15)
            f_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 12)
        except OSError:
            f_title = f_small = ImageFont.load_default()
        dr.text((gap, 8), f"Category {cat}: {titles[cat]}", fill=(30, 30, 30), font=f_title)
        y = 34
        for r, sid in enumerate(sids):
            case_no += 1
            g = I[I.sample_id == sid]
            meta = None
            for c, m in enumerate(["G0", "MIGC"]):
                f = g[g.method == m].sort_values("idx")
                meta = json.load(open(gen / m / "meta" / f"{sid}.json"))
                objs = [{"idx": o.idx, "color": o.color, "failure": o.failure, "box": mo["box"]} for o, mo in zip(f.itertuples(), meta["objects"])]
                im = draw(gen / m / "images" / f"{sid}.png", objs, size=S)
                x = gap + c * (S + gap)
                dr.text((x, y + 4), f"{'GLIGEN' if m == 'G0' else 'MIGC'}: {int(f.official_success.sum())}/{len(f)} instances correct", fill=(30, 30, 30), font=f_title)
                sheet.paste(im, (x, y + head))
            cap = meta["caption"]
            dr.text((gap, y + head + S + 4), f"case {cat}{r+1} · {sid} · {g.level.iloc[0]} · overlap {P.loc[sid, 'overlap']}", fill=(90, 90, 90), font=f_small)
            dr.text((gap, y + head + S + 22), f"\u201c{cap}\u201d"[:120], fill=(90, 90, 90), font=f_small)
            y += S + head + foot + gap
            desc = []
            for m, label in [("G0", "GLIGEN"), ("MIGC", "MIGC")]:
                f = g[g.method == m].sort_values("idx")
                bad = [f"{o.desc} → {STATUS_MARK[o.failure]}" + (f" (mask mostly {o.dominant_color}{', leaked' if o.crosstalk == 1 else ''})" if o.failure == "colour_wrong" else "") for o in f.itertuples() if o.failure != "success"]
                desc.append(f"{label}: " + ("; ".join(bad) if bad else "all correct"))
            rows.append({"case": f"{cat}{r+1}", "category": titles[cat], "sample_id": sid, "level": g.level.iloc[0],
                         "overlap_bucket": P.loc[sid, "overlap"], "caption": cap, "auto_description": " | ".join(desc), "human_observation": ""})
        sheet.save(out / "figures" / f"cases_{cat}.png")
    pd.DataFrame(rows).to_csv(out / "failure_cases" / "cases.csv", index=False)
    counts = {k: len(v) for k, v in sel.items()}
    pool = {"A_pool": int(((P.migc_succ == 1) & (P.g0_leak >= 1)).sum()), "C_pool": int(((P.g0_succ == 1) & (P.migc_succ < 1)).sum()),
            "images_where_MIGC_all_correct": int((P.migc_succ == 1).sum()), "images_where_GLIGEN_all_correct": int((P.g0_succ == 1).sum())}
    (out / "failure_cases" / "selection_summary.json").write_text(json.dumps({"selected": counts, **pool}, indent=2) + "\n")
    print(json.dumps({"selected": counts, **pool}, indent=2))
    print(pd.DataFrame(rows)[["case", "sample_id", "auto_description"]].to_string())


if __name__ == "__main__":
    main()
