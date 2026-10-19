"""Verify that report/report.tex carries no stale result numbers.

1. report/tables/{main_table,ft_table,numbers}.tex must be byte-identical to a fresh regeneration
   from outputs/final_analysis (analysis/make_report_tables.py) -> every table cell and macro is current.
2. The hand-written body of report.tex (excluding \\input files, comments and the bibliography) must not
   contain result-like decimals outside an explicit allow-list of fixed protocol constants; any other
   number must come through a macro from tables/numbers.tex.
Exit code 1 on any violation. Run before compiling the PDF.
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from analysis.make_report_tables import build, OUT  # noqa: E402

# fixed protocol constants / dataset facts that are legitimately typed in the text
ALLOW = {"0.25", "0.5", "0.2", "7.5", "1.4", "0.1", "0.3", "0.7", "0.051", "0.02", "0.06", "0.9", "0.3", "1.9",
         "0.49", "0.85", "0.05"}   # figure widths etc. inside \includegraphics are stripped anyway
ALLOW_CONTEXT = [r"\\includegraphics\[[^\]]*\]", r"\\usepackage\[[^\]]*\]", r"7\.9\{\\times\}10\^\{-4\}",
                 r"\\texttt\{alpha\\_type\}=\[0\.3,0,0\.7\]", r"\\vspace\{[^}]*\}", r"skip=\d+pt"]


def main():
    bad = []
    fresh = build()
    for name, text in fresh.items():
        disk = (OUT / name)
        if not disk.exists() or disk.read_text(encoding="utf-8") != text:
            bad.append(f"report/tables/{name} is stale or missing -> run analysis/make_report_tables.py")
    tex = (ROOT / "report/report.tex").read_text(encoding="utf-8")
    body = tex[tex.index("\\begin{document}"):tex.index("\\begin{thebibliography}")]
    body = "\n".join(re.split(r"(?<!\\)%", l)[0] for l in body.splitlines())
    for pat in ALLOW_CONTEXT:
        body = re.sub(pat, " ", body)
    for m in re.finditer(r"(?<![A-Za-z\\])[+\-]?\d+\.\d+", body):
        num = m.group(0).lstrip("+-")
        if num in ALLOW:
            continue
        line_no = body[:m.start()].count("\n") + 1
        ctx = body[max(0, m.start() - 40):m.end() + 40].replace("\n", " ")
        bad.append(f"hand-typed number {m.group(0)} in body (line ~{line_no}): ...{ctx}...")
    if bad:
        print("REPORT CHECK FAILED:\n  " + "\n  ".join(bad))
        sys.exit(1)
    print("report check passed: generated tables are current and no stray result numbers in the body")


if __name__ == "__main__":
    main()
