"""Shared figure style: one fixed colour per method across every figure (never cycled)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = {
    "G0": "#2a78d6",          # GLIGEN official  (blue)
    "GLIGEN": "#2a78d6",
    "MIGC": "#eb6834",        # MIGC official    (orange)
    "g1_lr5e-5": "#1baf7a",   # G1 position_net  (aqua)
    "g2_lr5e-5": "#eda100",   # G2 lr 5e-5       (yellow)
    "g2_lr1e-5": "#e87ba4",   # G2 lr 1e-5       (magenta)
}
LINESTYLE_BY_STEP = {"s000500": "--", "s002000": "-"}
SHORT = {"G0": "GLIGEN", "MIGC": "MIGC", "g1_lr5e-5": "G1 lr5e-5", "g2_lr5e-5": "G2 lr5e-5", "g2_lr1e-5": "G2 lr1e-5"}
TEXT, MUTED, GRID = "#2b2b2b", "#6b6b6b", "#dcdcdc"


def color_of(method):
    for k, v in COLORS.items():
        if method == k or method.startswith(k):
            return v
    return "#888888"


def label_of(method):
    for k, v in SHORT.items():
        if method == k:
            return v
        if method.startswith(k + "_s"):
            return f"{v} @{int(method.split('_s')[-1])}"
    return method


def apply_style():
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "legend.fontsize": 8,
        "axes.edgecolor": GRID, "axes.labelcolor": TEXT, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False,
        "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight", "lines.linewidth": 1.8,
    })
