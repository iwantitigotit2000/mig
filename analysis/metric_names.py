"""Single source of truth for report metric names.

Internal column names (instances.csv / images.csv, produced by analysis/aggregate.py) are kept
unchanged so every number stays traceable to the evaluator; the report uses the names below.
See docs/METRICS.md for definitions.
"""

# report name -> (table, internal column, higher_is_better, short definition)
INSTANCE_METRICS = {
    "Object Recall":            ("instances", "o2o_present",      True,  "one-to-one matched detection of the target category exists (object was drawn)"),
    "Position Success":         ("instances", "official_pos",     True,  "MIG-Bench position rule: max IoU(GT box, any same-category detection) >= 0.5"),
    "Attribute Success":        ("instances", "official_success", True,  "MIG-Bench instance success: position OK and HSV colour coverage >= 0.2 in the SAM mask"),
    "Instance Success (MIG-Bench)": ("instances", "official_success", True, "alias of Attribute Success; official per-instance score"),
    "Official mIoU":            ("instances", "official_miou",    True,  "MIG-Bench mIoU: matched IoU, zeroed when the colour is wrong"),
    "Localisation IoU":         ("instances", "loc_iou",          True,  "max IoU with any same-category detection, colour ignored"),
    "Color Leakage Rate":       ("instances", "crosstalk",        False, "instance whose dominant mask colour is another instance's target colour (>= 0.2); fraction of all instances"),
    "Missing Rate":             ("instances", "fail_missing",     False, "no detection of the category anywhere in the image"),
    "Mislocation Rate":         ("instances", "fail_mislocated",  False, "detected somewhere but max IoU < 0.5"),
    "Colour Error Rate":        ("instances", "fail_colour_wrong", False, "position OK but colour coverage < 0.2"),
}
IMAGE_METRICS = {
    "Image Success":            ("images", "official_image_success", True, "all instances of the image succeed (MIG-Bench)"),
    "CLIP-IQA":                 ("images", "clipiqa",                True, "no-reference quality score (pyiqa clipiqa), higher is better"),
}
# conditional metrics derived from the same columns
CONDITIONAL = {
    "Attribute Success | Position": ("official_color", "official_pos == 1", "colour correct among position-correct instances"),
    "Leakage share of colour errors": ("crosstalk", "failure == 'colour_wrong'", "fraction of colour errors that are leakage from another instance"),
}

FAILURE_COLUMNS = {"fail_missing": "missing", "fail_mislocated": "mislocated", "fail_colour_wrong": "colour_wrong"}

METHOD_LABELS = {
    "G0": "GLIGEN (official)", "MIGC": "MIGC (official)",
    "g1_lr5e-5_s000500": "GLIGEN G1 (pos-net, lr 5e-5, 500 steps)",
    "g1_lr5e-5_s002000": "GLIGEN G1 (pos-net, lr 5e-5, 2000 steps)",
    "g2_lr5e-5_s000500": "GLIGEN G2 (pos-net+fuser, lr 5e-5, 500 steps)",
    "g2_lr5e-5_s002000": "GLIGEN G2 (pos-net+fuser, lr 5e-5, 2000 steps)",
    "g2_lr1e-5_s000500": "GLIGEN G2 (pos-net+fuser, lr 1e-5, 500 steps)",
    "g2_lr1e-5_s002000": "GLIGEN G2 (pos-net+fuser, lr 1e-5, 2000 steps)",
}


def add_failure_columns(instances):
    """Add 0/1 failure indicator columns used by the report names."""
    df = instances.copy()
    for col, label in FAILURE_COLUMNS.items():
        df[col] = df.failure.eq(label).astype(float)
    df["crosstalk"] = df.crosstalk.fillna(0).astype(float)
    return df
