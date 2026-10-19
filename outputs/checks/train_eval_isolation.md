# Train / evaluation isolation check (independent record)

Data version `v1.0-locked-2026-10`; sha256 of every manifest in `manifests/MANIFEST_VERSIONS.json`.

| Check | Result |
|---|---|
| val / test layouts disjoint | yes (30 / 100 layouts) |
| pilot ⊂ val | yes (10 layouts) |
| all 3100 training images are COCO train2014 files | yes |
| evaluation layouts copied from a training image (mean matched IoU ≥ 0.9) | 0 of 130 |
| similar but distinct layouts (0.7 ≤ mean IoU < 0.9; all 2–3-object layouts of common categories) | 4 of 130 |
| highest / median best-match mean IoU over all evaluation layouts | 0.832 / 0.239 |

COCO-MIG layouts are derived from COCO val2014 and the training subset is drawn from train2014, which are disjoint image sets by construction. Because the bench file does not record COCO image IDs, an ID-level exclusion list cannot be produced; the layout-level duplicate search above is the strongest check possible from the committed manifests and is reported instead.

**Scope.** This layout-level search does NOT replace a COCO image-ID check; such a check is impossible from the bench file, and the report must say so rather than claim ID-level exclusion.
