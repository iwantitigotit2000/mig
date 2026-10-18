| metric | G0 | MIGC |
|---|---|---|
| n_images | 200 | 200 |
| official_instance_success | 0.321 [0.286, 0.361] | 0.665 [0.620, 0.710] |
| official_image_success | 0.065 [0.025, 0.110] | 0.280 [0.210, 0.360] |
| official_miou | 0.288 [0.259, 0.319] | 0.572 [0.534, 0.609] |
| loc_iou | 0.690 [0.658, 0.720] | 0.741 [0.714, 0.765] |
| presence_any | 0.968 | 0.991 |
| o2o_presence | 0.917 [0.889, 0.942] | 0.932 [0.902, 0.958] |
| o2o_instance_success | 0.312 [0.276, 0.353] | 0.657 [0.612, 0.704] |
| fail_missing | 0.033 | 0.009 |
| fail_mislocated | 0.159 | 0.119 |
| fail_colour_wrong | 0.487 | 0.207 |
| crosstalk_share_of_colour_wrong | 0.572 | 0.440 |
| eval_errors | 0 | 0 |
| clipiqa | 0.689 [0.672, 0.707] | 0.707 [0.686, 0.726] |
| sec_per_image | 5.748 | 6.517 |
| peak_mem_allocated_gb_max | 9.361 | 7.829 |

Paired differences (same samples), 95% layout-bootstrap CI:

| comparison | metric | diff | CI95 |
|---|---|---|---|
| MIGC-G0 | official_instance_success | +0.344 | [+0.298, +0.393] |
| MIGC-G0 | official_image_success | +0.215 | [+0.145, +0.285] |
| MIGC-G0 | official_miou | +0.284 | [+0.244, +0.324] |
| MIGC-G0 | loc_iou | +0.051 | [+0.024, +0.080] |
| MIGC-G0 | o2o_instance_success | +0.345 | [+0.298, +0.395] |
| MIGC-G0 | o2o_presence | +0.015 | [-0.012, +0.041] |
| MIGC-G0 | clipiqa | +0.018 | [+0.002, +0.035] |