| metric | G0 | MIGC |
|---|---|---|
| n_images | 20 | 20 |
| official_instance_success | 0.225 [0.149, 0.314] | 0.588 [0.393, 0.768] |
| official_image_success | 0.050 [0.000, 0.150] | 0.250 [0.049, 0.500] |
| official_miou | 0.215 [0.144, 0.297] | 0.535 [0.393, 0.667] |
| loc_iou | 0.729 [0.692, 0.763] | 0.733 [0.676, 0.790] |
| presence_any | 0.975 | 0.988 |
| o2o_presence | 0.925 [0.858, 1.000] | 0.863 [0.763, 0.952] |
| o2o_instance_success | 0.237 [0.156, 0.338] | 0.588 [0.389, 0.757] |
| fail_missing | 0.025 | 0.013 |
| fail_mislocated | 0.125 | 0.150 |
| fail_colour_wrong | 0.625 | 0.250 |
| crosstalk_share_of_colour_wrong | 0.680 | 0.500 |
| eval_errors | 0 | 0 |
| clipiqa | 0.703 [0.650, 0.752] | 0.665 [0.599, 0.727] |
| sec_per_image | 5.391 | 7.115 |
| peak_mem_allocated_gb_max | 9.361 | 7.833 |

Paired differences (same samples), 95% layout-bootstrap CI:

| comparison | metric | diff | CI95 |
|---|---|---|---|
| MIGC-G0 | official_instance_success | +0.363 | [+0.167, +0.547] |
| MIGC-G0 | official_image_success | +0.200 | [+0.000, +0.500] |
| MIGC-G0 | official_miou | +0.320 | [+0.163, +0.462] |
| MIGC-G0 | loc_iou | +0.004 | [-0.072, +0.071] |
| MIGC-G0 | o2o_instance_success | +0.350 | [+0.141, +0.542] |
| MIGC-G0 | o2o_presence | -0.062 | [-0.167, +0.012] |
| MIGC-G0 | clipiqa | -0.038 | [-0.098, +0.035] |