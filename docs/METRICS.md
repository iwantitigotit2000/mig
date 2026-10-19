# 指标名称对照表（报告用名 ↔ 评测内部列名）

评测器 `evaluation/mig_eval.py` 和汇总脚本 `analysis/aggregate.py` 的列名**保持不变**（便于追溯到官方 MIG-Bench 规则），
报告与图表统一使用左侧名称。映射代码在 `analysis/metric_names.py`，所有新图表脚本都从那里取名字。

## 每个目标实例（instance）一行

| 报告用名 | 内部列 | 定义 | 方向 |
|---|---|---|---|
| **Object Recall** | `o2o_present` | 对该类别做一对一（匈牙利）匹配后，该实例分到了一个检测框，即"对象被画出来了"。比 `present_any`（同类任意检测存在）严格：两个同类目标不能共用一个检测框。 | ↑ |
| **Position Success** | `official_pos` | MIG-Bench 位置规则：GroundingDINO 的同类检测框与 GT 框的最大 IoU ≥ 0.5（不要求一对一）。 | ↑ |
| **Attribute Success** = Instance Success (MIG-Bench) | `official_success` | 位置正确 **且** 用 GT 框提示 SAM 得到的 mask 内，目标颜色（HSV 官方色表）像素占比 ≥ 0.2。这是 MIG-Bench 的官方实例分数。 | ↑ |
| Attribute Success \| Position | `official_color` 在 `official_pos==1` 的子集上 | 位置对了的前提下颜色对的比例，用于把"颜色绑定"从"定位"中剥离出来。 | ↑ |
| **Official mIoU** | `official_miou` | MIG-Bench mIoU：匹配 IoU，颜色错误时记 0。 | ↑ |
| **Localisation IoU** | `loc_iou` | 与任意同类检测框的最大 IoU，不看颜色（新增诊断）。 | ↑ |
| **Color Leakage Rate** | `crosstalk` | mask 内占比最高的颜色不是目标色、而是**画面里另一个目标的颜色**且占比 ≥ 0.2。分母为全部实例。 | ↓ |
| Leakage share of colour errors | `crosstalk` 在 `failure=="colour_wrong"` 子集上 | 颜色错误里有多大比例是"串色"，其余是无关颜色/灰白等。 | — |
| Missing Rate | `failure=="missing"` | 全图没有该类别的任何检测。 | ↓ |
| Mislocation Rate | `failure=="mislocated"` | 有同类检测但最大 IoU < 0.5。 | ↓ |
| Colour Error Rate | `failure=="colour_wrong"` | 位置对但颜色占比 < 0.2。 | ↓ |

三类失败互斥，与 Attribute Success 四者之和为 1。

## 每张图（image）一行

| 报告用名 | 内部列 | 定义 |
|---|---|---|
| **Image Success** | `official_image_success` | 图中所有实例都 Attribute Success（MIG-Bench）。 |
| **CLIP-IQA** | `clipiqa` | 无参考质量分（pyiqa `clipiqa`），越高越好。 |

## 置信区间与配对

- 95% 区间：按 **布局（layout_id）** 成组的百分位 bootstrap（同一布局的全部对象和两个随机种子一起重采样），5000 次，种子 2026（`analysis/final_statistics.py`）。
- 方法间比较一律用**配对差值**（相同 sample_id + idx），再对差值做布局 bootstrap。
- 区间未做多重比较校正；验证集（30 布局）上的对比只作探索性结论。

## 困难条件分组（来自输入清单，不依赖任何生成结果）

| 分组 | 来源列 | 取值 |
|---|---|---|
| 对象数 | `level` | L2–L6 |
| 框重叠 | `overlap_bucket` | none / low (<0.1) / mid (<0.3) / high (≥0.3)，按与其他框的最大 IoU |
| 同类异色 | `same_cat_diff_color` | True / False |
| 目标大小（新增） | `manifests/cocomig_difficulty.csv` 的 `relative_size_q` | Q1（最小 25%）… Q4；另给 COCO 规则 `size_bucket_coco` |

注意：COCO-MIG 的 400 个测试目标在 512×512 下最小边长 66 px，**没有 COCO 定义的 small（< 32² px）目标**，54 个属于 medium，其余 large。
因此"小目标"分析使用相对四分位（Q1：归一化面积 < 0.051，约边长 < 115 px）。
