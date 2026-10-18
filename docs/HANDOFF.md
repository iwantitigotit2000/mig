# 交接记录（给新的 Claude 窗口）

最后更新：2026-10-16。用户要求：**用中文回复**；项目放在 **E 盘**；用户说“停”时不要再自动启动长任务；**任何运行或改文件之前都先问用户**。

**2026-10-16 决定：Stage B–C 全部改在学校 CCDS TC2 集群（A40）上运行，Stage B 在集群上从头跑。** 操作步骤见 `scripts/slurm/README_slurm.md`。集群相关脚本已写好，但**还没有在集群上运行或验证过**。

原始任务说明见 `docs/TASK_SPEC.md`（原文）。下面记录已完成的工作、关键决定和下一步。

---

## 1. 目录与环境

| 内容 | 位置 |
|---|---|
| 代码 + 结果（课程代码包） | `E:\transformer\mig_project` |
| 大文件（权重、官方仓库、COCO、训练检查点），不进代码包 | `E:\transformer\mig_assets` |
| Python 环境（conda 环境无法搬动，留在 D 盘） | `D:\condamini\envs\mig_gligen` / `mig_migc` / `mig_eval` |

所有路径集中在 `configs/paths.yaml`（`assets_root: E:/transformer/mig_assets`，可用环境变量 `MIG_ASSETS` 覆盖）。

三个环境（依赖冲突，所以分开）：
- `mig_gligen`：torch 2.1.2+cu121，transformers 4.19.2（GLIGEN 官方 ckpt 需要旧版 text encoder 键）。
- `mig_migc`：torch 2.1.2，diffusers 0.21.1，transformers 4.33.3，huggingface_hub 0.24.5（MIGC 官方要求）。
- `mig_eval`：torch 2.1.2，transformers 4.33.3，supervision 0.7.0，pycocotools，pyiqa 0.1.10（CLIP-IQA）。

硬件：本机是 RTX 4070 12 GB、32 GB 内存、Windows 11，**不是任务目标的 A40**（Stage A 在本机完成）。

集群（CCDS TC2，用户手册 `E:\transformer\CCDSGPU-TC2-Masters-UserGuide.pdf`）：
- QoS "normal"：CPU ≤ 10、GPU 1、主机内存 ≤ 30G、同时最多 2 个作业、单个作业最长 6 h、家目录 100 GB；分区 `MGPU-TC2`。
- 登录节点只能传文件、建环境、提交作业，不能跑代码。计算节点能否联网未知，所以权重全部预先上传，作业里打开 HF 离线模式。
- `scripts/slurm/*.sbatch` 共用 `scripts/slurm/env.sh`（加载 conda、设置 `MIG_ASSETS`/`PY_*`、离线模式、检查 GPU）。`scripts/lib.sh` 的解释器路径可以用 `PY_GLIGEN`/`PY_MIGC`/`PY_EVAL` 覆盖，默认仍是本机 D 盘路径。
- 环境：`scripts/cluster/mkenv_linux.sh` 按 `scripts/cluster/requirements_*.txt`（本机三个环境的 pip freeze）建环境。上传清单在 `scripts/cluster/upload_list.txt`（约 17 GB）。

## 2. 版本记录

| 项目 | 版本 |
|---|---|
| gligen/GLIGEN | f9dccb9c6cf48bad03c3666290a7dec8c5e58f3c |
| limuloo/MIGC | effd6cc2e6a9f7d2fe34409c300a5f037a87b9ec |
| LeyRio/MIG_Bench | 063ec4595f16632ab09b9ef71a7be00345ce1ead |
| GLIGEN 权重 | HF gligen/gligen-generation-text-box @ cc6e904（6.8 GB，含官方 opt 状态，iters=500000） |
| MIGC 权重 | MIGC_SD14.ckpt（Google Drive，229.5 MB） |
| SD1.4 | HF CompVis/stable-diffusion-v1-4 @ 133a221（MIGC 用） |
| CLIP ViT-L/14 | HF openai/clip-vit-large-patch14 @ 32bd642 |
| BERT | HF bert-base-uncased @ 86b5e09（GroundingDINO 用） |
| GroundingDINO | groundingdino_swint_ogc.pth（v0.1.0-alpha） |
| SAM | sam_vit_h_4b8939.pth |

两个模型都基于 SD1.4，基座一致。

## 3. 已完成并验证（真实运行）

**Stage A 已完成。**
- 适配器与官方脚本逐像素一致（像素差 0）：`scripts/check_gligen_official.py`、`scripts/check_migc_official.py`，结果在 `outputs/checks/`。
- GLIGEN 用 SDPA 替换 einsum 注意力：ε 最大误差 ≤ 7.9e-4（`outputs/checks/gligen_sdpa_check.json`），低于 fp16 舍入误差，正式生成用 SDPA。
- 样本清单 `manifests/`（`data_tools/build_cocomig_manifest.py`）：来自 MIGC 官方 `bench_file/mig_bench_anno.yaml`（800 布局）。按对象数 L2–L6 分层：val 30 布局、pilot 10 布局（val 的子集）、test 100 布局，互不重叠；每个布局两个种子 42、2025。标注了同类不同颜色和框重叠分桶（none / low<0.1 / mid<0.3 / high）。
- 训练数据（`data_tools/prepare_coco_train.py`）：COCO **train2014** 中 3000 张训练 + 100 张留出验证，与 val2014（COCO-MIG 的来源）无重叠。对象描述只用 COCO 类别名，**没有颜色监督**。
- pilot 10 布局 × 2 种子，两种方法都已生成和评测：`outputs/results/pilot/summary.md`。仅用于验证流程，不下结论。MIGC 实例成功率 0.59，GLIGEN 0.23；定位 IoU 都是 0.73；GLIGEN 主要败在颜色。
- 60 步短训练（`E:\transformer\mig_assets\runs\pilot_g1`、`pilot_g2`）：
  - G1 只训练 position_net，1,083,968 参数（0.10%），1.64 s/更新，5.8 GB。
  - G2 训练 position_net + fuser，209,102,240 参数（19.57%），1.74 s/更新，8.4 GB。
  - 两者的梯度检查和权重检查都通过：可训练参数都有梯度且都更新了，冻结参数无梯度、未改变。
  - 有效 batch = 8（2 × 累积 4），bf16 autocast。
- 生成速度：GLIGEN 约 5.4 s/张（峰值 9.4 GB），MIGC 约 6.5 s/张（峰值 7.8 GB）。

**Stage B 进行到一半，已被用户叫停。**
- 本机 `outputs/gen/test/G0/images` 已生成 152/200 张（RTX 4070），MIGC 测试集还没开始。
- 用户决定 Stage B 在集群上从头跑，所以这 152 张**不用于最终结果**。下载集群结果前，要先把本机这个目录移开。

## 4. 关键设计决定（写报告时要说明）

- **统一的配置**：512×512、50 步、CFG 7.5；负向提示统一为 MIGC 的默认值 `"worst quality, low quality, bad anatomy"`；初始噪声相同（`common.utils.initial_noise(seed)`）；生成用 fp16 autocast，权重保持 fp32。
- **无法统一的差异**：采样器不同（GLIGEN 官方用 PLMS，MIGC 官方用 Euler）。默认参数：GLIGEN alpha_type [0.3, 0, 0.7]；MIGC MIGCsteps=25、aug_phase_with_and=True。
- **官方 GLIGEN 代码的小处理**：
  - text encoder 的 `version` 参数指向本地 CLIP。
  - `restore_first_conv_from_SD` 已验证与 box-text 模型的输入卷积逐位相同，替换为 no-op，避免训练中替换模块。
- **G1/G2 的优化器从零开始**：没有加载 ckpt 里官方的 AdamW 状态，保证两组公平。没有重新初始化任何模块。
- **训练沿用官方设定**：ε-MSE、t~U[0,999]、10% grounding drop、prob_use_caption 0.5、max_boxes 30；AdamW wd 0，constant 学习率 + warmup 100。
- **评测**（`evaluation/mig_eval.py`）：
  - `official_*` 完全复现 MIG_Bench：最大 IoU ≥ 0.5；SAM 用 GT 框作提示；HSV 颜色比例 ≥ 0.2；官方 mIoU 对颜色错误记 0，对位置错误保留原 IoU。
  - 新增诊断：`loc_iou`（纯定位 IoU）、`o2o_*`（一对一匹配）、颜色串扰。
  - GroundingDINO 使用同文件里的 PyTorch 版可变形注意力（未编译 CUDA 算子），BERT 指向本地。
- **内存注意**：Windows 提交内存（commit）接近上限时会出现 MemoryError。生成脚本会自动重试并记录到 `errors.jsonl`；`scripts/lib.sh` 里有 GPU 锁，防止两条流水线同时运行。

## 5. 下一步（按顺序）

先在集群上上传资源并建环境（`scripts/slurm/README_slurm.md` 第 1、2 步），然后把下面每一步换成对应的 `sbatch scripts/slurm/<阶段>.sbatch` 提交（不要在本机运行 `bash scripts/...`）。每一步提交前都要先问用户。

1. 跑完 Stage B：`bash scripts/stageB.sh`，约 1 小时。之后写 `analysis/architecture_analysis.md`：用 `outputs/analysis/gligen_probe.json`、`migc_probe.json` 的真实维度，加两张结构图和方法差异表。
2. Stage C-1：`bash scripts/stageC_val_ratio.sh`。在 val 上比较 grounding 持续比例 r ∈ {0.3, 0.5, 0.8}，GLIGEN 和 MIGC 都做，约 40 分钟。
3. Stage C-2：`bash scripts/stageC_train.sh`。g1_lr5e-5、g2_lr5e-5、g2_lr1e-5 各 2000 步，每个约 1 小时，可断点恢复。
4. Stage C-3：`RATIO=<val 选出的值> G2_LR=<val 选出的值> bash scripts/stageC_eval_ft.sh`。在 val 上比较学习率和早期/后期检查点，然后用 test 跑 G1/G2。
5. Stage D：汇总表（`analysis/aggregate.py`）、训练曲线、≥6 组配对图、失败案例、README 复现命令、报告初稿。

**未运行**的内容一律在报告里标注为“未运行”或“估算”。
