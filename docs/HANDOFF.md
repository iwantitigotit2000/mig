# 交接记录（给新的 Claude 窗口）

最后更新：2026-10-20。项目根目录以当前仓库为准；大文件位置通过 `MIG_ASSETS` 配置，不在交接文档中记录个人电脑的绝对路径。

> **当前权威状态（2026-10-20）**：Stage D 的隔离检查、统计、架构分析、17 张图和 8 页报告已完成；CPU 分析已在 Apple Silicon Mac 上独立复现。C-3 微调测试集仍待 GPU 补跑。仓库内测试结果的 metadata 指向本地 RTX 4070，属于 `PROVISIONAL`；早期记录中的 A40 结果没有原始文件，不能视为已验证。下面第 1–4 节保留 2026-10-16 的历史环境与决策记录，当前待办以第 5 节及根目录 `MERGE_READY.md` 为准。

**2026-10-16 决定：Stage B–C 全部改在学校 CCDS TC2 集群（A40）上运行，Stage B 在集群上从头跑。** 操作步骤见 `scripts/slurm/README_slurm.md`。集群相关脚本已写好，但**还没有在集群上运行或验证过**。

原始任务说明见 `docs/TASK_SPEC.md`（原文）。下面记录已完成的工作、关键决定和下一步。

---

## 1. 目录与环境

| 内容 | 位置 |
|---|---|
| 代码 + 结果（课程代码包） | 仓库根目录 |
| 大文件（权重、官方仓库、COCO、训练检查点），不进代码包 | `${MIG_ASSETS}` |
| Python 环境 | conda 环境 `mig_gligen` / `mig_migc` / `mig_eval` |

所有路径集中在 `configs/paths.yaml`；本机资源根目录应通过环境变量 `MIG_ASSETS` 覆盖，避免提交个人绝对路径。

三个环境（依赖冲突，所以分开）：
- `mig_gligen`：torch 2.1.2+cu121，transformers 4.19.2（GLIGEN 官方 ckpt 需要旧版 text encoder 键）。
- `mig_migc`：torch 2.1.2，diffusers 0.21.1，transformers 4.33.3，huggingface_hub 0.24.5（MIGC 官方要求）。
- `mig_eval`：torch 2.1.2，transformers 4.33.3，supervision 0.7.0，pycocotools，pyiqa 0.1.10（CLIP-IQA）。

硬件：本机是 RTX 4070 12 GB、32 GB 内存、Windows 11，**不是任务目标的 A40**（Stage A 在本机完成）。

集群（CCDS TC2；用户手册保存在组员本地，不纳入仓库）：
- QoS "normal"：CPU ≤ 10、GPU 1、主机内存 ≤ 30G、同时最多 2 个作业、单个作业最长 6 h、家目录 100 GB；分区 `MGPU-TC2`。
- 登录节点只能传文件、建环境、提交作业，不能跑代码。计算节点能否联网未知，所以权重全部预先上传，作业里打开 HF 离线模式。
- `scripts/slurm/*.sbatch` 共用 `scripts/slurm/env.sh`（加载 conda、设置 `MIG_ASSETS`/`PY_*`、离线模式、检查 GPU）。`scripts/lib.sh` 的解释器路径可以用 `PY_GLIGEN`/`PY_MIGC`/`PY_EVAL` 覆盖；使用时应显式设置，避免依赖旧版配置中的本机绝对路径。
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
- 60 步短训练（`${MIG_ASSETS}/runs/pilot_g1`、`${MIG_ASSETS}/runs/pilot_g2`）：
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

## 5. 当前下一步（按优先级）

1. **先完成提交信息**：在 `report/report.tex` 替换 Group XX 和姓名；全组按实际贡献确认 `report/contribution.md`。
2. **有 GPU 时补跑 C-3 测试集**：仅评测验证集预先选出的 G1 与 G2 5e-5，避免再用测试集调参。将结果复制回仓库后完整运行 `bash scripts/stageD_analysis.sh`。
3. **若收到 A40 原始输出**：先补 `provenance.json`，选择一个 canonical run，再重建全部统计、图和报告；不要把本地 RTX 4070 与 A40 数字混在一张主表中。
4. **决定 sampler 处理**：当前 GLIGEN=PLMS、MIGC=Euler，并已作为限制披露。只有课程明确要求统一时才用共同支持的 DDIM 重跑。
5. **最终 CPU 检查**：运行 `bash scripts/stageD_analysis.sh 2>&1 | tee stageD_run.log`。Mac 复现记录在 `docs/MAC_CPU_REPRODUCTION.md`。

**已完成但有证据边界的项目**：

- `position_net`、学习率和 checkpoint：验证集分析完成，测试集尚未运行。
- MIGC 的 EA / LA / SAC 与架构比较：文档和图完成；没有组件消融，不能作因果表述。
- 测试 image ID：上游 bench 不提供实际 COCO image ID；已完成来源隔离、layout/sample 标识导出与布局级近重复检查。
- 失败模式、趋势图和报告：已完成；失败案例 CSV 的人工观察栏可由组员在最终提交前补充。

**未运行**的内容一律在报告里标注为“未运行”“待补”或“未验证”，不得根据日志推断为已完成。
