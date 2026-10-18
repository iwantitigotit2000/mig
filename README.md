# GLIGEN 与 MIGC 的多对象可控图像生成比较（课程项目）

> **新的 Claude 窗口请先读 `CLAUDE.md` → `docs/HANDOFF.md` → `docs/TASK_SPEC.md`。**

**当前状态（2026-10-18）**：Stage A、B、C-1、C-2 已完成；Stage C-3（微调模型在验证集上的评测，作业 43571）正在集群上运行；Stage D（图表、失败分析、报告）未开始。
本文第 4 节列出已完成的内容，第 5 节列出未完成的内容（按组员分工标注）。

---

## 1. 目录

```
configs/      paths.yaml（所有资源路径）、gen.yaml（统一生成参数）、train/*.yaml（G1/G2 训练）
adapters/     gligen_adapter.py、migc_adapter.py（调用官方代码的薄包装）
common/       工具函数、通用生成循环（跳过已完成样本、记录耗时和显存）
data_tools/   build_cocomig_manifest.py（COCO-MIG 划分）、prepare_coco_train.py（COCO train2014 子集）
training/     train_gligen.py（G1/G2 局部微调，可断点恢复）、coco_grounding_dataset.py
evaluation/   mig_eval.py（GroundingDINO + SAM + HSV，官方规则 + 新增诊断）、quality.py（CLIP-IQA）
analysis/     aggregate.py（汇总、分组、bootstrap 置信区间）、probe_*.py（真实前向的张量维度）
scripts/      gen_*.py、check_*.py、stage*.sh、slurm/（集群作业）、cluster/（集群环境与上传清单）
manifests/    固定样本清单（pilot / val / test / 训练集）
outputs/      生成图片、评测缓存、结果表、日志、检查结果
docs/         TASK_SPEC.md（原始任务）、HANDOFF.md（交接记录）
```

大文件（权重、官方仓库、COCO、训练检查点）在 `E:\transformer\mig_assets`（集群上是 `~/mig_assets`），不属于代码包。

## 2. 环境

三个 conda 环境（依赖冲突，所以分开），都是 Python 3.10 + torch 2.1.2+cu121：

| 环境 | 用途 | 关键版本 |
|---|---|---|
| `mig_gligen` | GLIGEN 生成和微调 | transformers 4.19.2，omegaconf 2.1.1，openai-clip，kornia 0.6.12 |
| `mig_migc` | MIGC 生成 | diffusers 0.21.1，transformers 4.33.3，huggingface_hub 0.24.5，accelerate 0.23.0 |
| `mig_eval` | 评测和汇总 | transformers 4.33.3，supervision 0.7.0，pycocotools，pyiqa 0.1.10，setuptools 69.5.1 |

- 本机：`E:\transformer\mig_assets\mkenv.sh`。
- 集群：`scripts/cluster/mkenv_linux.sh`，版本取自本机的 `pip freeze`（`scripts/cluster/requirements_*.txt`）。
- 注意：`pip freeze` 不会列出 setuptools，`mig_eval` 需要另外装 `setuptools==69.5.1`，否则 CLIP-IQA 会报 `No module named 'pkg_resources'`。

官方代码和权重版本：

| 项目 | 版本 |
|---|---|
| gligen/GLIGEN | f9dccb9 |
| limuloo/MIGC | effd6cc |
| LeyRio/MIG_Bench | 063ec45 |
| GLIGEN 权重 | HF gligen/gligen-generation-text-box @ cc6e904 |
| MIGC 权重 | MIGC_SD14.ckpt |
| 基座 | Stable Diffusion v1.4（两个模型相同） |

## 3. 运行方式

- **Stage A** 在本机（RTX 4070 12 GB）上运行。
- **Stage B–C** 在学校 CCDS TC2 集群（A40 46 GB）上运行。完整步骤和命令见 `scripts/slurm/README_slurm.md`。

```bash
# 本机（Git Bash，项目根目录）
bash scripts/stageA.sh          # pilot 生成 + 评测
bash scripts/stageA_train.sh    # 60 步短训练

# 集群（~/mig_project 下；每次提交前一个作业结束后再提交下一个）
sbatch scripts/slurm/stageB.sbatch                 # 测试集原模型生成 + 评测 + 结构探测
sbatch scripts/slurm/stageC_val_ratio.sbatch       # 验证集上比较 grounding 持续比例
sbatch scripts/slurm/stageC_train.sbatch           # G1 / G2 各 2000 步
sbatch --export=ALL,RATIO=0.5 scripts/slurm/stageC_eval_ft.sbatch              # 验证集：学习率与检查点
sbatch --export=ALL,RATIO=0.5,G2_LR=<lr> scripts/slurm/stageC_eval_ft.sbatch   # 测试集：G1 / G2
```

所有阶段都能中断后重跑：生成和评测会跳过已完成的样本；训练从 `ckpt_latest.pt` 恢复。

---

## 4. 已完成的内容

### 4.1 数据

- **测试集与验证集**（`data_tools/build_cocomig_manifest.py`）
  - 来源：MIGC 官方仓库的 `bench_file/mig_bench_anno.yaml`，共 800 个 COCO-MIG 布局。
  - 按对象数 L2–L6 分层抽样：test 100 个布局、val 30 个、pilot 10 个（pilot 是 val 的子集）。test 和 val 互不重叠。
  - 每个布局用两个种子：42 和 2025。
  - 每个对象都标注了同类异色（`same_cat_diff_color`），以及与其他框的最大重叠程度（`overlap_bucket`：none / low < 0.1 / mid < 0.3 / high ≥ 0.3）。
  - 统计信息见 `manifests/cocomig_split_summary.json`。
- **训练集**（`data_tools/prepare_coco_train.py`）
  - 从 COCO train2014 中选 3000 张训练图（`manifests/coco_train.jsonl`）和 100 张训练期验证图（`coco_trainval.jsonl`）。
  - 每条记录包含图片、5 句整体描述，以及每个对象的类别短语和框坐标。
  - 对象短语只用 COCO 类别名，**没有颜色监督**。
  - 与 COCO-MIG 的来源 val2014 隔离：脚本检查了所选文件名，与 val2014 的重叠数为 0。统计见 `manifests/coco_train_summary.json`。

### 4.2 原模型（Stage A、B）

- **与官方实现对齐**：两个适配器的输出与官方推理脚本**逐像素一致**（像素差为 0），检查结果在 `outputs/checks/`。GLIGEN 的注意力换成了 SDPA，噪声预测的最大误差 ≤ 7.9e-4，低于 fp16 舍入误差。
- **统一的生成配置**（`configs/gen.yaml`）：
  - 已统一：512×512、50 步、CFG 7.5、相同的负向提示、相同的初始噪声（由种子决定）。
  - **无法统一**：采样器。GLIGEN 用官方的 PLMS，MIGC 用官方的 Euler。
- **每次生成都保存**：图片、输入、检查点、完整配置、耗时和峰值显存（`outputs/gen/*/*/meta/`）。
- **测试集结果**（集群 A40，100 个布局 × 2 个种子 = 每个方法 200 张；方括号内为按布局 bootstrap 的 95% 置信区间）：

| 指标 | GLIGEN（G0） | MIGC | MIGC − G0 |
|---|---|---|---|
| 官方实例成功率（位置和颜色都对） | 0.319 | **0.669** | +0.350 [+0.303, +0.398] |
| 官方整图成功率 | 0.065 | **0.280** | +0.215 [+0.145, +0.285] |
| 官方 mIoU（颜色错误记 0） | 0.286 | **0.574** | +0.287 [+0.247, +0.326] |
| 纯定位 IoU | 0.691 | 0.741 | +0.050 [+0.023, +0.080] |
| 一对一匹配的对象出现率 | 0.917 | 0.932 | +0.015 [−0.013, +0.042]（不显著） |
| 失败：对象缺失 / 位置错误 / 颜色错误 | 0.031 / 0.160 / **0.490** | 0.009 / 0.116 / 0.206 | |
| 颜色错误中属于串扰的比例 | 0.577 | 0.442 | |
| 评测异常数 | 0 | 0 | |
| 每张耗时 / 峰值显存 | 5.2 s / 9.4 GB | 5.7 s / 7.8 GB | |
| CLIP-IQA | 待补算 | 待补算 | |

**观察**：两种方法画出对象的比例相近，MIGC 的定位略好一些。差距主要来自颜色绑定：GLIGEN 约一半的对象颜色错误，其中超过一半是串扰（用了画面中另一个对象的颜色）。

### 4.3 grounding 控制持续比例（Stage C-1，仅用验证集）

比例 r 的含义：GLIGEN 的 alpha_type 为 [r, 0, 1−r]；MIGC 的 MIGCsteps 为 r × 50。验证集 30 个布局 × 2 个种子。

| 指标 | G0 r=0.3 | G0 r=0.5 | G0 r=0.8 | MIGC r=0.3 | MIGC r=0.5 | MIGC r=0.8 |
|---|---|---|---|---|---|---|
| 官方实例成功率 | 0.258 | **0.279** | 0.275 | 0.613 | 0.662 | **0.675** |
| 纯定位 IoU | 0.721 | 0.746 | **0.748** | 0.693 | 0.755 | **0.759** |
| 颜色错误比例 | 0.596 | 0.621 | 0.625 | 0.221 | 0.229 | 0.221 |
| 每张耗时 | 5.3 s | 5.2 s | 5.2 s | 5.3 s | 5.6 s | 6.4 s |

- **GLIGEN**：r 从 0.3 加到 0.5，定位 IoU 提高 +0.025 [+0.011, +0.041]；但实例成功率只提高 +0.021 [−0.005, +0.052]，**不显著**，因为颜色错误略有增加。
- **选定 r = 0.5** 用于后续的微调评测。依据是验证集上实例成功率的点估计最高；它和 0.8 的差距在噪声范围内。测试集没有参与选择。

### 4.4 GLIGEN 局部微调（Stage C-2）

- **三组设置**：G1 只训练 position_net；G2 训练 position_net 和 fuser。三组都从同一个官方检查点开始，没有重新初始化任何模块。
- **三组相同的部分**：同一份训练数据、2000 步、有效 batch 8（2 × 梯度累积 4）、AdamW（weight decay 0）、warmup 100 后保持恒定学习率、随机种子 2024、bf16 autocast。配置在 `configs/train/`。
- **训练目标**：沿用官方设定，ε-MSE，t ~ U[0, 999]，10% 概率丢弃 grounding 输入，prob_use_caption 0.5。
- **优化器从零开始**：不加载检查点里官方的 AdamW 状态，保证两组公平。

| 运行 | 可训练参数 | 学习率 | 每步耗时（A40） | 峰值显存 |
|---|---|---|---|---|
| g1_lr5e-5 | 1,083,968（0.10%） | 5e-5 | 1.32 s | 5.8 GB |
| g2_lr5e-5 | 209,102,240（19.57%） | 5e-5 | 1.41 s | 8.4 GB |
| g2_lr1e-5 | 209,102,240（19.57%） | 1e-5 | 1.41 s | 8.4 GB |

- **训练产出**：训练日志、训练期验证损失（t ∈ {100, 300, 500, 800}）、验证样例图、500 步和 2000 步的检查点、可训练参数清单（`trainable_params.json`）。都在 `mig_assets/runs/<运行名>/`。
- **权重检查（2000 步）全部通过**：冻结参数没有任何改动；所有可训练参数都更新了。权重变化量（L2）：

| 运行 | position_net | fuser |
|---|---|---|
| G1 lr 5e-5 | 2.72 | — |
| G2 lr 5e-5 | 1.73 | 30.3 |
| G2 lr 1e-5 | 0.49 | 6.9 |

### 4.5 自动评测（`evaluation/mig_eval.py`、`quality.py`）

- **官方规则**（完全复现 MIG_Bench，指标名以 `official_*` 开头）：
  - GroundingDINO 检测对象，与目标框的最大 IoU ≥ 0.5 记为位置匹配；
  - SAM 以 GT 框为提示分割出对象；
  - 在 mask 内统计 HSV 目标颜色的比例，≥ 0.2 记为颜色正确；
  - 官方 mIoU 对颜色错误的对象记 0。
- **新增诊断**（与官方指标分开命名）：
  - `loc_iou`：纯定位 IoU，不考虑颜色；
  - `o2o_*`：一对一匹配；
  - 失败类型：缺失 / 位置错误 / 颜色错误；
  - 颜色串扰。
- **按组汇总**：按对象数（L2–L6）和框重叠程度分组；所有指标都给出按布局 bootstrap 的 95% 置信区间；同一批样本之间做配对差值。
- **保存与容错**：检测框、mask、颜色比例和失败状态都会缓存；评测异常会被计数并记录，不会被静默删除。

---

## 5. 未完成的内容

标记说明：❌ 未做；🟡 部分完成；⏳ 正在运行。

### 总览（按负责人）

| 未完成的项目 | 状态 | 负责人 | 对应分工 |
|---|---|---|---|
| 导出测试 image ID | ❌ | **C** | 首先向 A 提供测试集 image ID |
| 按 ID 排除重叠（目前只做了来源隔离） | 🟡 | **A**（用 C 的清单） | 根据成员 C 提供的测试 image ID 排除重叠样本 |
| 检查训练集与测试集的 ID 重叠 | 🟡 | **C** | 数据流程第 5 步 |
| 训练清单缺版本号 | 🟡 | **A** | 输出固定的训练数据清单和版本说明 |
| 小目标标注 | ❌ | **C** | 标注同类异色、框重叠和小目标等困难条件 |
| scheduler 未统一（见下文说明） | 🟡 | **B** | 统一分辨率、采样步数、CFG、scheduler 和种子 |
| position_net 影响分析 | ❌ | **A** | 分析 position_net 对位置控制和对象出现率的影响 |
| 学习率和检查点分析 | ⏳ | **A** | 完成学习率和 checkpoint 分析（等 C-3 结果） |
| `architecture_analysis.md`（含 MIGC 的 EA / LA / SAC） | ❌ | **B** 为主，GLIGEN 结构部分可由 A 补充 | 分析 EA、LA、SAC；撰写架构比较 |
| CLIP-IQA 待补算 | 🟡 | **C** | 实现 GroundingDINO、SAM、HSV 和 CLIP-IQA 评测流程 |
| 指标名对齐 | 🟡 | **C** | Object Recall、Position Success、Attribute Success、Color Leakage Rate |
| 失败模式分析 | 🟡 | **C** | 按对象数量和困难条件分析失败模式 |
| 趋势图和失败案例图 | ❌ | **C** | 制作总体结果表、趋势图和失败案例图 |
| 报告撰写 | ❌ | **A / B / C 各写一部分** | A：GLIGEN、训练数据、position_net；B：MIGC、架构比较、fuser、计算成本；C：数据集、评测方法、结果、局限、结论 |
| C-3 验证集部分 | ⏳ | **A 和 B** | G1 的检查点分析归 A；G2 的学习率比较归 B |

按人统计：
- **A**：4 项，外加自己负责的报告部分；
- **B**：2 项，外加自己负责的报告部分；
- **C**：7 项，外加自己负责的报告部分。

### 关于"scheduler 未统一"（需要组内决定，负责人 B）

**scheduler（采样器）**是生成图片时用的去噪算法。生成一张图，就是从随机噪声开始，分 50 步逐步去噪；scheduler 决定每一步怎样根据模型的预测更新图片。同一个模型、同一个初始噪声，换一个 scheduler，画出来的图也会有所不同。

| | GLIGEN | MIGC |
|---|---|---|
| 当前使用的 scheduler | PLMS | Euler |
| 来源 | GLIGEN 官方推理脚本的默认设置 | MIGC 官方 benchmark 脚本的设置 |

- 分辨率、步数、CFG、种子和初始噪声都已统一，**只有 scheduler 不同**。
- **当时没有统一的原因**：两个官方代码库不同。GLIGEN 用自带的 ldm 代码，只支持 PLMS 和 DDIM；MIGC 基于 diffusers，使用 Euler。当时选择各用官方设置，以代表"原模型"的效果，并在 `configs/gen.yaml` 中注明。
- **影响**：分工要求统一 scheduler。现在测试集上两种方法的差距，严格来说可能有一小部分来自采样器的不同，而不完全来自模型本身。
- **可选方案**：
  1. **保持现状**：在报告中写明这一差异及理由（两者都用官方默认设置）。不需要重跑。
  2. **两者都改用 DDIM**：两个代码库都支持 DDIM，但需要先核对能否与官方结果对上。改了以后，Stage B、C-1 和微调评测的生成都要重跑，在集群上估计需要 3–4 小时。

### 成员 A：训练数据、GLIGEN 基线、position_net 微调

| 任务 | 状态 | 还差什么 |
|---|---|---|
| 根据 C 提供的测试 image ID 排除重叠 | 🟡 | 目前只在来源上隔离（train2014 和 val2014），**没有按测试 image ID 清单逐一排除** |
| 训练数据清单和版本说明 | 🟡 | 清单和统计都有；**缺版本号和文件校验值（sha256）** |
| 学习率和检查点分析 | ⏳ | 等 C-3 在验证集上的结果 |
| 分析 position_net 对位置控制和对象出现率的影响 | ❌ | 依赖 C-3 中 G1 在测试集上的结果 |
| 撰写 GLIGEN、训练数据和 position_net 实验部分 | ❌ | |

### 成员 B：MIGC、position_net + fuser 微调、统一推理

| 任务 | 状态 | 还差什么 |
|---|---|---|
| 分析 MIGC 的 EA、LA、SAC | 🟡 | 维度探测已运行（`outputs/analysis/migc_probe.json`）；**`architecture_analysis.md` 未写**（包括两张结构图和方法差异表） |
| 统一 scheduler | 🟡 | **未统一**：GLIGEN 用 PLMS，MIGC 用 Euler，都是各自官方的设置。需要组内决定，见上文"关于 scheduler 未统一" |
| 撰写 MIGC、架构比较、fuser 和计算成本部分 | ❌ | |

### 成员 C：测试集、自动评测、失败模式分析

| 任务 | 状态 | 还差什么 |
|---|---|---|
| 向 A 提供测试集 image ID | ❌ | 没有导出 ID 清单 |
| 标注小目标 | ❌ | 同类异色和框重叠已标注；小目标没有 |
| CLIP-IQA | 🟡 | 流程在本机跑通了；集群环境缺 setuptools，Stage B 和 C-1 的质量分**待补算** |
| 指标命名 | 🟡 | 内容都有，但名字还没对齐到 Object Recall / Position Success / Attribute Success / Color Leakage Rate |
| 汇总所有模型结果 | 🟡 | 原模型和比例实验已汇总；微调模型等 C-3 |
| 按对象数和困难条件分析失败模式 | 🟡 | 分组表能自动生成，**分析还没写**；小目标这一组缺失 |
| 总体结果表、趋势图、失败案例图 | ❌ | 结果表能自动生成；**趋势图、训练曲线、≥ 6 组配对图、失败案例图都没做** |
| 撰写数据集、评测方法、结果、局限和结论部分 | ❌ | |

### 数据责任流程

| 步骤 | 状态 |
|---|---|
| 1. C 导出测试 image ID | ❌ |
| 2. A 据此构建无重叠训练集 | 🟡 只做了来源隔离 |
| 3. A 发布训练清单、样本数和版本号 | 🟡 缺版本号 |
| 4. A、B 使用同一版本训练集 | ✅ |
| 5. C 检查训练集与测试集的 ID 重叠 | 🟡 只按文件名检查过，没有独立的检查记录 |
| 6. 训练集锁定后不再修改 | ✅ 至今没有修改 |

### 运行中 / 待提交的作业

| 作业 | 状态 |
|---|---|
| C-3 验证集部分（`RATIO=0.5`，作业 43571） | ⏳ 运行中 |
| C-3 测试集部分（G1 和选定学习率的 G2） | 等 C-3 验证集结果选出 G2 学习率后提交 |
| Stage B、C-1 补算 CLIP-IQA | 在集群 `mig_eval` 环境装好 `setuptools==69.5.1` 后，重新提交对应作业 |
| 下载集群结果到本机 | 未做。本机 `outputs/` 里仍是 Stage A 的结果和 152 张本机生成的旧测试图，**不用于最终结果** |
