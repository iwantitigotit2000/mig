# GLIGEN 与 MIGC 的多对象可控图像生成比较（课程项目）

> **新的 Claude 窗口请先读 `CLAUDE.md` → `docs/HANDOFF.md` → `docs/TASK_SPEC.md`。**

**当前状态（2026-10-20）**：Stage D 的数据隔离检查、统计分析、架构比较、趋势图、失败案例图和 8 页报告均已完成，并在 Apple Silicon Mac 上独立复现了 CPU 分析流程。C-3 的微调验证集分析已完成，测试集评测仍待 GPU 环境补跑。仓库中已提交的测试结果来自本地 RTX 4070，按 `PROVISIONAL` 处理；README 早期记录所称的 A40 结果没有原始文件，不能视为已验证。
本文第 4 节列出已完成内容，第 5 节仅列出仍需 GPU、原始证据或小组确认的事项。

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

大文件（权重、官方仓库、COCO、训练检查点）放在 `MIG_ASSETS` 指向的外部资源目录（集群可设为 `~/mig_assets`），不属于代码包；不要把本机绝对路径提交到仓库。

## 2. 环境

三个 conda 环境（依赖冲突，所以分开），都是 Python 3.10 + torch 2.1.2+cu121：

| 环境 | 用途 | 关键版本 |
|---|---|---|
| `mig_gligen` | GLIGEN 生成和微调 | transformers 4.19.2，omegaconf 2.1.1，openai-clip，kornia 0.6.12 |
| `mig_migc` | MIGC 生成 | diffusers 0.21.1，transformers 4.33.3，huggingface_hub 0.24.5，accelerate 0.23.0 |
| `mig_eval` | 评测和汇总 | transformers 4.33.3，supervision 0.7.0，pycocotools，pyiqa 0.1.10，setuptools 69.5.1 |

- 本机：运行外部资源目录中的 `mkenv.sh`（目录由 `MIG_ASSETS` 指定）。
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
- **测试集结果**（仓库已提交的本地 RTX 4070 运行，100 个布局 × 2 个种子 = 每个方法 200 张；方括号内为按布局 bootstrap 的 95% 置信区间）。该运行因缺少明确的机器声明文件而标为 **PROVISIONAL**，详见 `outputs/final_analysis/PROVENANCE.md`：

| 指标 | GLIGEN（G0） | MIGC | MIGC − G0 |
|---|---|---|---|
| 官方实例成功率（位置和颜色都对） | 0.321 | **0.665** | +0.344 [+0.294, +0.393] |
| 官方整图成功率 | 0.065 | **0.280** | +0.215 [+0.150, +0.280] |
| 官方 mIoU（颜色错误记 0） | 0.288 | **0.572** | +0.284 [+0.243, +0.325] |
| 纯定位 IoU | 0.690 | 0.741 | +0.051 [+0.023, +0.080] |
| 一对一匹配的对象出现率 | 0.917 | 0.932 | +0.015 [−0.013, +0.042]（不显著） |
| 失败：对象缺失 / 位置错误 / 颜色错误 | 0.033 / 0.159 / **0.487** | 0.009 / 0.119 / 0.207 | |
| 颜色错误中属于串扰的比例 | 0.572 | 0.440 | |
| 评测异常数 | 0 | 0 | |
| 每张耗时 / 峰值显存 | 5.7 s / 9.4 GB | 6.5 s / 7.8 GB | |
| CLIP-IQA | 0.689 | **0.707** | +0.018 [+0.002, +0.034] |

**观察**：两种方法画出对象的比例相近，MIGC 的定位略好一些。差距主要来自颜色绑定：GLIGEN 约一半的对象颜色错误，其中超过一半是串扰（用了画面中另一个对象的颜色）。

### 4.3 grounding 控制持续比例（Stage C-1，仅用验证集；原始输出未提交）

比例 r 的含义：GLIGEN 的 alpha_type 为 [r, 0, 1−r]；MIGC 的 MIGCsteps 为 r × 50。验证集 30 个布局 × 2 个种子。以下数字来自早期小组日志；由于原始结果文件不在仓库中，报告只定性引用并明确标注为未验证。

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

### 4.6 Stage D 分析、图表与报告

- **数据与隔离检查**：固定并记录 test / val / pilot / train manifest 的版本、行数和 sha256；确认 test 与 val 布局不重叠，并检查 130 个评测布局与 3,100 张训练/训练验证图的框布局近重复，阈值 mean matched IoU ≥ 0.9 时为 0 个。
- **image ID 边界**：COCO-MIG bench 文件没有保存原 COCO image ID，因此不能导出或逐 ID 比对。仓库导出了可用的 layout ID、sample ID、来源与内容哈希，并完成来源隔离和布局级近重复检查；报告没有把这些替代检查写成 image-ID 检查。
- **架构分析**：`analysis/architecture_analysis.md` 已分析 GLIGEN fuser / `position_net` 与 MIGC 的 EA、LA、SAC，并给出两张架构图和比较表。没有做单组件消融，因此机制解释是证据支持的推断，不是因果结论。
- **微调分析**：`analysis/finetune_analysis.md` 已完成学习率、检查点、权重漂移、门值和 `position_net` 分析。结论仅来自 30-layout 验证集；G1/G2 测试集结果仍待补跑。
- **结果与失败分析**：完成总体与子组 bootstrap、颜色阈值敏感性、计算成本、17 张趋势/架构/失败案例图及固定规则选出的案例 A–E。`failure_cases/cases.csv` 的人工备注栏仍可由组员在最终提交前补充。
- **报告与复现**：提交版报告为 `report/group_7_report.pdf`（15 页，单栏；由 `report/make_docx.js` 生成的 Word 文档在 Google Docs 中定稿导出）。`report/report.tex` / `report.pdf` 是同一内容的双栏 LaTeX 版本，保留它是因为数字检查流水线（`check_report_numbers.py`）作用于它；两版数字均来自 `report/tables/`。`bash scripts/stageD_analysis.sh` 可从已提交的 CSV、缓存和图片重建分析、表格、图和 PDF，并检查报告数字。Mac CPU 独立复现记录见 `docs/MAC_CPU_REPRODUCTION.md`。

---

## 5. 仍需完成或由小组确认的内容

### 5.1 原分工项的当前状态

成员 A / B / C 尚未最终对应到具体同学，因此这里不再把任务绑定到个人；以证据状态为准。

| 原任务 | 当前状态 | 证据边界 / 下一步 |
|---|---|---|
| 导出测试 image ID | ⚠️ 无法从现有 bench 导出 | 上游 YAML 不含 COCO image ID。已导出 layout ID、sample ID 和哈希，并做来源与布局级隔离检查；不能冒充逐 image-ID 检查。 |
| 训练/评测隔离与 manifest 版本 | ✅ 已完成 | `manifests/MANIFEST_VERSIONS.json`、`manifests/eval_layout_ids.csv`、`outputs/checks/train_eval_isolation.*`。 |
| 小目标/困难条件标注 | ✅ 已完成可复现替代 | 数据中没有 COCO “small” 对象；按相对面积四分位分析，并保留同类异色与框重叠分组。 |
| `position_net` 影响分析 | ✅ 验证集完成；测试集待补 | `analysis/finetune_analysis.md`。没有证据表明它改善定位或对象出现率；正向点估计主要在颜色阈值通过率。 |
| 学习率和检查点分析 | ✅ 验证集完成 | 1e-5 未见效果；5e-5 点估计更高但与 1e-5 的直接对比区间跨 0；没有证据支持 2000 步优于 500 步。 |
| EA / LA / SAC 与架构比较 | ✅ 已完成 | `analysis/architecture_analysis.md` 和两张架构图；未做组件消融，不能表述为因果证明。 |
| 指标命名、总体/子组统计 | ✅ 已完成 | Object Recall、Position Success、Attribute Success、Color Leakage Rate 已统一，采用布局级配对 bootstrap。 |
| 趋势图和失败案例图 | ✅ 已完成 | 共 17 张图，案例 A–E 由固定规则选取；CSV 的人工备注栏可在最终提交前补充。 |
| 报告撰写 | ✅ 8 页草稿完成 | 数字自动核对通过；仍需替换 Group XX、姓名并确认贡献声明。 |
| C-3 微调测试集评测 | ❌ 待 GPU 补跑 | 当前 `position_net` / G2 结论只基于 30-layout 验证集，报告已明确说明。 |

### 5.2 仍缺少的实验或原始证据

1. **C-3 测试集**：在 GPU 上对验证集选出的 G1 与 G2 5e-5 配置跑测试集。没有 GPU 时可以提交现有版本，但不能把验证集结论写成测试集结论。
2. **A40 原始输出**：早期小组记录提到 A40 数字，但原始 CSV、metadata 与 provenance 文件不在仓库。拿到后应选择一个 canonical run，重新运行 Stage D，不能混用本地与集群数字。
3. **统一 sampler / 组件消融**：GLIGEN 使用官方 PLMS，MIGC 使用官方 Euler；EA/LA/SAC 与 fuser 的作用没有单独消融。这两点已作为限制写进报告。如果课程明确要求统一 sampler，则需用共同支持的 DDIM 重跑。
4. **人工案例备注**：`outputs/final_analysis/failure_cases/cases.csv` 已含自动判定和选择原因，但 `human_observation` 尚未逐行填写；最终定稿前可由组员对照 A–E 图补充，不应自动杜撰。

### 5.3 CPU 复现

从仓库根目录运行：

```bash
bash scripts/stageD_analysis.sh 2>&1 | tee stageD_run.log
```

该命令会检查数据隔离，重算统计、表格和 17 张图，验证报告数字，并在安装了 `pdflatex` 时重建 `report/report.pdf`。它不会重跑 GPU 训练、生成、GroundingDINO/SAM 或 CLIP-IQA 推理。Apple Silicon Mac 的独立复现记录见 `docs/MAC_CPU_REPRODUCTION.md`。

### 5.4 最终提交前清单

- [x] 提交版报告 `report/group_7_report.pdf` 已写入组号与姓名；`report/report.tex` 中的 `Group XX` 占位符只影响 LaTeX 草稿。
- [ ] `group_7_report.pdf` 第 4.1 节 "Table 2 and Figure 2 give the main result. ¹" 的脚注标记 ¹ 没有对应脚注正文（Google Docs 导出时丢失）；提交前在源文档中删除 ¹，或补回数据来源说明。
- [ ] 小组确认 `report/contribution.md`，按实际贡献分配，不沿用旧的 A/B/C 假设。
- [ ] 决定是否补跑 C-3 测试集；如不补跑，保留现有明确限制。
- [ ] 如收到 A40 原始文件，更新 provenance 并完整重跑 Stage D。
- [ ] 运行一次 `bash scripts/stageD_analysis.sh`，确认所有检查通过后再提交。
