# 在 CCDS TC2 集群上运行（Stage B–C 全部在集群上运行）

依据：`E:\transformer\CCDSGPU-TC2-Masters-UserGuide.pdf`（2025-08-13 版）。
QoS "normal"：CPU ≤ 10、GPU 1 张、主机内存 ≤ 30G（指内存，不是显存）、同时最多 2 个作业、单个作业最长 6 小时。
资源上限按用户累计，所以四个阶段要**一个接一个**提交。家目录配额 100 GB，本项目约用 45 GB（估算）。集群不备份，结果要及时下载回本机。

下文中 `<用户名>` 换成你的 NTU 账号（小写）。
- **【本机】** 表示在本机 Git Bash 里运行（VS Code 终端选 Git Bash 即可）。
- **【集群】** 表示 SSH 登录到集群后运行。

---

## 0. 登录集群（第一次登录才会创建家目录）
要在校园网内或连 NTU VPN。

【本机】

    ssh <用户名>@10.96.189.12

【集群】确认账号的 QoS 和限制，应看到 `normal  cpu=10,gres/gpu=1,mem=30G  2  06:00:00`：

    mytcinfo
    df -h ~                      # 看家目录剩余空间（需要约 45 GB）
    exit

---

## 1. 上传（约 17 GB）
用 tar 通过 ssh 直接传，不在本机生成临时压缩包；每条命令输入一次密码。大文件需要较长时间，中断了就重新运行同一条命令。

【本机】① 资源（权重、官方仓库、COCO 子集），按 `scripts/cluster/upload_list.txt` 传到集群的 `~/mig_assets/`：

    cd /e/transformer/mig_assets
    tar cf - -T <(grep -v '^#' ../mig_project/scripts/cluster/upload_list.txt | grep .) \
      | ssh xinkuo001@10.96.189.12 "mkdir -p ~/mig_assets && tar xf - -C ~/mig_assets"

【本机】② 代码和样本清单，传到 `~/mig_project/`，不传本机生成的图片：

    cd /e/transformer
    tar cf - --exclude='mig_project/outputs/gen' --exclude='__pycache__' mig_project \
      | ssh xinkuo001@10.96.189.12 "tar xf - -C ~"

【本机】③ CLIP-IQA 用的 RN50 权重：

    ssh <用户名>@10.96.189.12 "mkdir -p ~/.cache/clip"
    scp ~/.cache/clip/RN50.pt <用户名>@10.96.189.12:~/.cache/clip/

【集群】检查。`mig_assets` 应约 17G，`weights` 下应有 7 项，`train2014_subset` 应有 3100 张：

    du -sh ~/mig_assets ~/mig_assets/* ~/.cache/clip/RN50.pt
    ls ~/mig_assets/weights
    ls ~/mig_assets/coco/train2014_subset | wc -l
    ls ~/mig_project/outputs/logs          # 目录必须存在，SLURM 不会自动创建

---

## 2. 建环境（在登录节点，只需一次，约 20–40 分钟）
【集群】

    cd ~/mig_project
    bash scripts/cluster/mkenv_linux.sh 2>&1 | tee mkenv.log
    tail -3 mkenv.log                      # 最后一行是 MKENV_DONE 即成功
    module load anaconda && conda env list # 应列出 mig_gligen、mig_migc、mig_eval

如果提示 `module: command not found`，先运行 `source /etc/profile.d/modules.sh`，再重新运行上面的命令。

---

## 3. 按顺序提交作业
以下命令都在【集群】的 `~/mig_project` 下运行。**等前一步完成再提交下一步。**

每个作业开头会打印节点名、GPU 型号和驱动版本，并检查 CUDA 是否可用。如果 `.err` 里出现 `CUDA unavailable`，说明驱动不支持 cu121，要重建环境：

    module load anaconda && eval "$(conda shell.bash hook)"
    conda env remove -y -n mig_gligen; conda env remove -y -n mig_migc; conda env remove -y -n mig_eval
    TORCH_CUDA=cu118 bash scripts/cluster/mkenv_linux.sh 2>&1 | tee mkenv.log

### 第 1 步：Stage B，原模型在测试集上生成和评测（约 1 小时*）

    cd ~/mig_project
    sbatch scripts/slurm/stageB.sbatch                 # 会返回 Submitted batch job <jobid>
    squeue -u <用户名>                                  # ST 一栏：R 是运行中，PD 是排队
    tail -f outputs/logs/mig_stageB_<jobid>.out        # 看进度，Ctrl+C 退出查看（不影响作业）
    grep -c . outputs/logs/mig_stageB_<jobid>.err      # 错误日志行数

完成检查（输出 `STAGEB_DONE`，并且两个目录各有 200 张图）：

    grep STAGEB_DONE outputs/logs/mig_stageB_*.out
    ls outputs/gen/test/G0/images | wc -l; ls outputs/gen/test/MIGC/images | wc -l
    cat outputs/results/test_original/summary.md

如果超时或中断，直接重新运行 `sbatch scripts/slurm/stageB.sbatch`，已完成的样本会自动跳过。

### 第 2 步：Stage C-1，在验证集上比较 grounding 持续比例（约 40 分钟*）

    sbatch scripts/slurm/stageC_val_ratio.sbatch
    squeue -u <用户名>
    grep STAGEC_RATIO_DONE outputs/logs/mig_stageC_val_ratio_*.out
    cat outputs/results/val_ratio/summary.md           # 选出 GLIGEN 最好的比例 r（0.3 / 0.5 / 0.8）

### 第 3 步：Stage C-2，训练 G1、G2（3 组各 2000 步，每组约 1 小时*）

    sbatch scripts/slurm/stageC_train.sbatch
    squeue -u <用户名>
    tail -2 ~/mig_assets/runs/g1_lr5e-5/train_log.jsonl    # 看训练进度（g2_lr5e-5、g2_lr1e-5 同理）
    grep STAGEC_TRAIN_DONE outputs/logs/mig_stageC_train_*.out

如果作业到时间停了，但还没有 `STAGEC_TRAIN_DONE`，就重新提交同一条 `sbatch` 命令。训练会从 `ckpt_latest.pt` 接着跑。
完成后，每组应有 `ckpt_000500.pt` 和 `ckpt_002000.pt` 两个检查点：

    ls ~/mig_assets/runs/*/ckpt_00*.pt

### 第 4 步：Stage C-3，用微调后的检查点生成和评测（未估算）
分两次提交。`<r>` 用第 2 步选出的比例，例如 `0.3`。

① 只跑验证集：比较 G2 的两个学习率，以及早期（500 步）和后期（2000 步）检查点。

    sbatch --export=ALL,RATIO=<r> scripts/slurm/stageC_eval_ft.sbatch
    grep STAGEC_FT_DONE outputs/logs/mig_stageC_eval_ft_*.out
    cat outputs/results/val_finetune/summary.md        # 选出 G2 更好的学习率：5e-5 或 1e-5

② 在测试集上跑 G1 和选中的 G2。验证集部分已经完成，会自动跳过：

    sbatch --export=ALL,RATIO=<r>,G2_LR=<lr> scripts/slurm/stageC_eval_ft.sbatch
    grep STAGEC_FT_DONE outputs/logs/mig_stageC_eval_ft_*.out

\* 用时是按本机 RTX 4070 实测推算的，A40 上没有测过。

---

## 4. 常用命令【集群】

    squeue -u <用户名>          # 排队/运行中的作业
    scancel <jobid>             # 取消作业
    seff <jobid>                # 作业结束后看实际 CPU/内存用量
    myjobhistory                # 今天的作业记录
    ncdu ~                      # 看家目录空间（按 q 退出）

如果作业状态一直是 PD（排队），并且原因写着 `QOSMax...`，说明超出了 QoS 限制，作业不会运行。要先 `scancel <jobid>`，再检查是不是同时提交了多个作业。

---

## 5. 下载结果【本机】
下载到单独的 `E:\transformer\cluster_results\`，不覆盖本机 `outputs/gen/test/G0` 里那 152 张本机生成的旧图。

    mkdir -p /e/transformer/cluster_results
    # 生成的图片、评测结果、日志
    ssh <用户名>@10.96.189.12 "tar cf - -C ~/mig_project outputs" | tar xf - -C /e/transformer/cluster_results
    # 训练日志和保留的检查点（跳过体积大、只用于续训的 ckpt_latest.pt 和 init_state.pt）
    ssh <用户名>@10.96.189.12 "tar cf - -C ~/mig_assets --exclude=ckpt_latest.pt --exclude=init_state.pt runs" \
      | tar xf - -C /e/transformer/cluster_results
