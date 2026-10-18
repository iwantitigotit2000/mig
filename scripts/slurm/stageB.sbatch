#!/bin/bash
#SBATCH --partition=MGPU-TC2
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8      # QoS limit 10
#SBATCH --mem=30G              # host RAM (not GPU memory); QoS limit 30G
#SBATCH --time=05:00:00        # QoS MaxWall 6 h; every stage is resumable/skip-existing
#SBATCH --job-name=mig_stageB
#SBATCH --output=outputs/logs/%x_%j.out
#SBATCH --error=outputs/logs/%x_%j.err
# Submit from the project root: sbatch scripts/slurm/stageB.sbatch
set -e
cd "${SLURM_SUBMIT_DIR:-.}"
source scripts/slurm/env.sh
bash scripts/stageB.sh
