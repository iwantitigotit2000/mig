#!/bin/bash
# Create the three conda envs on the CCDS TC2 cluster, with the exact package versions of the
# local envs (requirements_*.txt = `pip freeze` of D:/condamini/envs/mig_*, torch installed separately).
# Run on the HEAD NODE (it has internet for conda/pip); it only downloads and installs packages,
# it does not run project code. GPU checks happen inside the SLURM jobs (scripts/slurm/env.sh).
#
# Usage (from the project root):  bash scripts/cluster/mkenv_linux.sh
# If the jobs later report that CUDA is unavailable because the node driver is too old for cu121,
# rebuild with:  TORCH_CUDA=cu118 bash scripts/cluster/mkenv_linux.sh
set -ex
cd "$(dirname "$0")/../.."
module load anaconda
eval "$(conda shell.bash hook)"
TORCH_CUDA=${TORCH_CUDA:-cu121}
for e in gligen migc eval; do
  ENV=mig_$e
  conda env list | grep -q "^$ENV " || conda create -y -q -n $ENV python=3.10.22
  conda activate $ENV
  python -m pip install -q torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/$TORCH_CUDA
  python -m pip install -q -r scripts/cluster/requirements_$e.txt
  conda deactivate
done
conda env list
echo MKENV_DONE
