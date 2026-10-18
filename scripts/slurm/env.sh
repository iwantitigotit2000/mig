# Sourced by every sbatch job on the CCDS TC2 cluster (runs on the compute node).
# Sets the asset root and env interpreters, forces offline model loading, and checks the GPU.
module load anaconda
eval "$(conda shell.bash hook)"

export MIG_ASSETS=${MIG_ASSETS:-$HOME/mig_assets}
py_of() { (conda activate "$1" >/dev/null 2>&1 && command -v python); }
export PY_GLIGEN=${PY_GLIGEN:-$(py_of mig_gligen)}
export PY_MIGC=${PY_MIGC:-$(py_of mig_migc)}
export PY_EVAL=${PY_EVAL:-$(py_of mig_eval)}

# All weights are uploaded beforehand (scripts/cluster/upload_list.txt); never download inside a job.
export HF_HOME=$MIG_ASSETS/hf_home
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

for v in PY_GLIGEN PY_MIGC PY_EVAL; do
  [ -x "${!v}" ] || { echo "ERROR: $v not found; run scripts/cluster/mkenv_linux.sh"; exit 2; }
done
[ -d "$MIG_ASSETS/weights" ] || { echo "ERROR: MIG_ASSETS=$MIG_ASSETS has no weights/"; exit 2; }
[ -f "$HOME/.cache/clip/RN50.pt" ] || echo "WARNING: ~/.cache/clip/RN50.pt missing; CLIP-IQA will try to download it"

echo "node=$(hostname) MIG_ASSETS=$MIG_ASSETS"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$PY_GLIGEN" -c "import torch; assert torch.cuda.is_available(), 'CUDA unavailable (driver too old for this torch build?)'; print(torch.__version__, torch.cuda.get_device_name(0))"

export MAX_HOURS=${MAX_HOURS:-4.7}
rm -rf outputs/.gpu_lock    # a job killed at the time limit cannot run its EXIT trap; QoS allows one GPU job at a time
