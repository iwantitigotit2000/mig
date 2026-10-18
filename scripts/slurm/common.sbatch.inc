# Shared notes for the sbatch files (CCDS TC2, QoS "normal": cpu<=10, gpu=1, mem<=30G per user,
# at most 2 running jobs, MaxWall 6 h). The limits are per user across ALL running jobs, so run the
# stages one after another. --mem is host RAM, not GPU memory; peak host memory measured locally:
# ~11 GB (eval), ~10 GB (generation). Job setup shared by all stages lives in scripts/slurm/env.sh.
