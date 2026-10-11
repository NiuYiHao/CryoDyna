#!/bin/bash
#SBATCH --job-name=drgn80s-preflight
#SBATCH --account=gaolab
#SBATCH --qos=normal
#SBATCH --partition=gpu1
#SBATCH --gres=gpu:geforce:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
#SBATCH --output=/share/home/niuyh/data/Cryodyna_optpose_paper_20261107/drgnai80s/logs/preflight-%j.log
set -euo pipefail
TASK_ROOT=/share/home/niuyh/data/Cryodyna_optpose_paper_20261107/drgnai80s
cd "$TASK_ROOT"
export PYTHONPATH="$TASK_ROOT/vendor:/share/home/niuyh/data/Cryodyna_optpose_paper_20261107/preflight_7c42361/env_overlay:$TASK_ROOT/source"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 MPLBACKEND=Agg PYTHONDONTWRITEBYTECODE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONFAULTHANDLER=1
/share/home/niuyh/.conda/envs/test_20260710_cu128/bin/python -u code/preflight80s.py
