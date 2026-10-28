#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=H2020DeciderFicarra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=16:00:00
#SBATCH --job-name=wsi_gmm
#SBATCH --output=/work/H2020DeciderFicarra/ccRCC/logs/wsi_gmm_%j.log
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"
# GMM prototypes of the WSI per fold (utils/fit_wsi_gmm.py). Extra arguments are passed on, e.g. --dry_run.
#   sbatch utils/fit_wsi_gmm.sbatch [--k 32] [--dry_run]
set -euo pipefail
umask 007
source "$(ls /homes/admin/spack/opt/spack/linux-*/anaconda3-*/etc/profile.d/conda.sh | head -n 1)"
conda activate /work/H2020DeciderFicarra/fmiccolis/envs/multimodal_decider_new
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
cd /work/H2020DeciderFicarra/ccRCC/OXA-MISS
python utils/fit_wsi_gmm.py --config config/OXA_MISS_final_pancancer8_v2.yaml \
    --out /work/H2020DeciderFicarra/ccRCC/wsi_gmm_cache/pancancer9_UNIv2_K32 "$@"   # one cache per training configuration
