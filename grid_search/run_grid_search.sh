#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --account=H2020DeciderFicarra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8   # data loading is the bottleneck (GPU ~10% with 2 CPUs); 8 fits every eligible node (>= 12 CPUs per GPU) and 8 runs x 8 = 64 of the 128 CPUs per user
#SBATCH --mem=180G   # page cache of the training slides (pancancer9 float16 ~148 GB, memory-mapped) + processes; external slides read at the end (main.py evaluate_external_at_end). Set from the measurement run
#SBATCH --time=24:00:00
#SBATCH --job-name=OXA_MISS_grid
#SBATCH --output=/work/H2020DeciderFicarra/ccRCC/logs/grid_%A_%a.log
#SBATCH --constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTXPro6000B_96G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"   # sm_75+ (torch 2.8): no 1080 / P100

# One version of a grid search per array task (version index = SLURM_ARRAY_TASK_ID). Submit with
# grid_search/run_grid_search.py, or directly:
#   sbatch --array=0-<N-1>%<max parallel> grid_search/run_grid_search.sh <config.yaml> <versions.json> [main.py args]
set -eo pipefail
# shared project: results, checkpoints and logs group-writable (dirs 770, files 660); the SLURM log is created
# before the script runs, with the submitting shell's umask
umask 007
chmod g+rw "/work/H2020DeciderFicarra/ccRCC/logs/grid_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.log" 2>/dev/null || true

if [[ $# -lt 2 ]]; then
    echo "usage: sbatch --array=0-<N-1> $0 <config.yaml> <versions.json> [extra main.py args]" >&2
    exit 1
fi
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "SLURM_ARRAY_TASK_ID not set: submit as an array job (sbatch --array=...)" >&2
    exit 1
fi
# absolute paths: the script cd's into the repo below, relative paths are of the submit directory
config_path="$(realpath "$1")"
versions_path="$(realpath "$2")"
shift 2
REPO=/work/H2020DeciderFicarra/ccRCC/OXA-MISS
[[ -f "$config_path" ]] || { echo "config not found: $config_path" >&2; exit 1; }
[[ -f "$versions_path" ]] || { echo "versions not found: $versions_path" >&2; exit 1; }

nvidia-smi
# shared environment (the same packages for everybody who launches): torch 2.8 with its own CUDA 12.8 libraries,
# so no cuda module (an older CUDA on LD_LIBRARY_PATH could shadow them)
ENV=/work/H2020DeciderFicarra/fmiccolis/envs/multimodal_decider_new
module unload cuda || true
# the spack architecture folder of anaconda changed (linux-ivybridge -> linux-x86_64_v2): look it up
source "$(ls /homes/admin/spack/opt/spack/linux-*/anaconda3-*/etc/profile.d/conda.sh | head -n 1)"
conda deactivate || true
conda activate "$ENV"
set -u  # after conda: its activation scripts read unset variables

echo "config: $config_path | versions: $versions_path | index: $SLURM_ARRAY_TASK_ID"
cd "$REPO"
"$ENV/bin/python" "$REPO/main.py" \
    --config "$config_path" \
    --grid_search_versions "$versions_path" \
    --grid_search_model_version_index "$SLURM_ARRAY_TASK_ID" \
    --verbose "$@"
