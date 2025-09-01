#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --job-name=gs_custom
#SBATCH --time=45:00
#SBATCH --output=/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/slurm_out/MUSE_TR_inference_%j.out
#SBATCH --cpus-per-task=2
#SBATCH --account=H2020DeciderFicarra
#SBATCH--constraint="gpu_L40S_48G|gpu_A40_48G|gpu_RTX5000_16G|gpu_RTXA5000_24G|gpu_RTX6000_24G"

# |gpu_2080Ti_11G
# Variabile booleana per indicare se il parametro è stato trovato
found=false

# Scorri tutti i parametri passati allo script
for ((i = 1; i <= $#; i++)); do
  arg="${!i}"
  
  if [[ $arg == --grid_search_model_version_index ]]; then
    # Prendi il valore dal parametro successivo
    next_index=$((i + 1))
    grid_search_model_version_index="${!next_index}"
    # echo "Valore di --grid_search_model_version_index: $grid_search_model_version_index"
    found=true
    break
  fi
done

if [ "$found" = false ]; then
    echo "Error: --grid_search_model_version_index parameter not found"
    exit 1
fi

nvidia-smi
module list
module avail
module unload cuda/12.1
module load cuda/11.8
nvcc --version

. /usr/local/anaconda3/etc/profile.d/conda.sh
conda deactivate
conda activate multimodal_decider
# #Treatment Response chemorefractory
# python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_TR_MISSING_MODALITIES_chemorefractory.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose --seed ${SLURM_ARRAY_TASK_ID} 

# #Overall Survival chemorefractory
# python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_OS_MISSING_MODALITIES_chemorefractory.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose --seed ${SLURM_ARRAY_TASK_ID} 

# OS dataset TCGA
#~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_MISSING_MODALITIES.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose 

# ~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_v2_MISSING_MODALITIES.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose 


# OS dataset TCGA - MUSE
~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py \
  --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/MUSE_TR.yaml \
  --grid_search_model_version_index ${grid_search_model_version_index} --verbose 