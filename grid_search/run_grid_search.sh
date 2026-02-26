#!/bin/bash
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --job-name=Frontiers_OXA
#SBATCH --time=01:00:00
#SBATCH --output=/work/H2020DeciderFicarra/fmiccolis/Frontiers/logs/%j_OXA_MISS.out
#SBATCH --cpus-per-task=2
#SBATCH --account=H2020DeciderFicarra
#SBATCH--constraint="gpu_A40_45G|gpu_L40S_45G|gpu_RTX5000_16G|gpu_RTX6000_24G|gpu_RTX_A5000_24G"

# |gpu_2080Ti_11G
# Variabile booleana per indicare se il parametro è stato trovato
# found=false

nvidia-smi
# module list
# module avail
module unload cuda/12.1
module load cuda/11.8
# nvcc --version

# . /usr/local/anaconda3/etc/profile.d/conda.sh
source /homes/admin/spack/opt/spack/linux-ivybridge/anaconda3-2023.09-0-*/etc/profile.d/conda.sh
conda deactivate
conda activate multimodal_decider
# # Scorri tutti i parametri passati allo script
# for ((i = 1; i <= $#; i++)); do
#   arg="${!i}"
  
#   if [[ $arg == --grid_search_model_version_index ]]; then
#     # Prendi il valore dal parametro successivo
#     next_index=$((i + 1))
#     grid_search_model_version_index="${!next_index}"
#     # echo "Valore di --grid_search_model_version_index: $grid_search_model_version_index"
#     found=true
#     break
#   fi
# done

# if [ "$found" = false ]; then
#     echo "Error: --grid_search_model_version_index parameter not found"
#     exit 1
# fi


# Parse arguments con while (scalabile e pulito)
while [[ $# -gt 0 ]]; do
  case "$1" in
    --grid_search_model_version_index)
      grid_search_model_version_index="$2"
      shift 2
      ;;
    --model_name)
      model_name="$2"
      shift 2
      ;;
    *)
      echo "❌ Errore: argomento sconosciuto: $1"
      exit 1
      ;;
  esac
done


 Check obbligatori
if [[ -z "$grid_search_model_version_index" ]]; then
    echo "❌ Errore: parametro --grid_search_model_version_index mancante"
    exit 1
fi

if [[ -z "$model_name" ]]; then
    echo "❌ Errore: parametro --model_name mancante"
    exit 1
fi

echo "✅ Parametri letti correttamente:"
echo "Grid Search Index: $grid_search_model_version_index"
echo "Model Name: $model_name"
config_path="/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/${model_name}.yaml"
# Mappa modello -> config file
# case "$model_name" in
#   "OXA_MISS")
#     config_path="/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/OXA_MISS.yaml"
#     ;;
#   "CLAM")\
#     config_path="/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/CLAM.yaml"
#     ;;
#   "ABMIL")
#     config_path="/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/ABMIL.yaml"
#     ;;
#   "TransMIL")
#     config_path="/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/TransMIL.yaml"
#     ;;
#   *)
#     echo "❌ Errore: modello sconosciuto '$model_name'"
#     exit 1
#     ;;
# esac

echo "🔗 Config YAML selezionato: $config_path"

# Esegui il tuo Python
~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/main.py \
    --config "$config_path" \
    --grid_search_model_version_index "$grid_search_model_version_index" \
    --verbose


# #Treatment Response chemorefractory
# python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_TR_MISSING_MODALITIES_chemorefractory.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose --seed ${SLURM_ARRAY_TASK_ID} 

# #Overall Survival chemorefractory
# python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_OS_MISSING_MODALITIES_chemorefractory.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose --seed ${SLURM_ARRAY_TASK_ID} 

# OS dataset TCGA
# ~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_MISSING_MODALITIES.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose 

# ~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/custom_multimodal_XA_v2_MISSING_MODALITIES.yaml --grid_search_model_version_index ${grid_search_model_version_index} --verbose 


# OS dataset TCGA - MUSE
# ~/.conda/envs/multimodal_decider/bin/python /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/main.py \
#   --config /work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/experiments/MUSE_OS_MISSING_MODALITIES.yaml \
#   --grid_search_model_version_index ${grid_search_model_version_index} --verbose 