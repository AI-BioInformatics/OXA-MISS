import os, json
from itertools import product
import pandas as pd

csvpath="/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/MUSE_runs_mapping_withseed.csv"
df_prev_runs=pd.read_csv(csvpath)

# Normalizza il campo modality_setting per facilitare il match
df_prev_runs['modality_setting'] = df_prev_runs['modality_setting'].astype(str)


model_name = "ProSurv" #Custom_Multimodal_XA

# Usa le chiavi con dot notation per parametri annidati
changes_template = {
    
    # --- W&B E GENERALI ---

    'title': ["{model_name}_augtrain_{tumor_type}"], #custom_multimodal_XA_MISSING_MOD_RATE_kfold_5_all_tissues
    'project_dir': ["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results"],
    'seed': [43], #,
    
    'wandb.mode': ["online"],
    'wandb.project' : ["augmented_training"],
    
    
        # --- DATA LOADER ---
    
        # --- MISSING MODALITY TABLES ---
    'data_loader.missing_modalities_tables.active': [True],
    'data_loader.missing_modalities_tables.missing_mod_rate': ["complete"],#complete
    
     # ---OTHER DATA LOADER PARAMS ---
    'data_loader.file_genes_group' : ["/work/H2020DeciderFicarra/D2_4/Development/HGCN/HGCN_code/pathways_ensg.json"],
    'data_loader.datasets_configs': [["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/datasets/{dataset}_{tumor_type}_dataset.yaml"]],
    'data_loader.task_type': ["Survival"],
    'data_loader.max_patches': [4096],
    'data_loader.batch_size': [1],
    'data_loader.real_batch_size': [4],
    'data_loader.n_bins': [4],
    'data_loader.sample': [True],
    'data_loader.test_sample': [False],
    'data_loader.load_slides_in_RAM': [True],
    'data_loader.label_name': ["Survival"],
    'data_loader.censorships_name': ["None"],
    'data_loader.eps': [1e-6],
    'data_loader.num_workers': [2],
    'data_loader.train_size': [0.7],
    'data_loader.val_size': [0.15],
    'data_loader.test_size': [0.15],
    'data_loader.random_state': [42],
    'data_loader.preprocessing': ["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/preprocessing.yaml"],
    'data_loader.augmentation': ["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/augmentation.yaml"],
    # --- K-FOLD SPLIT ---
    'data_loader.KFold.splits': ["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/splits_train_val_test/{tumor_type}","/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/aug_splits_train_val_test/{tumor_type}"],
    'data_loader.missing_modality_table' : ["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/aug_missing_mod/{tumor_type}_missing_modality_table.csv"],
    'data_loader.KFold.internal_val_size': [0.0],

        # --- MISSING MODALITY TEST ---
    'missing_modality_test.active': [True],
    'missing_modality_test.test_scenarios_on_each_epoch': [True],
    'missing_modality_test.scenarios': [['complete','genomics_miss_100','wsi_miss_100']],  # Lista vuota o nomi


    # --- MODEL ---
    'model.name': ["ProSurv"],
    'model.pretrained': [False],
    'model.save_checkpoints': [True],
    # --- MODEL.KWARGS ---
    'model.kwargs.genomics_group_name': [["tumor_suppression","oncogenesis","protein_kinases","cellular_differentiation","cytokines_and_growth"]],
    'model.kwargs.cnv_group_name': [["tumor_suppression","oncogenesis","protein_kinases","cellular_differentiation","cytokines_and_growth"]],
    'model.kwargs.input_modalities': [["WSI", "Genomics"]],


    # --- LOSS ---
    'loss.name': ["NLLSurvLoss"],
    'loss.kwargs.alpha': [0.0],
    'loss.kwargs.eps': [1e-7],
    'loss.kwargs.reduction': ["mean"],

    # --- SCHEDULER ---
    'scheduler.batch_step': [False],
    'scheduler.name': ["MultiStepLR"], #"OneCycleLR","MultiStepLR","CosineAnnealingLR"
    'scheduler.milestones': [[10]],
    'scheduler.gamma': [0.2],
    'scheduler.pct_start': [0.1],
    'scheduler.steps_per_epoch': [1],

    # --- OPTIMIZER ---
    'optimizer.learning_rate': [0.0001], #, ,  da lanciare 0.001
    'optimizer.name': ["RAdam"],
    'optimizer.weight_decay': [0.0001], #0.001,0.0001,0.00001 da testare dopo
    'optimizer.momentum': [None],
    # --- TRAINER ---
    'trainer.reload': [False],
    'trainer.checkpoint': [''],
    'trainer.do_train': [False],
    'trainer.do_test': [True],
    'trainer.do_inference': [False],
    'trainer.do_kfold': [False],
    'trainer.epochs': [20],
    'trainer.patience': [20],
    'trainer.AEM_lamda': [0.3],
    'trainer.Save_XA_attention_files': [True],
}


# Funzione per costruire dict annidati a partire da una stringa 'a.b.c'
def set_nested_value(dictionary, path, value):
    keys = path.split('.')
    for key in keys[:-1]:
        dictionary = dictionary.setdefault(key, {})
    dictionary[keys[-1]] = value

# Funzione helper per un'unica combinazione
def nested_dict_from_flat(keys, values):
    config = {}
    for k, v in zip(keys, values):
        set_nested_value(config, k, v)
    return config

tumor_types = ["chemorefractory"] #[#"BLCA", "BRCA", "COAD", "HNSC", "KIRC","KIRP" ,"LUAD", "LUSC","OV","STAD"]# ["PDS"] #, "NACT_primary"]#["BLCA", "BRCA", "COAD", "HNSC", "KIRC","KIRP" ,"LUAD", "LUSC","OV","STAD"] #"BRCA" manca solo seed 44 (running) "LUAD", "KIRP" devo sistemare i fold
dataset= 'Decider'# "TCGA"
# Output combinazioni finali
combinations = []

# Per ogni tipo di tumore, genera un blocco changes con il nome già inserito
for tumor in tumor_types:
    changes = {}
    for key, val_list in changes_template.items():
        resolved = []
        for v in val_list:
            if isinstance(v, str):
                resolved.append(v.format(tumor_type=tumor, model_name=model_name, dataset=dataset))
            elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
                resolved.append([elem.format(tumor_type=tumor, model_name=model_name, dataset=dataset) for elem in v])
            else:
                resolved.append(v)

        changes[key] = resolved

    # Genera combinazioni come prima
    keys = list(changes.keys())
    values = list(changes.values())
    raw_combinations = list(product(*values))
    tumor_combinations = [nested_dict_from_flat(keys, combo) for combo in raw_combinations]
    combinations.extend(tumor_combinations)

# # Ensure the directory exists
output_dir = '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/grid_search/models_versions'
os.makedirs(output_dir, exist_ok=True)
print(f"Total combinations: {len(combinations)}")


# Aggiorna i checkpoint nei dizionari di configurazione
updated_combinations = []
missing = []

for combo in combinations:
    seed = combo.get('seed')
    tumor_type = combo['title'].split("_")[3]  # Estratto da title es: MUSE_OV_kfold_5_all_tissues

    # Trova la modality setting usata (deriva da 'missing_mod_rate')
    miss_rate = combo.get('data_loader', {}).get('missing_modalities_tables', {}).get('missing_mod_rate')
    modality_setting = miss_rate if isinstance(miss_rate, str) else miss_rate[0]  # gestione liste

    # Filtro sul CSV
    matched = df_prev_runs[
        (df_prev_runs['seed'] == seed) &
        (df_prev_runs['results_path'].str.contains(tumor_type)) &
        (df_prev_runs['modality_setting'] == modality_setting)
    ]

    if not matched.empty:
        # Prendi la prima occorrenza (nel caso ci siano più righe matching)
        checkpoint_path = matched.iloc[0]['results_path']
        combo['trainer']['checkpoint'] = checkpoint_path
        updated_combinations.append(combo)
    else:
        missing.append((tumor_type, seed, modality_setting))
print(f"Total matched combinations: {len(updated_combinations)}")
# Scrittura finale JSON aggiornato
output_file_updated = os.path.join(output_dir, f'{model_name}_versions.json')
with open(output_file_updated, 'w') as f:
    json.dump(updated_combinations, f, indent=4)

print(f"Updated combinations with checkpoints written to {output_file_updated}")

if missing:
    print(f"Missing combinations (no checkpoint found):")
    for m in missing:
        print(m)

# # Scrivi le combinazioni in un file JSON
# output_file = os.path.join(output_dir, f'{model_name}_versions.json')
# with open(output_file, 'w') as f:
#     json.dump(combinations, f, indent=4)

# print(f"Combinations written to {output_file}")




