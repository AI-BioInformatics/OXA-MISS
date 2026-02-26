import os, json
import argparse
from itertools import product


if __name__ == "__main__":
    #Prendi model name da riga di comando
    parser = argparse.ArgumentParser(description='Run grid search for model versions.')
    parser.add_argument('--model_name', type=str, required=True, default='MUSE', help='Name of the model (e.g., Custom_Multimodal_XA or MUSE).')
    # parser.add_argument('--encoder', type=str, required=True, default='UNI', help='Name of the encoder (e.g., resnet, CONCH, provgigapath).',choices=['resnet','UNI','CONCH','provgigapath'])
    model_name = parser.parse_args().model_name
    # encoder = parser.parse_args().encoder
    
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

    tumor_types = ["STAD"] #"BLCA","BRCA","COAD","HNSC","STAD"] #["LUAD","HNSC","STAD","BRCA", "BLCA","COAD","KIRC","KIRP","LUSC","OV"] #"OV" , #["OV"] ## ["PDS"] #, "NACT_primary"]#["BLCA", "BRCA", "COAD", "HNSC", "KIRC","KIRP" ,"LUAD", "LUSC","OV","STAD"] #"BRCA" manca solo seed 44 (running) "LUAD", "KIRP" devo sistemare i fold
    dataset= "TCGA"#'Decider'
    encoder_list=['resnet', 'UNI', 'CONCH', 'provgigapath']
    
    
    
    
    # Output combinazioni finali
    combinations = []
    for encoder in encoder_list:
        if encoder == "CONCH":
            input_dim = 768
        elif encoder == "provgigapath":
            input_dim = 1536
        else:
            input_dim = 1024

    # Per ogni tipo di tumore, genera un blocco changes con il nome già inserito
        for tumor in tumor_types:      
                # Usa le chiavi con dot notation per parametri annidati
            changes_template = {
                
                # --- W&B E GENERALI ---

                'title': ["{model_name}_{tumor_type}"], #custom_multimodal_XA_ING_MOD_RATE_kfold_5_all_tissues
                'project_dir': ["/work/H2020DeciderFicarra/fmiccolis/Frontiers/{encoder}_feats/{model_name}/{tumor_type}"],
                'seed': [43], #,
                
                'wandb.mode': ["online"],
                
                
                    # --- DATA LOADER ---
                
                    # --- ING MODALITY TABLES ---
                'data_loader.missing_modalities_tables.active': [False], #True,
                'data_loader.missing_modalities_tables.missing_mod_rate': ["complete"],#,"missing_all_30","missing_all_60"],#complete
                
                # ---OTHER DATA LOADER PARAMS ---
                'data_loader.file_genes_group' : ["genes_groups/pathways_ensg.json"],
                'data_loader.datasets_configs': [["/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/{dataset}_{tumor_type}_dataset_{encoder}.yaml"]],
                'data_loader.task_type': ["Survival"],
                'data_loader.max_patches': [4096],
                'data_loader.batch_size': [1],
                'data_loader.real_batch_size': [1],
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
                'data_loader.preprocessing': ["/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/preprocessing.yaml"],
                'data_loader.augmentation': ["/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/config/augmentation.yaml"],
                # --- K-FOLD SPLIT ---
                'data_loader.KFold.splits':["/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/splits/splits_train_val_test/{tumor_type}"], 
                'data_loader.KFold.internal_val_size': [0.0],

                    # --- MISSING MODALITY TEST ---
                'missing_modality_test.active': [False],
                'missing_modality_test.test_scenarios_on_each_epoch': [False],
                'missing_modality_test.scenarios': [[]],  # Lista vuota o nomi


                # --- MODEL ---
                'model.name': ["{model_name}"],
                'model.pretrained': [False],
                'model.save_checkpoints': [True],
                # --- MODEL.KWARGS ---
                'model.kwargs.input_dim': [input_dim],
                'model.kwargs.genomics_group_name': [["tumor_suppression","oncogenesis","protein_kinases","cellular_differentiation","cytokines_and_growth"]],
                'model.kwargs.genomics_group_dropout': [[0.35]],
                'model.kwargs.cnv_group_name': [["tumor_suppression","oncogenesis","protein_kinases","cellular_differentiation","cytokines_and_growth"]],
                'model.kwargs.cnv_group_dropout': [[0.2]],
                'model.kwargs.inner_dim': [128], #256, 512 da testare -- 128 prima
                'model.kwargs.num_latent_queries': [4], #2, 4 da testare -- 1 prima
                'model.kwargs.wsi_dropout': [0],
                'model.kwargs.use_layernorm': [False],
                'model.kwargs.dropout': [0.5],
                'model.kwargs.output_dim': [4],
                'model.kwargs.input_modalities': [["WSI"]],
                'model.kwargs.use_WSI_level_embs': [False],
                'model.kwargs.WSI_level_embs_fusion_type': ["concat"],
                'model.kwargs.WSI_level_encoder_dropout': [0.2],
                'model.kwargs.WSI_level_encoder_sizes': [[768, 40, 3]],
                'model.kwargs.WSI_level_encoder_LayerNorm': [False],

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
                'optimizer.name': ["AdoptAtan2"],
                'optimizer.weight_decay': [0.001], #0.001,0.0001,0.00001 da testare dopo
                'optimizer.momentum': [None],


                # --- TRAINER ---
                'trainer.reload': [False],
                'trainer.checkpoint': ['/work/H2020DeciderFicarra/fmiccolis/Frontiers/{encoder}_feats/{model_name}/{tumor_type}'],
                'trainer.do_train': [False],
                'trainer.do_test': [True],
                'trainer.do_inference': [False],
                'trainer.robust_training': [False] ,
                'trainer.do_kfold': [True],
                'trainer.epochs': [5],
                'trainer.patience': [2],
                'trainer.AEM_lamda': [0],
                'trainer.Save_XA_attention_files': [False],
            }
  
            changes = {}
            for key, val_list in changes_template.items():
                resolved = []
                for v in val_list:
                    current_tumor_str = tumor
                    if isinstance(v, str):
                        resolved.append(v.format(input_dim=input_dim, tumor_type=current_tumor_str,model_name=model_name,encoder=encoder))
                    elif isinstance(v, list) and all(isinstance(elem, str) for elem in v):
                        resolved.append([elem.format(input_dim=input_dim, tumor_type=current_tumor_str, dataset=dataset,model_name=model_name,encoder=encoder) for elem in v])
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
    output_dir = '/work/H2020DeciderFicarra/fmiccolis/Frontiers/OXA-MISS/grid_search/models_versions'
    os.makedirs(output_dir, exist_ok=True)
    # # Scrivi le combinazioni in un file JSON
    output_file = os.path.join(output_dir, f'{model_name}_versions.json')
    with open(output_file, 'w') as f:
        json.dump(combinations, f, indent=4)

    print(f"Combinations written to {output_file}")
    print(f"Total combinations: {len(combinations)}")




