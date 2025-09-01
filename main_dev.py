import sys
import os
import argparse
import logging
import yaml
import random
import json
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch

from hashlib import shake_256
from munch import munchify, unmunchify
import wandb
from experiments.model_manager_dev import ModelManager
from dataloader.dataloader_multidataset import Multimodal_Bio_Dataset
from dataloader.dataloader_utils import get_dataloaders
from experiments.utils import import_class_from_path, ResultsStore
import collections.abc


os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# os.environ["TORCH_USE_CUDA_DSA"] = "1"
print("CUDA Device Count: ", torch.cuda.device_count())
print("PyTorch CUDA Version: ", torch.version.cuda)
print("CUDA Available: ", torch.cuda.is_available())
print("CUDNN version: ", torch.backends.cudnn.version())
print("Device Name: ", torch.cuda.get_device_name(0))

# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5)  # output len: 2*5=10
    return h.upper()

def repair_config(config, seed):
    if seed is not None:
        config.seed = seed
    if not hasattr(config.trainer, 'Save_XA_attention_files'):
        config.trainer.Save_XA_attention_files = False
    if not hasattr(config.model.kwargs, 'use_WSI_level_embs') and config.model.name.startswith('Custom_Multimodal_XA'):
        config.model.kwargs.use_WSI_level_embs = False
    return config

def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA.
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    # torch.use_deterministic_algorithms(True)
    # torch.set_float32_matmul_precision('high')
    # torch.backends.cuda.matmul.allow_tf32 = False
    # torch.backends.cudnn.allow_tf32 = False
    # 1) balanced reproducibility - good performance tradeoff
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # 2) Guaranteed reproducibility - major performance drop
    # torch.backends.cudnn.enabled = False
    # Ensure that you have not set torch.backends.cudnn.enabled = False
    print("torch.backends.cudnn.benchmark:", torch.backends.cudnn.benchmark)
    print("torch.backends.cudnn.deterministic:", torch.backends.cudnn.deterministic)
    print("torch.backends.cudnn.enabled:", torch.backends.cudnn.enabled)

def recursive_update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = recursive_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d
if __name__ == "__main__":
    wandb.require("core")
    start_time = time.time()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    hostname = os.environ.get("HOSTNAME", 'unknown')
    logging.info(f"HOSTNAME: {hostname}")

    results_store = ResultsStore()

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", required=True, type=str,
                            help="the config file to be used to run the experiment")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    arg_parser.add_argument("--seed", default=42, type=int, help="Random Seed")        
    arg_parser.add_argument("--grid_search_model_version_index", type=int, default=None)  
    arg_parser.add_argument("--TCGA_dataset_name", type=str, default=None)     
    arg_parser.add_argument("--TRAINING_missing_mod_rate", type=str, default=None)     
    arg_parser.add_argument("--demo",  action='store_true')     
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    import_path = f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/experiments/models/{config.model.name}.py"
    ModelClass = import_class_from_path(import_path, config.model.name)

    config = repair_config(config, args.seed)

    if args.TRAINING_missing_mod_rate is not None:
        # se config.data_loader.missing_modalities_tables.active is True, then we can set the missing_mod_rate
        if not hasattr(config.data_loader, 'missing_modalities_tables') or not config.data_loader.missing_modalities_tables.active:
            raise ValueError("config.data_loader.missing_modalities_tables.active must be True to set TRAINING_missing_mod_rate")
        config.data_loader.missing_modalities_tables.missing_mod_rate = args.TRAINING_missing_mod_rate

    
    if args.grid_search_model_version_index is not None:
        grid_search_model_version_index = args.grid_search_model_version_index
        if not isinstance(grid_search_model_version_index, int):
            raise ValueError("grid_search_model_version_index must be an integer")
        if grid_search_model_version_index < 0:  
            raise ValueError("grid_search_model_version_index must be a positive integer")
        config.grid_search_model_version_index = grid_search_model_version_index
        json_path = os.path.join(config.trainer.grid_search_versions_path, config.model.name + "_versions.json")
        with open(json_path, 'r') as file:
            grid_search_version = json.load(file)[grid_search_model_version_index]

        config = munchify(recursive_update(config, grid_search_version))

    if args.TCGA_dataset_name is not None and 'BRCA' in config.data_loader.KFold.splits:
        if len(config.data_loader.datasets_configs) == 1 and 'BRCA' in config.data_loader.datasets_configs[0]:
            config.data_loader.datasets_configs[0] = config.data_loader.datasets_configs[0].replace("BRCA", args.TCGA_dataset_name)
            config.data_loader.KFold.splits = config.data_loader.KFold.splits.replace("BRCA", args.TCGA_dataset_name)
            config.title = f"{config.title}_{args.TCGA_dataset_name}_{config.seed}"
            config.data_loader.missing_modality_table = config.data_loader.missing_modality_table.replace("BRCA", args.TCGA_dataset_name)


    for k, v in config.items():
        pad = ' '.join(['' for _ in range(25-len(k))])
        logging.info(f"{k}:{pad} {v}")


    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        logging.error("Project_dir does not exist: {}".format(config.project_dir))
        raise SystemExit

    # check if preprocessing is set and file exists
    logging.info(f'loading preprocessing')
    if config.data_loader.preprocessing is None:
        preprocessing = []
    elif not os.path.exists(config.data_loader.preprocessing):
        logging.error("Preprocessing file does not exist: {}".format(config.data_loader.preprocessing))
        preprocessing = []
    else:
        with open(config.data_loader.preprocessing, 'r') as preprocessing_file:
            preprocessing = yaml.load(preprocessing_file, yaml.FullLoader)
            preprocessing = munchify(preprocessing)

    # check if augmentation is set and file exists
    logging.info(f'loading augmentation')
    if config.data_loader.augmentation is None:
        augmentation = []
    elif not os.path.exists(config.data_loader.augmentation):
        logging.error("augmentation file does not exist: {}".format(config.data_loader.augmentation))
        augmentation = []
    else:
        with open(config.data_loader.augmentation, 'r') as augmentation_file:
            augmentation = yaml.load(augmentation_file, yaml.FullLoader)
            augmentation = munchify(augmentation)
    # make title unique to avoid overriding
    todays_date = date.today()
    now = datetime.now()
    config.title = f'{config.title}_YY{todays_date.year}-MM{str(todays_date.month).zfill(2)}-DD{str(todays_date.day).zfill(2)}-HH{now.hour:02}-MM{now.minute:02}_{timehash()}'
    parent_directory = os.path.join(config.project_dir, config.title)
    config.parent_directory = parent_directory
    checkpoint_last_epoch = os.path.join(parent_directory, 'model_last_epoch.pt')
    checkpoint_model_lowest_loss = os.path.join(parent_directory, 'model_lowest_loss.pt')
    checkpoint_model_highest_metric = os.path.join(parent_directory, 'model_highest_metric.pt')
    os.makedirs(parent_directory, exist_ok=True)
    logging.info(f'project directory: {parent_directory}')

    # Setup logger's handlers
    file_handler = logging.FileHandler(os.path.join(parent_directory, 'output.log'))
    log_format = logging.Formatter('%(asctime)s:%(levelname)s:%(message)s')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)

    if args.verbose:
        stdout_handler = logging.StreamHandler(sys.stdout)
        stdout_handler.setFormatter(log_format)
        logger.addHandler(stdout_handler)

    # Copy config file to project_dir, to be able to reproduce the experiment
    copy_config_path = os.path.join(parent_directory, 'config.yaml')
    
    # Dump the modified config to the copied config file
    with open(copy_config_path, 'w') as config_file:
        yaml.dump(unmunchify(config), config_file, default_flow_style=False, sort_keys=True)

    wandb_name = f"{config.title}"
    # start wandb
    wandb.init(
        project=config.wandb.project if hasattr(config.wandb,'project') else "multimodal_decider",
        entity="multimodal_decider",
        name=wandb_name,
        config=unmunchify(config),
        mode=config.wandb.mode,
        settings=wandb.Settings(_service_wait=900)
    )

    # THE FOLLOWING TRANSFORMATIONS MUST BE CREATED ACCORDINGLY TO THE DATALOADER/TRANSFORMS.PY, PREPROCESSING, AUGMENTATIONS AND CONFIG(DATALOADER.NORMALIZE) YAML FILES
    # THE FOLLOWING IS A TOY DATASET 
    # MOST OF THE FOLLOWING INSTRUCTIONS MUST BE WRAPPED IN A DATALOADER CLASS
    
    # CREATE DATALOADERS
    dataset = Multimodal_Bio_Dataset(    
                            datasets_configs=config.data_loader.datasets_configs, 
                            task_type=config.data_loader.task_type,                           
                            max_patches=config.data_loader.max_patches,
                            n_bins=config.data_loader.n_bins,
                            eps=config.data_loader.eps,
                            sample=config.data_loader.sample,
                            load_slides_in_RAM=config.data_loader.load_slides_in_RAM,
                            file_genes_group=config.data_loader.file_genes_group,
                            genomics_group_name=config.model.kwargs.genomics_group_name if hasattr(config.model.kwargs, 'genomics_group_name') else None,
                            cnv_group_name=config.model.kwargs.cnv_group_name if hasattr(config.model.kwargs, 'cnv_group_name') else None,
                            use_WSI_level_embs=config.model.kwargs.use_WSI_level_embs if hasattr(config.model.kwargs, 'use_WSI_level_embs') else None,
                            use_missing_modalities_tables=config.data_loader.missing_modalities_tables.active if hasattr(config.data_loader, 'missing_modalities_tables') else False,
                            missing_mod_rate=config.data_loader.missing_modalities_tables.missing_mod_rate if hasattr(config.data_loader, 'missing_modalities_tables') else None,
 
                            missing_modality_test_scenarios=config.missing_modality_test.scenarios if hasattr(config, 'missing_modality_test') and config.missing_modality_test.active else [],
                            input_modalities = config.model.kwargs.input_modalities,
                            missing_modality_table = config.data_loader.missing_modality_table if hasattr(config.data_loader, 'missing_modality_table') else None,
                            model_name = config.model.name if hasattr(config.model, 'name') else None,
                        )
    
     # GET INDICES FOR TRAIN, VALIDATION, AND TEST SETS
    train_patients, val_patients, test_patients = dataset.get_train_test_val_splits(
                                                                                train_size=config.data_loader.train_size, 
                                                                                val_size=config.data_loader.val_size, 
                                                                                test_size=config.data_loader.test_size, 
                                                                                random_state=config.data_loader.random_state
                                                                                )
    
    
    if 'Genomics' in config.model.kwargs.input_modalities:
        config.model.kwargs.genomics_group_input_dim = [dataset.genes_groups[key]['count'] for key in dataset.genomics_group_name]
    if 'CNV' in config.model.kwargs.input_modalities:
        config.model.kwargs.cnv_group_input_dim = [dataset.genes_groups[key]['count'] for key in dataset.cnv_group_name]
    
    
    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(            
                                                                        dataset=dataset,
                                                                        train_patients=train_patients, 
                                                                        val_patients=val_patients, 
                                                                        test_patients=test_patients,
                                                                        config=config
                                                                        )

    if config.scheduler.name=="OneCycleLR":
        steps_per_epoch  = len(train_dataloader)
        config.scheduler["steps_per_epoch"]=steps_per_epoch

    mm = ModelManager(config, ModelClass, results_store)
    mm.net = torch.compile(mm.net)
    if config.trainer.reload:
        if not os.path.exists(config.trainer.checkpoint):
            logging.error(f'Checkpoint file does not exist: {config.trainer.checkpoint}')
            raise SystemExit
        else:
            try:
                mm.load_checkpoint(config.trainer.checkpoint, device=config.model.device)
            except:
                mm.load_checkpoint(checkpoint_model_lowest_loss, device=config.model.device)
    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        # GET INDICES FOR TRAIN, VALIDATION, AND TEST SETS
        train_patients, val_patients, test_patients = dataset.get_train_test_val_splits(
                                                                                train_size=config.data_loader.train_size, 
                                                                                val_size=config.data_loader.val_size, 
                                                                                test_size=config.data_loader.test_size, 
                                                                                random_state=config.data_loader.random_state
                                                                                )

        if "Genomics" in config.model.kwargs.input_modalities:
            dataset.normalize_genomics(train_patients, val_patients, test_patients)
        if "CNV" in config.model.kwargs.input_modalities:
            dataset.normalize_cnv(train_patients, val_patients, test_patients)
        train_dataloader, val_dataloader, test_dataloader = get_dataloaders(            
                                                                            dataset=dataset,
                                                                            train_patients=train_patients, 
                                                                            val_patients=val_patients, 
                                                                            test_patients=test_patients,
                                                                            config=config
                                                                            )

        mm.train(train_dataloader, 
                    val_dataloader, 
                    test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    debug=args.debug, 
                    checkpoint_last_epoch=checkpoint_last_epoch, 
                    checkpoint_model_highest_metric=checkpoint_model_highest_metric,
                    checkpoint_model_lowest_loss=checkpoint_model_lowest_loss,
                    device=config.model.device, 
                    path=f"{parent_directory}", 
                    config=config)
        mm.evaluate(test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    checkpoint=checkpoint_last_epoch, 
                    best=True, 
                    device=config.model.device, 
                    path=f"{parent_directory}",
                    Save_XA_attention_files = config.trainer.Save_XA_attention_files)

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        
        if type(config.data_loader.KFold.splits) is str:
            path_files = config.data_loader.KFold.splits
            lista_voci = os.listdir(path_files)
            splits = sorted([os.path.join(path_files,f) for f in lista_voci if os.path.isfile(os.path.join(path_files, f))])
        else:
            splits = config.data_loader.KFold.splits

        if config.missing_modality_test.active:
            test_scenarios = config.missing_modality_test.scenarios
            for i_scenario, scenario in enumerate(test_scenarios):   
                print('Testing the model with missing modality scenario:', scenario)
                for i, split_path in enumerate(splits):            
                    foldname = f"Fold_{i+1}"
                    logging.info(f'Fold {i+1}...')
                    split_df = pd.read_csv(split_path)
                    has_val = "val" in split_df.columns and split_df['val'].notnull().any()
                    has_test = "test" in split_df.columns and split_df['test'].notnull().any()
                    if not has_val and not has_test:
                        raise ValueError(f"Fold {split_path} has no test patients")
                    if has_test:
                        test_patients = split_df["test"].dropna().values.astype(str)
                    else:
                        test_patients = split_df["val"].dropna().values.astype(str)

                    train_dataloader, val_dataloader, test_dataloader = get_dataloaders(    
                                                                                    dataset=dataset,
                                                                                    train_patients=None, 
                                                                                    val_patients=None, 
                                                                                    test_patients=test_patients,
                                                                                    config=config
                                                                                )

                    checkpoint_last_epoch=os.path.join(config.trainer.checkpoint,f'model_last_epoch.pt')
                    checkpoint_model_lowest_loss = os.path.join(config.trainer.checkpoint, f'model_lowest_loss.pt')
                    checkpoint_model_highest_metric = os.path.join(config.trainer.checkpoint, f'model_highest_metric.pt') 
                    log_on_telegram = False if not hasattr(config, 'log_on_telegram') else config.log_on_telegram

                    mm.evaluate(test_dataloader, 
                                task_type=config.data_loader.task_type, 
                                checkpoint_last_epoch=checkpoint_last_epoch, 
                                checkpoint_model_highest_metric=checkpoint_model_highest_metric,
                                checkpoint_model_lowest_loss=checkpoint_model_lowest_loss,
                                best=True, 
                                device=config.model.device, 
                                path=f"{parent_directory}", 
                                kfold=foldname,
                                log_aggregated = i==len(splits)-1,
                                log_on_telegram = log_on_telegram,
                                Save_XA_attention_files = config.trainer.Save_XA_attention_files,
                                eval_missing_modality_scenario = scenario,
                                print_demo_results = args.demo and i_scenario==len(test_scenarios)-1 and i==len(splits)-1 ,
                                is_demo=args.demo)

    # Test the model
    if config.trainer.do_inference:
        logging.info('Inference...')
        mm.evaluate(test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    checkpoint_last_epoch=checkpoint_last_epoch, 
                    checkpoint_model_highest_metric=checkpoint_model_highest_metric,
                    checkpoint_model_lowest_loss=checkpoint_model_lowest_loss,
                    device=config.model.device, 
                    path=f"{parent_directory}",
                    Save_XA_attention_files = config.trainer.Save_XA_attention_files)

    if config.trainer.do_kfold:
        logging.info('K-Fold...')

        if type(config.data_loader.KFold.splits) is str:
            path_files = config.data_loader.KFold.splits
            lista_voci = os.listdir(path_files)
            splits = sorted([os.path.join(path_files,f) for f in lista_voci if os.path.isfile(os.path.join(path_files, f))])
        else:
            splits = config.data_loader.KFold.splits

        
        for i, split_path in enumerate(splits):            
            del mm
            torch.cuda.empty_cache()
            # Setup to be deterministic
            logging.info(f'setup to be deterministic')
            setup(config.seed)
            foldname = f"Fold_{i+1}"
            logging.info(f'Fold {i+1}...')
            if f"{i}.csv" not in split_path:
                print('Loading split', split_path.split('/')[-1])
            split_df = pd.read_csv(split_path)
                

            
            has_val = "val" in split_df.columns and split_df['val'].notnull().any()
            has_test = "test" in split_df.columns and split_df['test'].notnull().any()
            if 'train' not in split_df.columns or not split_df['train'].notnull().any():
                raise ValueError(f"Fold {split_path} has no training patients")

            if not has_val and not has_test:
                raise ValueError(f"Fold {split_path} has no test patients")
            train_patients = split_df["train"].values.astype(str)

            if config.data_loader.KFold.internal_val_size > 0.0:
                if has_val and has_test:
                    # unisco la colonna di val e train
                    val_patients = split_df["val"].dropna().values.astype(str)
                    train_patients = np.concatenate((train_patients, val_patients))
                elif has_val:
                    test_patients = split_df["val"].dropna().values.astype(str)
                else:
                    test_patients = split_df["test"].dropna().values.astype(str)
                np.random.shuffle(train_patients)
                val_patients = train_patients[:int(len(train_patients)*config.data_loader.KFold.internal_val_size)]
                train_patients = train_patients[int(len(train_patients)*config.data_loader.KFold.internal_val_size):]
                len_val = len(val_patients)
            else:
                if has_val and has_test:
                    val_patients = split_df["val"].dropna().values.astype(str)
                    test_patients = split_df["test"].dropna().values.astype(str)
                    len_val = len(val_patients)
                elif has_val:
                    test_patients = split_df["val"].dropna().values.astype(str)
                    val_patients = None
                    len_val = 0
                else:
                    test_patients = split_df["test"].dropna().values.astype(str)
                    val_patients = None
                    len_val = 0
                
            
            
            if "Genomics" in config.model.kwargs.input_modalities:
                dataset.normalize_genomics(train_patients, val_patients, test_patients)
            if "CNV" in config.model.kwargs.input_modalities:
                dataset.normalize_cnv(train_patients, val_patients, test_patients)
            train_dataloader, val_dataloader, test_dataloader = get_dataloaders(    
                                                                                    dataset=dataset,
                                                                                    train_patients=train_patients, 
                                                                                    val_patients=val_patients, 
                                                                                    test_patients=test_patients,
                                                                                    config=config
                                                                                )
            len_train = len(train_dataloader.dataset)
            len_test = len(test_dataloader.dataset)
            if val_patients is not None:
                len_val = len(val_dataloader.dataset)
                best = True
            print("{} has Train: {}, Val: {}, Test: {} patients".format(foldname, len_train, len_val, len_test))
            
            
            if config.scheduler.name=="OneCycleLR":
                steps_per_epoch  = len(train_dataloader)
                config.scheduler["steps_per_epoch"]=steps_per_epoch
            mm = ModelManager(config, ModelClass, results_store)
            mm.train(train_dataloader, 
                        val_dataloader, 
                        test_dataloader, 
                        task_type=config.data_loader.task_type, 
                        checkpoint_last_epoch=checkpoint_last_epoch, 
                        checkpoint_model_highest_metric=checkpoint_model_highest_metric,
                        checkpoint_model_lowest_loss=checkpoint_model_lowest_loss,
                        device=config.model.device, 
                        path=f"{parent_directory}", 
                        kfold=foldname, 
                        config=config, 
                        debug=args.debug)
            
            log_on_telegram = True if not hasattr(config, 'log_on_telegram') else config.log_on_telegram
            mm.evaluate(test_dataloader, 
                        task_type=config.data_loader.task_type, 
                        checkpoint_last_epoch=checkpoint_last_epoch, 
                        checkpoint_model_highest_metric=checkpoint_model_highest_metric,
                        checkpoint_model_lowest_loss=checkpoint_model_lowest_loss,
                        best=best, 
                        device=config.model.device, 
                        path=f"{parent_directory}", 
                        kfold=foldname,
                        log_aggregated = i==len(splits)-1,
                        log_on_telegram = log_on_telegram,
                        Save_XA_attention_files = config.trainer.Save_XA_attention_files,)
            
            if config.missing_modality_test.active:
                test_scenarios = config.missing_modality_test.scenarios
                for i_scenario, scenario in enumerate(test_scenarios):
                    print('Testing the model with missing modality scenario:', scenario)
                    

                    mm.evaluate(test_dataloader, 
                                task_type=config.data_loader.task_type, 
                                checkpoint_last_epoch=checkpoint_last_epoch, 
                                checkpoint_model_highest_metric=checkpoint_model_highest_metric,
                                checkpoint_model_lowest_loss=checkpoint_model_lowest_loss,
                                best=best, 
                                device=config.model.device, 
                                path=f"{parent_directory}", 
                                kfold=foldname,
                                log_aggregated = i==len(splits)-1,
                                log_on_telegram = log_on_telegram,
                                Save_XA_attention_files = config.trainer.Save_XA_attention_files,
                                eval_missing_modality_scenario = scenario,)

    end_time = time.time() 
    execution_time = end_time - start_time
    days = execution_time // (24 * 3600)
    hours = (execution_time % (24 * 3600)) // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    print(f"########################\n\n\nThe program took (DD--HH:MM:SS) {int(days)}-{int(hours)}:{int(minutes)}:{int(seconds)} to run.\n\n\n#################################")     
    wandb.finish()