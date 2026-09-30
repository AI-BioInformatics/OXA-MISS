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
from experiments.model_manager import ModelManager
from dataloader.dataloader_multidataset import Multimodal_Bio_Dataset
from dataloader.dataloader_utils import get_dataloaders
from experiments.utils import import_class_from_path, ResultsStore
import collections.abc
import re
import warnings
import matplotlib
matplotlib.use("Agg")  # plots are only saved (SLURM nodes have no display)

# torch 2.0 internally still goes through TypedStorage when loading pickled models / tensors (torch.load):
# harmless, but printed at every checkpoint and slide load
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")



# os.environ["TORCH_USE_CUDA_DSA"] = "1"
print("CUDA Device Count: ", torch.cuda.device_count())
print("PyTorch CUDA Version: ", torch.version.cuda)
print("CUDA Available: ", torch.cuda.is_available())
print("CUDNN version: ", torch.backends.cudnn.version())
if torch.cuda.is_available():
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
def list_split_files(splits):
    """The split files of a KFold.splits entry: a folder (every file in it, sorted) or a list of files."""
    if isinstance(splits, str):
        return sorted(os.path.join(splits, f) for f in os.listdir(splits) if os.path.isfile(os.path.join(splits, f)))
    return splits


def read_split(split_path):
    """(train, val, test) patient ids of a split file; a missing or empty column is None and, without a
    test column, the val column is the test set. Columns are padded with NaN, so they are dropped."""
    df = pd.read_csv(split_path)
    column = lambda c: df[c].dropna().values.astype(str) if c in df.columns and df[c].notnull().any() else None
    train, val, test = column("train"), column("val"), column("test")
    if val is None and test is None:
        raise ValueError(f"Fold {split_path} has no test patients")
    if test is None:
        test, val = val, None
    return train, val, test


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
    arg_parser.add_argument("--demo_test",  action='store_true')          
    arg_parser.add_argument("--demo_training",  action='store_true')     
    args = arg_parser.parse_args()

    execution_dir = os.path.dirname(os.path.abspath(__file__))
    if os.path.basename(execution_dir) not in ["MultimodalDecider", "OXA-MISS"]:
        raise(ValueError('main.py must be executed from the original repo directory'))

    repo_dir = execution_dir

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

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

    import_path = f"{repo_dir}/experiments/models/{config.model.name}.py"
    ModelClass = import_class_from_path(import_path, config.model.name)

    config = repair_config(config, args.seed)

    if args.TRAINING_missing_mod_rate is not None:
        # se config.data_loader.missing_modalities_tables.active is True, then we can set the missing_mod_rate
        if not hasattr(config.data_loader, 'missing_modalities_tables') or not config.data_loader.missing_modalities_tables.active:
            raise ValueError("config.data_loader.missing_modalities_tables.active must be True to set TRAINING_missing_mod_rate")
        config.data_loader.missing_modalities_tables.missing_mod_rate = args.TRAINING_missing_mod_rate

    if args.TCGA_dataset_name is not None:
        # run the same config on another TCGA cohort: the cohort of the (single) dataset yaml, e.g.
        # config/TCGA_BLCA_dataset_UNI.yaml, is replaced in the dataset yaml, splits and missing table paths
        if len(config.data_loader.datasets_configs) != 1:
            raise ValueError("--TCGA_dataset_name needs exactly one dataset config")
        match = re.search(r"TCGA_([A-Z]+)_", os.path.basename(config.data_loader.datasets_configs[0]))
        if match is None:
            raise ValueError(f"--TCGA_dataset_name: no TCGA_<COHORT>_ in {config.data_loader.datasets_configs[0]}")
        old_cohort, new_cohort = match.group(1), args.TCGA_dataset_name.upper()
        swap = lambda path: path.replace(f"TCGA_{old_cohort}", f"TCGA_{new_cohort}")
        config.data_loader.datasets_configs[0] = swap(config.data_loader.datasets_configs[0])
        config.data_loader.KFold.splits = swap(config.data_loader.KFold.splits)
        if config.data_loader.get('missing_modality_table'):
            config.data_loader.missing_modality_table = swap(config.data_loader.missing_modality_table)
        for path in [config.data_loader.datasets_configs[0], config.data_loader.KFold.splits]:
            if not os.path.exists(path):
                raise ValueError(f"--TCGA_dataset_name {new_cohort}: {path} does not exist")
        config.title = f"{config.title}_{new_cohort}_{config.seed}"


    for k, v in config.items():
        pad = ' '.join(['' for _ in range(25-len(k))])
        logging.info(f"{k}:{pad} {v}")


    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'
        torch.autograd.set_detect_anomaly(True)  # slow: only for debugging

    # Check if project_dir exists
    if not os.path.exists(config.project_dir):
        os.makedirs(config.project_dir, exist_ok=True)
        # logging.error("Project_dir does not exist: {}".format(config.project_dir))
        # raise SystemExit

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
    # run name with the input modalities: the 31 modality combinations are otherwise indistinguishable
    experiment_title = config.title
    modalities_str = '+'.join(config.model.kwargs.input_modalities)
    config.title = f'{config.title}_{modalities_str}_YY{todays_date.year}-MM{str(todays_date.month).zfill(2)}-DD{str(todays_date.day).zfill(2)}-HH{now.hour:02}-MM{now.minute:02}_{timehash()}'
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
        # the logging.info calls above already created the default stderr handler ("INFO:root:..."):
        # replace it, otherwise every message is printed twice
        for handler in list(logger.handlers):
            if type(handler) is logging.StreamHandler:
                logger.removeHandler(handler)
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
        mode="disabled" if args.debug else config.wandb.mode,  # an explicit mode overrides WANDB_DISABLED
        group=experiment_title,  # runs of the same experiment (e.g. the modality combinations) grouped together
        tags=[config.model.name] + list(config.model.kwargs.input_modalities),
        settings=wandb.Settings(_service_wait=900)
    )
    # per-epoch curves (overall loss and c-index of each fold) are charts with the epoch as x-axis, so the
    # folds overlay; they are kept out of the run summary, which has only the results/* keys
    wandb.define_metric("Epoch", summary="none")
    for key in ["LR", "Train/*", "Valid/*", "Test/*"]:
        wandb.define_metric(key, step_metric="Epoch", summary="none")

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
    
    if 'Genomics' in config.model.kwargs.input_modalities:
        config.model.kwargs.genomics_group_input_dim = [dataset.genes_groups[key]['count'] for key in dataset.genomics_group_name]
    if 'CNV' in config.model.kwargs.input_modalities:
        config.model.kwargs.cnv_group_input_dim = [dataset.genes_groups[key]['count'] for key in dataset.cnv_group_name]
    # CT / MRI embedding sizes from the dataset (same value as its zero placeholders): mednet 2048, else CT 768 / MRI 320
    first_dataset = next(iter(dataset.datasets))
    config.model.kwargs.ct_emb_dim = dataset._placeholder_shape('CT', first_dataset)[1] if 'CT' in config.model.kwargs.input_modalities else None
    config.model.kwargs.mri_emb_dim = dataset._placeholder_shape('MRI', first_dataset)[1] if 'MRI' in config.model.kwargs.input_modalities else None

    # The random train/val/test split and the model below are used only by do_train / do_test / do_inference /
    # reload: a k-fold only run builds its own loaders and models for each fold, so they are skipped
    # (they cost dataloader workers, a model and a torch.compile, then were thrown away).
    mm = None
    if config.trainer.do_train or config.trainer.do_test or config.trainer.do_inference or config.trainer.reload:
        # GET INDICES FOR TRAIN, VALIDATION, AND TEST SETS
        train_patients, val_patients, test_patients = dataset.get_train_test_val_splits(
                                                                                    train_size=config.data_loader.train_size, 
                                                                                    val_size=config.data_loader.val_size, 
                                                                                    test_size=config.data_loader.test_size, 
                                                                                    random_state=config.data_loader.random_state
                                                                                    )
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
        
        splits = list_split_files(config.data_loader.KFold.splits)
        # the base test set, or each missing modality scenario (the checkpoints of each fold are
        # <checkpoint dir>/model_*_Fold_<k>.pt: evaluate() adds the fold suffix)
        test_scenarios = config.missing_modality_test.scenarios if config.missing_modality_test.active else [None]
        log_on_telegram = False if not hasattr(config, 'log_on_telegram') else config.log_on_telegram
        for i_scenario, scenario in enumerate(test_scenarios):
            if scenario is not None:
                print('Testing the model with missing modality scenario:', scenario)
            for i, split_path in enumerate(splits):
                foldname = f"Fold_{i+1}"
                logging.info(f'Fold {i+1}...')
                _, _, test_patients = read_split(split_path)
                _, _, test_dataloader = get_dataloaders(dataset=dataset, train_patients=None, val_patients=None,
                                                        test_patients=test_patients, config=config)
                mm.evaluate(test_dataloader,
                            task_type=config.data_loader.task_type,
                            checkpoint_last_epoch=os.path.join(config.trainer.checkpoint, 'model_last_epoch.pt'),
                            checkpoint_model_highest_metric=os.path.join(config.trainer.checkpoint, 'model_highest_metric.pt'),
                            checkpoint_model_lowest_loss=os.path.join(config.trainer.checkpoint, 'model_lowest_loss.pt'),
                            best=True,
                            device=config.model.device,
                            path=f"{parent_directory}",
                            kfold=foldname,
                            log_aggregated = i==len(splits)-1,
                            log_on_telegram = log_on_telegram,
                            Save_XA_attention_files = config.trainer.Save_XA_attention_files,
                            eval_missing_modality_scenario = scenario,
                            print_demo_results = args.demo_test and scenario is not None and i_scenario==len(test_scenarios)-1 and i==len(splits)-1,
                            is_demo_test=args.demo_test,
                            repo_path=repo_dir)

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
                    Save_XA_attention_files = config.trainer.Save_XA_attention_files,
                    repo_path=repo_dir)

    if config.trainer.do_kfold:
        logging.info('K-Fold...')

        splits = list_split_files(config.data_loader.KFold.splits)
        
        for i, split_path in enumerate(splits):            
            mm = None  # free the previous fold's model before building the next one
            torch.cuda.empty_cache()
            # Setup to be deterministic
            logging.info(f'setup to be deterministic')
            setup(config.seed)
            foldname = f"Fold_{i+1}"
            logging.info(f'Fold {i+1}...')
            if f"{i}.csv" not in split_path:
                print('Loading split', split_path.split('/')[-1])
            train_patients, val_patients, test_patients = read_split(split_path)
            if train_patients is None:
                raise ValueError(f"Fold {split_path} has no training patients")
            if config.data_loader.KFold.internal_val_size > 0.0:
                # validation set drawn from the training patients (a val column of the split joins them)
                if val_patients is not None:
                    train_patients = np.concatenate((train_patients, val_patients))
                np.random.shuffle(train_patients)
                n_val = int(len(train_patients)*config.data_loader.KFold.internal_val_size)
                val_patients, train_patients = train_patients[:n_val], train_patients[n_val:]
            len_val = len(val_patients) if val_patients is not None else 0
            
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
            # evaluate() tests the lowest-loss / highest-metric checkpoints when they exist (with a validation
            # set), and always the last epoch one
            best = True
            if val_patients is not None:
                len_val = len(val_dataloader.dataset)
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
                        best=True, 
                        device=config.model.device, 
                        path=f"{parent_directory}", 
                        kfold=foldname,
                        log_aggregated = i==len(splits)-1,
                        log_on_telegram = log_on_telegram,
                        Save_XA_attention_files = config.trainer.Save_XA_attention_files,
                        repo_path=repo_dir)
            
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
                                print_demo_results = args.demo_training and i_scenario==len(test_scenarios)-1 and i==len(splits)-1 ,
                                eval_missing_modality_scenario = scenario,
                                repo_path=repo_dir)

    end_time = time.time() 
    execution_time = end_time - start_time
    days = execution_time // (24 * 3600)
    hours = (execution_time % (24 * 3600)) // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    print(f"########################\n\n\nThe program took (DD--HH:MM:SS) {int(days)}-{int(hours)}:{int(minutes)}:{int(seconds)} to run.\n\n\n#################################")     
    wandb.finish()