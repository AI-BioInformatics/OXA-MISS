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
from dataloader.dataloader_utils import get_dataloaders, make_dataloader
from dataloader.clinical_tokens import ClinicalTokens
from dataloader.kfold import list_split_files, read_split, dataset_names_of, kfold_split_files, read_fold
from dataloader.build import build_dataset, clinical_tokens_config, fold_patients
from experiments.utils import import_class_from_path, ResultsStore
import collections.abc
import itertools
import inspect
import re
import warnings
import matplotlib
matplotlib.use("Agg")  # plots are only saved (SLURM nodes have no display)

# SLURM writes stdout to a file, where Python buffers print() in blocks: the dataset loading messages
# appeared only at the end of the run. Line buffering shows them as they happen.
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

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
    if seed is not None:                 # --seed overrides; otherwise the config's (e.g. a grid version's) seed
        config.seed = seed
    elif config.get('seed') is None:
        config.seed = 42
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
MODALITIES = ["WSI", "Genomics", "CNV", "CT", "MRI", "Clinical"]


def resolve_modalities(config):
    """data_loader.modalities -> {train: [...], val: [...], test: [[...], ...]} in place, and
    model.kwargs.input_modalities = train (the model is built on the training modalities).
      train: modalities used for training (default: model.kwargs.input_modalities, the legacy key)
      val:   modalities of the validation set, early stopping / checkpoint selection (default: train)
      test:  one modality set, or a list of sets: the model of each fold is tested on every set
             (default: train)
    val and test must be subsets of train: a modality never seen in training has an untrained encoder."""
    m = config.data_loader.get('modalities') or {}
    train = m.get('train') or config.model.kwargs.get('input_modalities')
    if not train:
        raise ValueError("No modalities: set data_loader.modalities.train (or model.kwargs.input_modalities)")
    val = m.get('val') or train
    test = m.get('test') or [train]
    if isinstance(test[0], str):
        test = [test]
    sets = {'train': [train], 'val': [val], 'test': test}
    for partition, modality_sets in sets.items():
        for modality_set in modality_sets:
            unknown = [x for x in modality_set if x not in MODALITIES]
            if unknown or len(set(modality_set)) != len(modality_set):
                raise ValueError(f"data_loader.modalities.{partition}: {modality_set} (unknown {unknown} or duplicates; known {MODALITIES})")
            if partition != 'train' and not set(modality_set) <= set(train):
                raise ValueError(f"data_loader.modalities.{partition} {modality_set} is not a subset of train {train}: "
                                 f"{sorted(set(modality_set) - set(train))} would go through untrained encoders")
    unique_test = []
    for modality_set in test:  # same set twice -> tested once
        if set(modality_set) not in [set(t) for t in unique_test]:
            unique_test.append(list(modality_set))
    legacy = config.model.kwargs.get('input_modalities')
    if legacy and m.get('train') and set(legacy) != set(train):
        logging.info(f"model.kwargs.input_modalities {legacy} replaced by data_loader.modalities.train {train}")
    config.model.kwargs.input_modalities = list(train)
    config.data_loader.modalities = munchify({'train': list(train), 'val': list(val), 'test': unique_test})
    return config


def modalities_title(modalities):
    """Run name part: the training modalities, then the val / test ones when they differ."""
    title = '+'.join(modalities.train)
    if set(modalities.val) != set(modalities.train):
        title += '_VAL_' + '+'.join(modalities.val)
    if len(modalities.test) > 1:  # the sets are in the config / wandb tags: a folder name has 255 chars at most
        title += f'_TEST_{len(modalities.test)}sets'
    elif set(modalities.test[0]) != set(modalities.train):
        title += '_TEST_' + '+'.join(modalities.test[0])
    return title


def set_clinical_normalization(mm, dataset, train_patients, config):
    """Age mean / std of the fold's training patients, stored in the model (self-contained checkpoint)."""
    if dataset.clinical_tokens is None or 'Clinical' not in config.model.kwargs.input_modalities:
        return
    net = getattr(mm.net, '_orig_mod', mm.net)  # torch.compile wrapper
    train_patients = [p for p in train_patients if p in dataset.patient_df.index]
    mean, std = dataset.clinical_tokens.num_stats(train_patients)
    net.set_clinical_normalization(mean, std, cat_mode=dataset.clinical_tokens.cat_modes(train_patients))
    logging.info(f"clinical numerical {dataset.clinical_tokens.token_names[:len(mean)]}: mean {mean}, std {std} (training patients)")


def set_reference_survival(mm, dataset, train_patients):
    """Censoring distribution of Uno's c-index / IBS: the fold's training patients."""
    if train_patients is None or "censorship" not in dataset.patient_df.columns:
        return
    rows = dataset.patient_df.loc[[p for p in train_patients if p in dataset.patient_df.index]]
    mm.set_reference_survival(rows["censorship"].values, rows["time"].values)


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
    arg_parser.add_argument("--seed", default=None, type=int, help="random seed (default: the config's seed, else 42)")        
    arg_parser.add_argument("--grid_search_model_version_index", type=int, default=None)  
    arg_parser.add_argument("--grid_search_versions", type=str, default=None,
                            help="versions json of the grid search (default: <trainer.grid_search_versions_path>/<model>_versions.json)")
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
        json_path = args.grid_search_versions or \
            os.path.join(config.trainer.grid_search_versions_path, config.model.name + "_versions.json")
        with open(json_path, 'r') as file:
            grid_search_version = json.load(file)[grid_search_model_version_index]

        config = munchify(recursive_update(config, grid_search_version))

    import_path = f"{repo_dir}/experiments/models/{config.model.name}.py"
    ModelClass = import_class_from_path(import_path, config.model.name)

    config = repair_config(config, args.seed)
    config = resolve_modalities(config)
    # a model with its own radiology encoder (attention pooling over the CT / MRI regions) reads the unpooled
    # token matrices; the others one mean-pooled vector per exam
    config.data_loader.radiology_tokens = 'num_ct_latent_queries' in inspect.signature(ModelClass.__init__).parameters
    # module B adversary active: the study must not be an input it is asked to unlearn (the pattern target too:
    # the missingness pattern differs by study, so the encoder would be pushed to hide the study token as well)
    if config.model.kwargs.get('shortcut_adversary_weight', 0) > 0 and config.data_loader.get('clinical', {}).get('use_study_id'):
        logging.info("shortcut adversary active: data_loader.clinical.use_study_id set to False (the study is not a clinical input)")
        config.data_loader.clinical.use_study_id = False

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
        paths = [config.data_loader.datasets_configs[0]]  # its kfold_splits are those of the new cohort
        if config.data_loader.get('KFold', {}).get('splits'):  # legacy splits in the main yaml
            config.data_loader.KFold.splits = swap(config.data_loader.KFold.splits)
            paths.append(config.data_loader.KFold.splits)
        if config.data_loader.get('missing_modality_table'):
            config.data_loader.missing_modality_table = swap(config.data_loader.missing_modality_table)
        for path in paths:
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
    # datasets and modalities in the run name: one grid can run several tumor / modality combinations
    datasets_str = '+'.join(n.replace('TCGA_', '') for n in dataset_names_of(config))
    modalities_str = f"{datasets_str}_{modalities_title(config.data_loader.modalities)}"
    # CT / MRI encoders (config/radiology.yaml) when the run uses that modality
    radiology_encoders = config.data_loader.get('radiology_encoders') or {}
    modalities_str += ''.join(f"_{m}-{radiology_encoders[m]}" for m in ('CT', 'MRI')
                              if m in radiology_encoders and m in config.model.kwargs.input_modalities)
    # clinical token columns (ablations of the clinical variables)
    clinical = clinical_tokens_config(config)
    if clinical and 'Clinical' in config.model.kwargs.input_modalities:
        columns = [c.replace('ajcc_', '') for c in clinical.get('categorical', ClinicalTokens.DEFAULT_CATEGORICAL)]
        modalities_str += '_CLIN-' + '+'.join(columns + (['study'] if clinical.get('use_study_id') else []))
    # short run name: start time + an ID unique per job (SLURM job / array task: two jobs submitted together
    # never overwrite each other, and the name points to the SLURM log; outside SLURM a random hash). The
    # description (title, datasets, modalities, encoders, clinical columns) goes to config.run_description
    # and to the wandb notes; the title is the wandb group and the results folder of the experiment.
    job_id = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    run_id = (f"{job_id}-{task_id}" if task_id else job_id) if job_id else timehash()
    config.run_description = f"{experiment_title}_{modalities_str}"
    config.title = f"{now:%Y-%m-%d_%H-%M-%S}_{run_id}"
    parent_directory = os.path.join(config.project_dir, experiment_title, config.title)
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
        notes=config.run_description,  # what the run is (the name is only time + ID)
        tags=[config.model.name] + dataset_names_of(config) + list(config.model.kwargs.input_modalities) +
             [f"test_{'+'.join(t)}" for t in config.data_loader.modalities.test] +
             list(config.wandb.get('tags') or []) + [f"seed_{config.seed}"],   # wandb.tags: e.g. ablation, module:<flag>
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
    dataset = build_dataset(config)
    
    config.data_loader.dataset_names = list(dataset.datasets)
    # study / cancer type indices of the module B adversary (OXA_MISS_final)
    if 'n_studies' in inspect.signature(ModelClass.__init__).parameters:
        config.model.kwargs.n_studies = len(dataset.study_names)
        config.model.kwargs.n_cancer_types = len(dataset.cancer_type_names)
        config.model.kwargs.n_sites = len(dataset.site_names)   # TCGA tissue source sites (adversary target "site")
        config.data_loader.cancer_type_names = list(dataset.cancer_type_names)
    # clinical as tokens (data_loader.clinical): only for models with a clinical tokenizer, and vice versa
    takes_clinical_tokens = 'clinical_cat_cardinalities' in inspect.signature(ModelClass.__init__).parameters
    if 'Clinical' in config.model.kwargs.input_modalities:
        if dataset.clinical_tokens is not None and not takes_clinical_tokens:
            raise ValueError(f"data_loader.clinical (clinical tokens) set, but {config.model.name} takes the clinical_path columns: remove it")
        if dataset.clinical_tokens is None and takes_clinical_tokens:
            raise ValueError(f"{config.model.name} takes clinical tokens: set data_loader.clinical in the config")
        if dataset.clinical_tokens is not None:
            config.model.kwargs.clinical_n_numerical = len(dataset.clinical_tokens.token_names) - len(dataset.clinical_tokens.cat_cols)
            config.model.kwargs.clinical_cat_cardinalities = list(dataset.clinical_tokens.cat_cardinalities)
            config.data_loader.clinical_token_names = list(dataset.clinical_tokens.token_names)
    if 'WSI' in config.model.kwargs.input_modalities and 'input_dim' in config.model.kwargs:
        # patch feature size from the features of each dataset: they must use the same WSI encoder
        wsi_dims = {}
        for name in dataset.datasets:
            if dataset.patient_df.loc[dataset.patient_df["dataset_name"] == name, "has_WSI"].any():
                wsi_dims[name] = dataset._wsi_feature_dim(name)
        if len(set(wsi_dims.values())) > 1:
            raise ValueError(f"WSI features of different size (different encoders?) across datasets: {wsi_dims}")
        if wsi_dims and config.model.kwargs.input_dim != next(iter(wsi_dims.values())):
            logging.info(f"model.kwargs.input_dim {config.model.kwargs.input_dim} -> {next(iter(wsi_dims.values()))} (WSI features)")
            config.model.kwargs.input_dim = next(iter(wsi_dims.values()))
    if 'Genomics' in config.model.kwargs.input_modalities:
        config.model.kwargs.genomics_group_input_dim = [dataset.genes_groups[key]['count'] for key in dataset.genomics_group_name]
    if 'CNV' in config.model.kwargs.input_modalities:
        config.model.kwargs.cnv_group_input_dim = [dataset.genes_groups[key]['count'] for key in dataset.cnv_group_name]
    # CT / MRI embedding sizes read from the features (same value as their zero placeholders)
    first_dataset = next(iter(dataset.datasets))
    config.model.kwargs.ct_emb_dim = dataset._placeholder_shape('CT', first_dataset)[1] if 'CT' in config.model.kwargs.input_modalities else None
    config.model.kwargs.mri_emb_dim = dataset._placeholder_shape('MRI', first_dataset)[1] if 'MRI' in config.model.kwargs.input_modalities else None
    if config.data_loader.task_type == "Survival":
        config.data_loader.survival_bins = [float(b) for b in dataset.bins]  # days: survival curves for IBS / D-cal
    # the saved config is the one actually run (input_dim, clinical tokens... come from the dataset)
    with open(copy_config_path, 'w') as config_file:
        yaml.dump(unmunchify(config), config_file, default_flow_style=False, sort_keys=True)
    wandb.config.update({"model": unmunchify(config.model), "data_loader": unmunchify(config.data_loader)},
                        allow_val_change=True)

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
        set_clinical_normalization(mm, dataset, train_patients, config)
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
        
        splits = kfold_split_files(config)
        # each test modality set, on the base test set or each missing modality scenario (the checkpoints
        # of each fold are <checkpoint dir>/model_*_Fold_<k>.pt: evaluate() adds the fold suffix)
        test_scenarios = config.missing_modality_test.scenarios if config.missing_modality_test.active else [None]
        log_on_telegram = False if not hasattr(config, 'log_on_telegram') else config.log_on_telegram
        for test_modalities, (i_scenario, scenario) in itertools.product(config.data_loader.modalities.test,
                                                                         enumerate(test_scenarios)):
            print(f'Testing the model on {test_modalities}' + (f' with missing modality scenario: {scenario}' if scenario else ''))
            for i, fold_files in enumerate(splits):
                foldname = f"Fold_{i+1}"
                logging.info(f'Fold {i+1}...')
                fold_train, fold_val, test_patients = read_fold(fold_files, dataset)
                reference = [p for p in (fold_train, fold_val) if p is not None]
                set_reference_survival(mm, dataset, np.concatenate(reference) if reference else None)
                test_dataloader = make_dataloader(dataset, "test", test_patients, test_modalities, config)
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
                            repo_path=repo_dir,
                            test_modalities=test_modalities)

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

        splits = kfold_split_files(config)

        for i, fold_files in enumerate(splits):            
            mm = None  # free the previous fold's model before building the next one
            torch.cuda.empty_cache()
            # Setup to be deterministic
            logging.info(f'setup to be deterministic')
            setup(config.seed)
            foldname = f"Fold_{i+1}"
            logging.info(f'Fold {i+1}...')
            for _, split_path in fold_files:
                if f"{i}.csv" not in split_path:
                    print('Loading split', split_path)
            train_patients, val_patients, test_patients = fold_patients(config, dataset, fold_files)
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
            set_clinical_normalization(mm, dataset, train_patients, config)
            set_reference_survival(mm, dataset, train_patients)
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
            # the model of the fold on every test modality set (the first one is test_dataloader), on the base
            # test set and each missing modality scenario
            test_scenarios = config.missing_modality_test.scenarios if config.missing_modality_test.active else []
            for i_test, test_modalities in enumerate(config.data_loader.modalities.test):
                if i_test > 0:
                    test_dataloader = make_dataloader(dataset, "test", test_patients, test_modalities, config)
                for i_scenario, scenario in enumerate([None] + list(test_scenarios)):
                    if scenario is not None:
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
                                print_demo_results = args.demo_training and scenario is not None and i_scenario==len(test_scenarios) and i==len(splits)-1,
                                eval_missing_modality_scenario = scenario,
                                repo_path=repo_dir,
                                test_modalities=test_modalities)

    end_time = time.time() 
    execution_time = end_time - start_time
    days = execution_time // (24 * 3600)
    hours = (execution_time % (24 * 3600)) // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    print(f"########################\n\n\nThe program took (DD--HH:MM:SS) {int(days)}-{int(hours)}:{int(minutes)}:{int(seconds)} to run.\n\n\n#################################")     
    wandb.finish()