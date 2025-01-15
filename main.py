import sys
import os
import argparse
import logging
import logging.config
import shutil
import yaml
import random
import time
from datetime import date, datetime

import numpy as np
import pandas as pd
import torch
from torchvision import transforms

from hashlib import shake_256
from munch import munchify, unmunchify
import wandb
from torch.utils.data import Dataset, DataLoader, Subset, SequentialSampler, SubsetRandomSampler
from experiments.model_manager import ModelManager
from dataloader.dataloader_multidataset import WSI_Dataset
from dataloader.dataloader_utils import get_dataloaders
import gc
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
print("CUDA Device Count:", torch.cuda.device_count())
print("PyTorch CUDA Version:", torch.version.cuda)
print("CUDA Available:", torch.cuda.is_available())


# used to generate random names that will be appended to the
# experiment name
def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5)  # output len: 2*5=10
    return h.upper()


def setup(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # If using CUDA.
    torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # Ensure that you have not set torch.backends.cudnn.enabled = False


if __name__ == "__main__":
    wandb.require("core")
    start_time = time.time()
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    hostname = os.environ.get("HOSTNAME", 'unknown')
    logging.info(f"HOSTNAME: {hostname}")

    # Parse arguments
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("-c", "--config", required=True, type=str,
                            help="the config file to be used to run the experiment")
    arg_parser.add_argument("--verbose", action='store_true', help="Log also to stdout")
    arg_parser.add_argument("--debug", action='store_true', help="debug, no wandb")
    arg_parser.add_argument("--seed", default=42, type=int, help="Random Seed")        
    # arg_parser.add_argument("--AEM_lamda", default=-1, type=float, help="Attention Entropy Maximization regularization hyperparameter")                            
    args = arg_parser.parse_args()

    # check if the config files exists
    if not os.path.exists(args.config):
        logging.info("Config file does not exist: {}".format(args.config))
        raise SystemExit

    # Munchify the dict to access entries with both dot notation and ['name']
    logging.info(f'Loading the config file...')
    config = yaml.load(open(args.config, "r"), yaml.FullLoader)
    config = munchify(config)

    if args.seed != None:
        config.seed = args.seed

    for k, v in config.items():
        pad = ' '.join(['' for _ in range(25-len(k))])
        logging.info(f"{k}:{pad} {v}")


    # Setup to be deterministic
    logging.info(f'setup to be deterministic')
    setup(config.seed)

    if args.debug:
        os.environ['WANDB_DISABLED'] = 'true'


    wandb_name = f"{config.title}"
    # start wandb
    wandb.init(
        project="multimodal_decider",
        entity="multimodal_decider",
        name=wandb_name,
        config=unmunchify(config),
        mode=config.wandb.mode,
        settings=wandb.Settings(_service_wait=900)
    )

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
    checkpoint_model = os.path.join(parent_directory, 'checkpoint.pt')
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
    shutil.copy(args.config, copy_config_path)

    
    # transf_train = create_transforms(preprocessing, augmentation, config, eval=False, compose=True)
    # transf_eval = create_transforms(preprocessing, augmentation, config, eval=True, compose=True)
    # THE FOLLOWING TRANSFORMATIONS MUST BE CREATED ACCORDINGLY TO THE DATALOADER/TRANSFORMS.PY, PREPROCESSING, AUGMENTATIONS AND CONFIG(DATALOADER.NORMALIZE) YAML FILES
    # THE FOLLOWING IS A TOY DATASET 
    # MOST OF THE FOLLOWING INSTRUCTIONS MUST BE WRAPPED IN A DATALOADER CLASS
    
    # CREATE DATALOADERS
    dataset = WSI_Dataset(    
                            datasets_configs=config.data_loader.datasets_configs, 
                            task_type=config.data_loader.task_type,                           
                            max_patches=config.data_loader.max_patches,
                            n_bins=config.data_loader.n_bins,
                            eps=config.data_loader.eps,
                            sample=config.data_loader.sample,
                            load_slides_in_RAM=config.data_loader.load_slides_in_RAM,
                        )
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

    mm = ModelManager(config)

    if config.trainer.reload:
        if not os.path.exists(config.trainer.checkpoint):
            logging.error(f'Checkpoint file does not exist: {config.trainer.checkpoint}')
            raise SystemExit
        else:
            mm.load_checkpoint(config.trainer.checkpoint, device=config.model.device)
    # Train the model
    if config.trainer.do_train:
        logging.info('Training...')
        mm.train(train_dataloader, 
                    val_dataloader, 
                    test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    debug=args.debug, 
                    checkpoint=checkpoint_model, 
                    device=config.model.device, 
                    path=f"{parent_directory}", 
                    config=config)
        mm.evaluate(test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    checkpoint=checkpoint_model, 
                    best=True, 
                    device=config.model.device, 
                    path=f"{parent_directory}")

    # Test the model
    if config.trainer.do_test:
        logging.info('Testing the model...')
        mm.evaluate(test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    checkpoint=config.trainer.checkpoint, 
                    best=True, 
                    device=config.model.device, 
                    path=f"{parent_directory}")

    # Test the model
    if config.trainer.do_inference:
        logging.info('Inference...')
        mm.evaluate(test_dataloader, 
                    task_type=config.data_loader.task_type, 
                    checkpoint=config.trainer.checkpoint, 
                    device=config.model.device, 
                    path=f"{parent_directory}")

    if config.trainer.do_kfold:
        logging.info('K-Fold...')

        if type(config.data_loader.KFold.splits) is str:
            path_files = config.data_loader.KFold.splits
            lista_voci = os.listdir(path_files)
            splits = [f for f in lista_voci if os.path.isfile(os.path.join(path_files, f))]
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
            split_df = pd.read_csv(split_path)
            train_patients = split_df["train"].values.astype(str)
            if config.data_loader.KFold.internal_val_size > 0.0:
                np.random.shuffle(train_patients)
                val_patients = train_patients[:int(len(train_patients)*config.data_loader.KFold.internal_val_size)]
                train_patients = train_patients[int(len(train_patients)*config.data_loader.KFold.internal_val_size):]
                len_val = len(val_patients)
            else:
                val_patients = None
                len_val = 0
            test_patients = split_df["val"].dropna().values.astype(str)

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
            print("{} has Train: {}, Val: {}, Test: {} patients".format(foldname, len_train, len_val, len_test))
            
            
            if config.scheduler.name=="OneCycleLR":
                steps_per_epoch  = len(train_dataloader)
                config.scheduler["steps_per_epoch"]=steps_per_epoch
            mm = ModelManager(config)
            mm.train(train_dataloader, 
                        val_dataloader, 
                        test_dataloader, 
                        task_type=config.data_loader.task_type, 
                        checkpoint=checkpoint_model, 
                        device=config.model.device, 
                        path=f"{parent_directory}", 
                        kfold=foldname, 
                        config=config, 
                        debug=args.debug)
            if config.data_loader.KFold.internal_val_size > 0.0:
                best = True
            else:
                best = False
            mm.evaluate(test_dataloader, 
                        task_type=config.data_loader.task_type, 
                        checkpoint=checkpoint_model, 
                        best=best, 
                        device=config.model.device, 
                        path=f"{parent_directory}", 
                        kfold=foldname)
                

    end_time = time.time() 
    execution_time = end_time - start_time
    days = execution_time // (24 * 3600)
    hours = (execution_time % (24 * 3600)) // 3600
    minutes = (execution_time % 3600) // 60
    seconds = execution_time % 60
    print(f"########################\n\n\nThe program took (DD--HH:MM:SS) {int(days)}-{int(hours)}:{int(minutes)}:{int(seconds)} to run.\n\n\n#################################")     
    wandb.finish()