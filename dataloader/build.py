"""Dataset and fold patients of a config, shared by main.py and the analysis scripts (utils/), so that they
build exactly the same data."""
import logging

import numpy as np
import yaml

from .dataloader_multidataset import Multimodal_Bio_Dataset
from .kfold import read_fold


CLINICAL_TOKENS_KEYS = ("table", "categorical", "use_study_id", "use_cancer_type", "excluded_studies")


def clinical_tokens_config(config):
    """data_loader.clinical -> arguments of dataloader.clinical_tokens.ClinicalTokens (None: the clinical_path
    columns of the dataset yamls, as used by OXA_MISS)."""
    clinical = config.data_loader.get('clinical')
    if not clinical:
        return None
    unknown = set(clinical) - set(CLINICAL_TOKENS_KEYS)
    if unknown:
        raise ValueError(f"data_loader.clinical: unknown keys {sorted(unknown)} (known {list(CLINICAL_TOKENS_KEYS)})")
    return {k: v for k, v in clinical.items() if v is not None}


def _dataset_name(path):
    with open(path) as f:
        return yaml.safe_load(f)['name']


def external_test_configs(config):
    """(yaml paths, names) of the external test datasets of the run: data_loader.external_test_configs, e.g. the
    CPTAC cohorts of a TCGA pan-cancer run. Each fold model is also tested on them (k-fold only); they are in no
    fold. An external dataset is skipped when the training datasets already contain data of the same source
    (the prefix of the name, e.g. CPTAC in CPTAC_CCRCC): a ccRCC run (CPTAC_CCRCC + TCGA_KIRC) gets no external
    CPTAC test."""
    paths = list(config.data_loader.get('external_test_configs') or [])
    if not paths:
        return [], []
    training_sources = {_dataset_name(p).split('_')[0] for p in config.data_loader.datasets_configs}
    kept, names = [], []
    for path in paths:
        name = _dataset_name(path)
        if name.split('_')[0] in training_sources:
            logging.info(f"external test {name} skipped: the training datasets contain {name.split('_')[0]} data")
            continue
        kept.append(path)
        names.append(name)
    return kept, names


def build_dataset(config):
    """The Multimodal_Bio_Dataset of a (resolved) config: modalities, clinical tokens, radiology encoders...
    The external test datasets (external_test_configs) are loaded too, flagged as external."""
    external_paths, external_names = external_test_configs(config)
    return Multimodal_Bio_Dataset(
                            datasets_configs=list(config.data_loader.datasets_configs) + external_paths,
                            external_datasets=external_names, 
                            task_type=config.data_loader.task_type,                           
                            max_patches=config.data_loader.max_patches,
                            n_bins=config.data_loader.n_bins,
                            eps=config.data_loader.eps,
                            sample=config.data_loader.sample,
                            load_slides_in_RAM=config.data_loader.load_slides_in_RAM,
                            slides_cache_dtype=config.data_loader.get('slides_cache_dtype', 'float32'),
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
                            radiology_encoders = config.data_loader.get('radiology_encoders'),
                            clinical_tokens = clinical_tokens_config(config),
                            radiology_tokens = bool(config.data_loader.get('radiology_tokens', False)),
                            radiology_token_filter = config.data_loader.get('radiology_token_filter', 'valid'),
                            radiology_token_meta = bool(config.data_loader.get('radiology_token_meta', False)),
                            wsi_representation = config.data_loader.get('wsi_representation', 'patches'),
                            wsi_gmm_dir = config.data_loader.get('wsi_gmm_dir'),
                        )


def fold_patients(config, dataset, fold_files):
    """(train, val, test) patients of a fold; with KFold.internal_val_size the validation set is drawn from the
    training patients (a val column of the split joins them) with the global numpy RNG: call it right after
    seeding (main.py: setup(config.seed) at the start of every fold)."""
    train_patients, val_patients, test_patients = read_fold(fold_files, dataset)
    if train_patients is None:
        raise ValueError(f"Fold {[f for _, f in fold_files]} has no training patients")
    if config.data_loader.KFold.internal_val_size > 0.0:
        if val_patients is not None:
            train_patients = np.concatenate((train_patients, val_patients))
        np.random.shuffle(train_patients)
        n_val = int(len(train_patients)*config.data_loader.KFold.internal_val_size)
        val_patients, train_patients = train_patients[:n_val], train_patients[n_val:]
    return train_patients, val_patients, test_patients
