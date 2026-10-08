"""K-fold splits: one split folder per dataset (parameters.kfold_splits of the dataset yaml); with several
datasets fold k concatenates the k-th split of each (used by main.py and the baseline scripts)."""
import logging
import os

import numpy as np
import pandas as pd
import yaml


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


def dataset_names_of(config):
    """Names of the dataset yamls of the run (the `name` key), e.g. ['CPTAC', 'TCGA_KIRC']."""
    names = []
    for path in config.data_loader.datasets_configs:
        with open(path) as f:
            names.append(yaml.load(f, yaml.FullLoader)['name'])
    return names


def kfold_split_files(config):
    """The split files of every fold: one list of (dataset name, file) per fold.
    Every dataset yaml has its own parameters.kfold_splits (a folder: every file in it, sorted; or a list
    of files) and fold k is the concatenation of the k-th file of every dataset, so the folds of a multi
    dataset run need no dedicated splits. Legacy: data_loader.KFold.splits, one split for all the datasets
    (dataset name None), has precedence when set."""
    legacy = config.data_loader.get('KFold', {}).get('splits')
    if legacy:
        logging.warning("data_loader.KFold.splits is deprecated: set parameters.kfold_splits in each dataset yaml")
        return [[(None, f)] for f in list_split_files(legacy)]
    per_dataset = {}
    for path in config.data_loader.datasets_configs:
        with open(path) as f:
            dataset_config = yaml.load(f, yaml.FullLoader)
        splits = dataset_config.get('parameters', {}).get('kfold_splits')
        if not splits:
            raise ValueError(f"{path}: no parameters.kfold_splits (and no data_loader.KFold.splits)")
        per_dataset[dataset_config['name']] = list_split_files(splits)
    n_folds = {name: len(files) for name, files in per_dataset.items()}
    if len(set(n_folds.values())) != 1:
        raise ValueError(f"The datasets have a different number of folds: {n_folds}")
    return [list(zip(per_dataset.keys(), files)) for files in zip(*per_dataset.values())]


def read_fold(fold_files, dataset):
    """(train, val, test) patient ids of a fold: the splits of its datasets concatenated (see read_split).
    A dataset split must contain only patients of its dataset, and a patient must be in one partition."""
    parts = {'train': [], 'val': [], 'test': []}
    for dataset_name, path in fold_files:
        split = dict(zip(parts, read_split(path)))
        if dataset_name is not None:
            ids = np.concatenate([v for v in split.values() if v is not None])
            ids = ids[np.isin(ids, dataset.patient_df.index)]
            other = ids[dataset.patient_df.loc[ids, 'dataset_name'].to_numpy() != dataset_name]
            if len(other):
                raise ValueError(f"{path} (splits of {dataset_name}) has {len(other)} patients of other datasets, e.g. {list(other[:5])}")
        for partition, patients in split.items():
            if patients is not None:
                parts[partition].append(patients)
    train, val, test = (np.concatenate(parts[k]) if parts[k] else None for k in parts)
    named = {'train': train, 'val': val, 'test': test}
    for a, b in [('train', 'val'), ('train', 'test'), ('val', 'test')]:
        if named[a] is not None and named[b] is not None:
            shared = np.intersect1d(named[a], named[b])
            if len(shared):
                raise ValueError(f"Fold {[f for _, f in fold_files]}: {len(shared)} patients in both {a} and {b}, e.g. {list(shared[:5])}")
    return train, val, test
