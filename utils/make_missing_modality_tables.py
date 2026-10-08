"""
Missing modality tables of every dataset, for all the modalities (WSI, Genomics, CNV, CT, MRI, Clinical).

Columns (True = the modality is kept for the patient in that setting; one row per patient of the label file):
    complete                    every modality kept
    <mod>_miss_<r>              r% of the patients that HAVE <mod> lose it (r in 0, 25, 50, 75, 100)
    missing_all_<mod>_<r>       r% of the patients with >= 2 modalities lose a random non-empty subset of their
                                modalities, keeping at least one (r in 30, 60); the other patients keep all
Masks are sampled per dataset and per test fold (parameters.kfold_splits of the dataset yaml), so every fold
has the same rate. Availability is the union over the dataset yamls with the same name (WSI encoder variants,
e.g. CPTAC UNI / UNIv2), CT/MRI with the mednet features, Clinical from the clean clinical table: the tables
are the same for every encoder (the dataloader always ANDs them with what the patient really has).

Used by data_loader.missing_modalities_tables (training) and missing_modality_test.scenarios (test), through
parameters.missing_modalities_table_path of the dataset yamls (updated by --update_yamls).

Usage (reads the genomics of every dataset: run it as a SLURM job, not on the login node):
    python utils/make_missing_modality_tables.py [--out splits/missing_modality_tables] [--update_yamls]
"""
import argparse
import contextlib
import glob
import io
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataloader.dataloader_multidataset import Multimodal_Bio_Dataset  # noqa: E402

MODALITIES = ['WSI', 'Genomics', 'CNV', 'CT', 'MRI', 'Clinical']
MISS_RATES = [0, 25, 50, 75, 100]
MISSING_ALL_RATES = [30, 60]
SEED = 42
GENE_GROUPS = ["tumor_suppression", "oncogenesis", "protein_kinases", "cellular_differentiation", "cytokines_and_growth"]


def dataset_yamls():
    """dataset name -> its dataset yamls (encoder variants share the name, the label file and the splits)."""
    groups = {}
    for path in sorted(glob.glob(os.path.join(REPO, 'config', '*.yaml'))):
        with open(path) as f:
            cfg = yaml.safe_load(f)
        if isinstance(cfg, dict) and 'parameters' in cfg and 'name' in cfg and cfg['parameters'].get('kfold_splits'):
            groups.setdefault(cfg['name'], []).append(path)
    return groups


def availability(paths):
    """case_id -> has_<modality> (union over the yamls) for the patients of the label file."""
    has = None
    for path in paths:
        with contextlib.redirect_stdout(io.StringIO()):
            ds = Multimodal_Bio_Dataset(datasets_configs=[path], task_type='Survival', n_bins=4,
                                        file_genes_group=os.path.join(REPO, 'genes_groups/pathways_ensg.json'),
                                        genomics_group_name=GENE_GROUPS, cnv_group_name=GENE_GROUPS,
                                        input_modalities=MODALITIES, model_name='OXA_MISS',
                                        radiology_encoders={'CT': 'mednet', 'MRI': 'mednet'}, clinical_tokens={})
        h = ds.patient_df[[f'has_{m}' for m in MODALITIES]].astype(bool)
        h.columns = MODALITIES
        has = h if has is None else has.reindex(has.index.union(h.index), fill_value=False) | \
            h.reindex(has.index.union(h.index), fill_value=False)
    return has


def test_folds(split_dir, patients):
    """case_id -> test fold (splits_<k>.csv of the dataset), -1 if in no test column."""
    fold = pd.Series(-1, index=patients)
    for k, f in enumerate(sorted(glob.glob(os.path.join(split_dir, '*.csv')))):
        test = pd.read_csv(f)['test'].dropna().astype(str)
        fold[fold.index.isin(test)] = k
    return fold


def make_table(has, fold, rng):
    modalities = [m for m in MODALITIES if has[m].any()]
    table = pd.DataFrame(index=has.index)
    table['complete'] = True
    for m in modalities:
        for r in MISS_RATES:
            keep = pd.Series(True, index=has.index)
            for k in sorted(fold.unique()):
                owners = has.index[(fold == k) & has[m]]
                n_drop = int(round(r / 100 * len(owners)))
                keep[rng.choice(owners, size=n_drop, replace=False)] = False
            table[f'{m.lower()}_miss_{r}'] = keep
    for r in MISSING_ALL_RATES:
        keep = pd.DataFrame(True, index=has.index, columns=modalities)
        n_available = has[modalities].sum(axis=1)
        for k in sorted(fold.unique()):
            candidates = has.index[(fold == k) & (n_available >= 2)]
            for pid in rng.choice(candidates, size=int(round(r / 100 * len(candidates))), replace=False):
                own = [m for m in modalities if has.loc[pid, m]]
                kept = rng.choice(own, size=rng.integers(1, len(own)), replace=False)   # 1 .. len-1 kept
                keep.loc[pid, [m for m in own if m not in kept]] = False
        for m in modalities:
            table[f'missing_all_{m.lower()}_{r}'] = keep[m]
    return table


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'splits', 'missing_modality_tables'))
    ap.add_argument('--update_yamls', action='store_true', help='set parameters.missing_modalities_table_path')
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)
    for name, paths in dataset_yamls().items():
        with open(paths[0]) as f:
            split_dir = yaml.safe_load(f)['parameters']['kfold_splits']
        has = availability(paths)
        table = make_table(has, test_folds(split_dir, has.index), rng)
        out = os.path.join(args.out, f'{name}_missing_modality_table.csv')
        table.rename_axis('case_id').reset_index().to_csv(out, index=False)
        summary = {m: int(has[m].sum()) for m in MODALITIES if has[m].any()}
        print(f'{name:14s} {len(table)} patients, with modality {summary} -> {out}')
        if args.update_yamls:
            for path in paths:
                text = open(path).read()
                line = f'  missing_modalities_table_path: {out}'
                if re.search(r'^  missing_modalities_table_path:.*$', text, flags=re.M):
                    text = re.sub(r'^  missing_modalities_table_path:.*$', line, text, flags=re.M)
                else:
                    text = text.rstrip('\n') + '\n' + line + '\n'
                open(path, 'w').write(text)


if __name__ == '__main__':
    main()
