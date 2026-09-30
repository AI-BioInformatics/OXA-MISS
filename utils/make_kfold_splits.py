"""
Patient-level 5-fold splits for multimodal experiments, built from the updated OS label files.

Every modality combination selects a different subset of patients (those with at least one of its
modalities), so balancing the folds only on the OS event is not enough: e.g. the few CPTAC patients with
CT/MRI could end up with no events in a test fold. The folds are built greedily so that, for every
modality combination (WSI, Genomics, CNV, CT, MRI, Clinical), the patients of that combination are
balanced across folds per dataset and per OS event (see stratified_folds). Rows of the label files are slides: the unit is the patient,
so all the slides of a patient are in the same fold.

Outputs (columns: train, test; fold k is the test set of splits_k):
    <out>/<DATASET>/splits_{0..4}.csv      one folder per label file in dataloader/dataset_updated/
    <ccrcc_out>/splits_{0..4}.csv          combined ccRCC cohort (CPTAC + TCGA-KIRC) of the OXA_MISS_ccRCC configs

Usage:
    python utils/make_kfold_splits.py [--n_folds 5] [--out splits/OS_grouped_5fold] [--ccrcc_out /work/H2020DeciderFicarra/ccRCC/CPTAC_ccRCC]
"""
import argparse
import contextlib
import glob
import io
import itertools
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataloader.dataloader_multidataset import Multimodal_Bio_Dataset  # noqa: E402

LABELS_DIR = os.path.join(REPO, 'dataloader', 'dataset_updated')
N_FOLDS = 5
SEED = 42
MODALITIES = ['WSI', 'Genomics', 'CNV', 'CT', 'MRI', 'Clinical']
GENE_GROUPS = ["tumor_suppression", "oncogenesis", "protein_kinases", "cellular_differentiation", "cytokines_and_growth"]
# label file -> dataset yaml with the paths of its modalities
DATASET_CONFIGS = {'CPTAC_CCRCC': 'config/ccRCC.yaml', 'TCGA_KIRC': 'config/TCGA_KIRC_dataset.yaml'}


def dataset_config(name):
    return os.path.join(REPO, DATASET_CONFIGS.get(name, f'config/{name}_dataset_UNI.yaml'))


def patient_table(name):
    """One row per patient of the label file: event (1 = dead) and has_<modality> as the dataloader sees them."""
    labels = pd.read_csv(os.path.join(LABELS_DIR, f'{name}_labels.csv'), sep='\t', dtype={'case_id': str})
    labels = labels.dropna(subset=['case_id', 'FUT', 'Survival'])  # same rows the dataloader keeps
    patients = labels.groupby('case_id').agg(event=('Survival', 'first'))
    patients['event'] = patients['event'].astype(int)
    with contextlib.redirect_stdout(io.StringIO()):
        ds = Multimodal_Bio_Dataset(datasets_configs=[dataset_config(name)], task_type='Survival', n_bins=4,
                                    file_genes_group=os.path.join(REPO, 'genes_groups/pathways_ensg.json'),
                                    genomics_group_name=GENE_GROUPS, cnv_group_name=GENE_GROUPS,
                                    input_modalities=MODALITIES, model_name='OXA_MISS')
    has = ds.patient_df[[f'has_{m}' for m in MODALITIES]].reindex(patients.index).fillna(False).astype(bool)
    has.columns = MODALITIES
    patients = patients.join(has)
    patients['dataset'] = ds.patient_df['dataset_name'].reindex(patients.index).fillna(name)
    return patients


def stratified_folds(patients, seed=SEED):
    """case_id -> fold. Greedy assignment balancing, for every modality combination C, the patients of C
    (those with at least one modality of C) per dataset and OS event: each patient goes to the fold where
    the groups (C, dataset, event) it belongs to are the least filled, among the folds not yet full.
    Patients of the rarest groups (dataset x event x modality pattern) are placed first."""
    rng = np.random.default_rng(seed)
    available = [m for m in MODALITIES if patients[m].any()]
    combos = [c for r in range(1, len(available) + 1) for c in itertools.combinations(available, r)]
    has = patients[available].to_numpy(bool)
    member = np.stack([has[:, [available.index(m) for m in c]].any(axis=1) for c in combos], axis=1)  # patients x combos
    datasets = patients['dataset'].astype(str).to_numpy()
    events = patients['event'].to_numpy()
    # group counts per fold: (combination, dataset, event) and (combination, all datasets, event)
    counts = {}
    capacity = int(np.ceil(len(patients) / N_FOLDS))
    fold_size = np.zeros(N_FOLDS, dtype=int)
    pattern = patients[available].astype(int).astype(str).agg(''.join, axis=1)
    strata = patients['dataset'].astype(str) + '|' + patients['event'].astype(str) + '|' + pattern
    stratum_size = strata.map(strata.value_counts())
    order = sorted(range(len(patients)), key=lambda i: (stratum_size.iloc[i], rng.random()))
    fold_of = np.empty(len(patients), dtype=int)
    keys_of = [[(ci, d, events[i]) for ci in np.flatnonzero(member[i]) for d in (datasets[i], 'all')] for i in range(len(patients))]
    # each group weighs 1 / its size: counts are compared as fractions of the group, so small groups
    # (e.g. the 6 CPTAC patients with CT and an event) are balanced as well as the large ones
    group_size = {}
    for keys in keys_of:
        for key in keys:
            group_size[key] = group_size.get(key, 0) + 1
    for i in order:
        keys = keys_of[i]
        score = np.array([sum(counts.get(key, np.zeros(N_FOLDS))[k] / group_size[key] for key in keys) for k in range(N_FOLDS)], dtype=float)
        score[fold_size >= capacity] = np.inf
        best = np.flatnonzero(score == score.min())
        best = best[fold_size[best] == fold_size[best].min()]
        k = int(rng.choice(best))
        fold_of[i] = k
        fold_size[k] += 1
        for key in keys:
            counts.setdefault(key, np.zeros(N_FOLDS))[k] += 1
    return pd.Series(fold_of, index=patients.index)


def write_splits(fold_of, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k in range(N_FOLDS):
        train = sorted(fold_of.index[fold_of != k])
        test = sorted(fold_of.index[fold_of == k])
        pd.concat([pd.Series(train, name='train'), pd.Series(test, name='test')], axis=1) \
            .to_csv(os.path.join(out_dir, f'splits_{k}.csv'), index=False)


def read_folds(split_dir):
    """case_id -> fold of an existing split folder (test column of splits_k), None if missing."""
    fold_of = {}
    for k in range(N_FOLDS):
        f = os.path.join(split_dir, f'splits_{k}.csv')
        if not os.path.exists(f):
            return None
        for pid in pd.read_csv(f)['test'].dropna().astype(str):
            fold_of.setdefault(pid, k)
    return pd.Series(fold_of)


def balance_report(patients, fold_of):
    """For every modality combination (and every dataset inside it): test events per fold."""
    available = [m for m in MODALITIES if patients[m].any()]
    rows = []
    for r in range(1, len(available) + 1):
        for combo in itertools.combinations(available, r):
            sub = patients[patients[list(combo)].any(axis=1)]
            groups = [('all', sub)] + ([(d, g) for d, g in sub.groupby('dataset')] if sub['dataset'].nunique() > 1 else [])
            for dataset, g in groups:
                folds = fold_of.reindex(g.index)
                events = [int(((folds == k) & (g['event'] == 1)).sum()) for k in range(N_FOLDS)]
                rows.append({'combination': '+'.join(combo), 'dataset': dataset, 'patients': len(g),
                             'events': int(g['event'].sum()), 'min_test_events': min(events),
                             'max_test_events': max(events)})
    return pd.DataFrame(rows)


def make(name, patients, out_dir):
    old = read_folds(out_dir)
    fold_of = stratified_folds(patients)
    for k in range(N_FOLDS):  # sanity: partition of the patients
        assert (fold_of == k).sum() > 0
    assert fold_of.notna().all()
    write_splits(fold_of, out_dir)
    new = balance_report(patients, fold_of)
    sizes = fold_of.value_counts().sort_index().tolist()
    line = (f'{name:20s} {len(patients)} pts, {int(patients.event.sum())} events | fold sizes {sizes} | '
            f'{len(new)} (combination, dataset) subsets: worst min test events/fold {new.min_test_events.min()}, '
            f'subsets with a fold without events {int((new.min_test_events == 0).sum())}')
    if old is not None and set(old.index) >= set(patients.index):
        prev = balance_report(patients, old.reindex(patients.index))
        line += f' (previous splits: {int((prev.min_test_events == 0).sum())})'
    print(line)
    return new


def main():
    global N_FOLDS
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_folds', type=int, default=N_FOLDS)
    ap.add_argument('--out', default=os.path.join(REPO, 'splits', 'OS_grouped_5fold'))
    ap.add_argument('--ccrcc_out', default='/work/H2020DeciderFicarra/ccRCC/CPTAC_ccRCC')
    args = ap.parse_args()
    N_FOLDS = args.n_folds

    tables = {}
    for f in sorted(glob.glob(os.path.join(LABELS_DIR, '*_labels.csv'))):
        name = os.path.basename(f).replace('_labels.csv', '')
        tables[name] = patient_table(name)
        make(name, tables[name], os.path.join(args.out, name))

    # combined ccRCC cohort (config/ccRCC.yaml + config/TCGA_KIRC_dataset.yaml)
    patients = pd.concat([tables['CPTAC_CCRCC'], tables['TCGA_KIRC']])
    assert not patients.index.duplicated().any()
    report = make('ccRCC (CPTAC+KIRC)', patients, args.ccrcc_out)
    # not inside ccrcc_out: main.py reads every file of the splits folder as a fold
    report.to_csv(os.path.join(args.out, 'ccRCC_fold_balance_by_modality_combination.csv'), index=False)


if __name__ == '__main__':
    main()
