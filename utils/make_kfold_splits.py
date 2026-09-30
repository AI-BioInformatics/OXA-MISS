"""
Patient-grouped, event-stratified 5-fold splits built from the updated OS label files.

For each label file in dataloader/dataset_updated/ a folder <out>/<DATASET>/splits_{0..4}.csv is written
(columns: train, test; fold k is the test set of splits_k). Rows of the label files are slides, so the
folds are built with StratifiedGroupKFold (groups = case_id, one row per patient), and every patient ends up
in exactly one test fold.
Stratification: OS event (Survival, 1 = Dead); for the combined ccRCC splits (CPTAC + TCGA-KIRC) also the dataset.

Usage:
    python utils/make_kfold_splits.py [--out splits/OS_grouped_5fold] [--ccrcc_out /work/H2020DeciderFicarra/ccRCC/CPTAC_ccRCC]
"""
import argparse
import glob
import os

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS_DIR = os.path.join(REPO, 'dataloader', 'dataset_updated')
N_FOLDS = 5
SEED = 42


def load_labels(label_file):
    df = pd.read_csv(label_file, sep='\t', dtype={'case_id': str})
    df = df.dropna(subset=['case_id', 'FUT', 'Survival'])  # same rows the dataloader keeps
    df['Survival'] = df['Survival'].astype(int)
    return df


def grouped_folds(df, strata):
    """case_id -> fold index. df: one row per slide; strata: per-row stratification labels.
    Stratification is done on one row per patient: on slide rows, patients with many slides
    would weigh more and the event balance of the folds gets much worse (e.g. CPTAC)."""
    patients = pd.DataFrame({'case_id': df['case_id'].values, 'strata': pd.Series(strata).values}) \
        .drop_duplicates('case_id').reset_index(drop=True)
    sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    fold_of = {}
    for k, (_, test_idx) in enumerate(sgkf.split(patients, patients['strata'], groups=patients['case_id'])):
        for pid in patients['case_id'].iloc[test_idx]:
            fold_of[pid] = k
    return pd.Series(fold_of)


def write_splits(fold_of, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for k in range(N_FOLDS):
        train = sorted(fold_of.index[fold_of != k])
        test = sorted(fold_of.index[fold_of == k])
        pd.concat([pd.Series(train, name='train'), pd.Series(test, name='test')], axis=1) \
            .to_csv(os.path.join(out_dir, f'splits_{k}.csv'), index=False)


def check_and_report(name, fold_of, patients, out_dir):
    assert set(fold_of.index) == set(patients.index), f'{name}: some patients have no fold'
    for k in range(N_FOLDS):
        s = pd.read_csv(os.path.join(out_dir, f'splits_{k}.csv'))
        tr, te = set(s['train'].dropna()), set(s['test'].dropna())
        assert not (tr & te) and tr | te == set(patients.index), f'{name}: fold {k} is not a partition'
    tab = patients.groupby(fold_of.reindex(patients.index)).agg(n=('event', 'size'), events=('event', 'sum'))
    desc = ' | '.join(f'f{k}: {r.n} pts, {r.events} ev' for k, r in tab.iterrows())
    print(f'{name:12s} {len(patients)} pts, {int(patients.event.sum())} events -> {desc}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'splits', 'OS_grouped_5fold'))
    ap.add_argument('--ccrcc_out', default='/work/H2020DeciderFicarra/ccRCC/CPTAC_ccRCC')
    args = ap.parse_args()

    per_dataset = {}
    for f in sorted(glob.glob(os.path.join(LABELS_DIR, '*_labels.csv'))):
        name = os.path.basename(f).replace('_labels.csv', '')
        df = load_labels(f)
        per_dataset[name] = df
        fold_of = grouped_folds(df, df['Survival'])
        out_dir = os.path.join(args.out, name)
        write_splits(fold_of, out_dir)
        patients = df.groupby('case_id').agg(event=('Survival', 'first'))
        check_and_report(name, fold_of, patients, out_dir)

    # combined ccRCC cohort used by the OXA_MISS_ccRCC experiments (config/ccRCC.yaml + config/TCGA_KIRC_dataset.yaml)
    df = pd.concat([per_dataset['CPTAC_CCRCC'].assign(dataset='CPTAC'),
                    per_dataset['TCGA_KIRC'].assign(dataset='TCGA_KIRC')], ignore_index=True)
    assert not df.groupby('case_id')['dataset'].nunique().gt(1).any()
    fold_of = grouped_folds(df, df['dataset'] + '_' + df['Survival'].astype(str))
    write_splits(fold_of, args.ccrcc_out)
    patients = df.groupby('case_id').agg(event=('Survival', 'first'))
    check_and_report('ccRCC (CPTAC+KIRC)', fold_of, patients, args.ccrcc_out)


if __name__ == '__main__':
    main()
