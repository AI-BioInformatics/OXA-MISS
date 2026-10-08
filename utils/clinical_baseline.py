"""
Clinical-only survival baselines on the same patients and folds as main.py: CoxPH (lifelines, ridge penalty)
and gradient-boosted survival trees (scikit-survival), on the clinical token variables of data_loader.clinical
(age, sex, stage, T, N, M, optionally the study). Missing values: one-hot without a category (all zeros) for the
categorical variables, training mean + missing indicator for age. Reviewers expect this baseline: trees are
still strong on medium-sized tabular data (Grinsztajn et al., NeurIPS 2022).

Metrics per fold (test patients with clinical data): Harrell c-index, Uno c-index, integrated Brier score;
mean +- std over the folds and per dataset. Results: experiments/test_results_csv/clinical_baseline.csv.

Usage (CPU, a few minutes; run it as a SLURM job):
    python utils/clinical_baseline.py --config config/OXA_MISS_final_ccRCC.yaml [--study_id]
"""
import argparse
import contextlib
import datetime
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from munch import munchify

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataloader.dataloader_multidataset import Multimodal_Bio_Dataset  # noqa: E402
from dataloader.clinical_tokens import NUM_COLS  # noqa: E402
from dataloader.kfold import kfold_split_files, read_fold  # noqa: E402
from experiments.utils import safe_c_index  # noqa: E402

warnings.filterwarnings("ignore")


def design_matrix(tokens, case_ids, age_mean):
    """One-hot categorical (missing = all zeros), age (missing -> training mean) + age missing indicator."""
    rows = [tokens.row[c] for c in case_ids]
    columns, names = [], []
    for j, col in enumerate(tokens.cat_cols):
        values = tokens.cat[rows, j]
        for k in range(1, tokens.cat_cardinalities[j] + 1):
            columns.append((values == k).astype(float)); names.append(f"{col}={k}")
    for j, col in enumerate(NUM_COLS):
        observed = tokens.num_mask[rows, j]
        columns.append(np.where(observed, tokens.num[rows, j], age_mean[j])); names.append(col)
        columns.append((~observed).astype(float)); names.append(f"{col}_missing")
    X = pd.DataFrame(np.stack(columns, axis=1), columns=names, index=case_ids)
    return X.loc[:, X.std() > 0] if len(X) > 1 else X


def survival_curves(model_name, model, X, grid):
    """(N, len(grid)) predicted survival at the grid times."""
    if model_name == "coxph":
        return model.predict_survival_function(X, times=grid).T.to_numpy()
    return np.stack([fn(grid) for fn in model.predict_survival_function(X)])


def main():
    from lifelines import CoxPHFitter
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv

    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--study_id', action='store_true', help='add the study token (one-hot) as a variable')
    args = ap.parse_args()
    config = munchify(yaml.safe_load(open(args.config)))
    clinical = dict(config.data_loader.get('clinical') or {})
    if args.study_id:
        clinical['use_study_id'] = True
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = Multimodal_Bio_Dataset(datasets_configs=config.data_loader.datasets_configs, task_type='Survival',
                                         file_genes_group=os.path.join(REPO, config.data_loader.file_genes_group),
                                         genomics_group_name=[], cnv_group_name=[], input_modalities=['Clinical'],
                                         model_name='clinical_baseline', clinical_tokens=clinical)
    tokens, df = dataset.clinical_tokens, dataset.patient_df
    df = df[df['has_Clinical']]
    rows = []
    for k, fold_files in enumerate(kfold_split_files(config)):
        train, val, test = read_fold(fold_files, dataset)
        train = np.concatenate([p for p in (train, val) if p is not None])   # no early stopping here
        train, test = [p for p in train if p in df.index], [p for p in test if p in df.index]
        age_mean = [float(np.mean(tokens.num[[tokens.row[c] for c in train], j][tokens.num_mask[[tokens.row[c] for c in train], j]]))
                    for j in range(len(NUM_COLS))]
        X_train, X_test = design_matrix(tokens, train, age_mean), design_matrix(tokens, test, age_mean)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0.0)
        y_train = Surv.from_arrays(event=df.loc[train, 'censorship'].values == 0, time=df.loc[train, 'time'].values.astype(float))
        y_test = Surv.from_arrays(event=df.loc[test, 'censorship'].values == 0, time=df.loc[test, 'time'].values.astype(float))
        times = df.loc[test, 'time'].values.astype(float)
        t_max = min(df.loc[train, 'time'].max(), np.percentile(times, 95))
        grid = np.linspace(np.percentile(times, 5), t_max, 50, endpoint=False)
        models = {}
        cox = CoxPHFitter(penalizer=0.1)
        cox.fit(pd.concat([X_train, df.loc[train, ['time']].astype(float),
                           (df.loc[train, 'censorship'] == 0).astype(int).rename('event')], axis=1),
                duration_col='time', event_col='event')
        models['coxph'] = (cox, cox.predict_partial_hazard(X_test).values)
        gbt = GradientBoostingSurvivalAnalysis(n_estimators=200, learning_rate=0.05, max_depth=3, random_state=42)
        gbt.fit(X_train.values, y_train)
        models['gbt'] = (gbt, gbt.predict(X_test.values))
        for name, (model, risk) in models.items():
            row = {'model': name, 'fold': k + 1, 'patients': len(test), 'events': int(y_test['event'].sum()),
                   'c-index': safe_c_index(df.loc[test, 'censorship'].values, times, risk)}
            try:
                row['c-index_uno'] = concordance_index_ipcw(y_train, y_test, risk, tau=t_max)[0]
            except Exception:
                row['c-index_uno'] = np.nan
            try:
                X_eval = X_test if name == 'coxph' else X_test.values
                row['IBS'] = integrated_brier_score(y_train, y_test, survival_curves(name, model, X_eval, grid), grid)
            except Exception:
                row['IBS'] = np.nan
            for ds in sorted(df.loc[test, 'dataset_name'].unique()):
                sel = (df.loc[test, 'dataset_name'] == ds).values
                row[f'{ds}_c-index'] = safe_c_index(df.loc[test, 'censorship'].values[sel], times[sel], risk[sel])
            rows.append(row)
    folds = pd.DataFrame(rows)
    print(folds.round(3).to_string(index=False))
    summary = []
    for name, g in folds.groupby('model'):
        out = {'model': name, 'datasets': '+'.join(dataset.datasets), 'clinical': '+'.join(tokens.cat_cols),
               'End Time': datetime.datetime.now().isoformat(timespec='seconds'), 'patients': int(g['patients'].sum())}
        for metric in [c for c in folds.columns if c not in ('model', 'fold', 'patients', 'events')]:
            out[f'{metric}_mean'], out[f'{metric}_std'] = round(g[metric].mean(), 3), round(g[metric].std(ddof=0), 3)
        summary.append(out)
    summary = pd.DataFrame(summary)
    print('\n' + summary.to_string(index=False))
    csv = os.path.join(REPO, 'experiments', 'test_results_csv', 'clinical_baseline.csv')
    if os.path.exists(csv):  # runs on other datasets add other per-dataset columns
        summary = pd.concat([pd.read_csv(csv), summary], ignore_index=True)
    summary.to_csv(csv, index=False)
    print(f'-> {csv}')


if __name__ == '__main__':
    main()
