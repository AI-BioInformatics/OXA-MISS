"""
Unimodal CoxPH baselines on the same patients and folds as main.py: for each modality, a CoxPH (lifelines,
ridge penalty) trained on that modality alone, on all the datasets of the config together (e.g. TCGA
pan-cancer), with the patients that have it.

Features of each modality (fitted on the fold's training patients only):
    Clinical   one-hot categorical (missing = all zeros), age (missing -> training mean) + missing indicator
    Genomics   genes of the pathway groups (log), standardized, PCA
    WSI        mean of the raw patch features over all the patient's slides, standardized, PCA
    CT / MRI   mean-pooled exam vector (features_meanpooled), standardized, PCA
The WSI means are computed once (reads every slide) and cached in --cache_dir.

Metrics per fold (test patients with the modality): Harrell and Uno c-index, integrated Brier score, and the
c-index within each cohort; mean +- std over the folds.
Results: experiments/test_results_csv/unimodal_baseline.csv (one row per modality).

Usage (CPU, hours for the WSI means the first time: SLURM job):
    python utils/unimodal_baseline.py --config config/OXA_MISS_final_ccRCC.yaml \
        --datasets KIRC+BLCA+COAD+KIRP+LIHC+LUAD+LUSC+STAD [--encoder UNI] \
        [--modalities WSI Genomics Clinical CT MRI] [--n_components 32]
--datasets (tumor names of config/datasets.yaml, as for grid_search/make_modality_grid.py) replaces the
datasets_configs of the config, so no dedicated yaml is needed.
"""
import argparse
import contextlib
import copy
import datetime
import io
import os
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import yaml
from munch import munchify

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "utils"))
sys.path.insert(0, os.path.join(REPO, "grid_search"))
from dataloader.build import build_dataset, fold_patients  # noqa: E402
from dataloader.kfold import kfold_split_files  # noqa: E402
from clinical_baseline import design_matrix as clinical_design  # noqa: E402
from experiments.utils import safe_c_index  # noqa: E402

warnings.filterwarnings("ignore")


def wsi_means(dataset, cache_dir):
    """patient -> mean raw patch feature over all its slides (patch-weighted), cached per dataset."""
    out = {}
    for name, params in dataset.datasets.items():
        cache = os.path.join(cache_dir, f"wsi_mean_{name}_{os.path.basename(os.path.dirname(params.pt_files_path))}.npz")
        if os.path.exists(cache):
            try:   # another job may be writing it (atomic rename below), or it may be from an interrupted run
                data = np.load(cache, allow_pickle=True)
                out.update(dict(zip(data["patients"], data["means"])))
                continue
            except Exception as e:
                print(f"  cache {cache} unreadable ({e}): recomputing")
        patients = dataset.patient_df.index[(dataset.patient_df["dataset_name"] == name) & dataset.patient_df["has_WSI"]]
        ids, means = [], []
        for pid in patients:
            total, count = 0.0, 0
            for slide in dataset.slides_on_disk[pid]:
                bag = torch.load(dataset._slide_path(params.pt_files_path, slide), weights_only=True, map_location="cpu").float()
                total, count = total + bag.sum(dim=0), count + bag.shape[0]
            ids.append(pid)
            means.append((total / count).numpy())
        os.makedirs(cache_dir, exist_ok=True)
        tmp = f"{cache}.{os.getpid()}.tmp.npz"   # write then rename: concurrent jobs never read a partial file
        np.savez(tmp, patients=np.array(ids), means=np.stack(means))
        os.replace(tmp, cache)
        print(f"  WSI means of {name}: {len(ids)} patients -> {cache}")
        out.update(dict(zip(ids, means)))
    return out


def features(modality, dataset, patients, wsi_cache):
    """Raw feature matrix (patients x features) of a modality."""
    if modality == "WSI":
        return np.stack([wsi_cache[p] for p in patients])
    if modality == "Genomics":
        return dataset.genomics.loc[list(patients)].to_numpy(dtype=float)
    if modality in ("CT", "MRI"):
        return np.stack([dataset._radiology_features(modality, dataset.patient_df.loc[p, "dataset_name"], p).numpy().reshape(-1)
                         for p in patients])
    raise ValueError(modality)


def main():
    from lifelines import CoxPHFitter
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from sksurv.util import Surv

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--datasets", default=None, metavar="TUMOR+TUMOR", help="tumors of config/datasets.yaml")
    ap.add_argument("--encoder", default="UNI", help="WSI features of --datasets")
    ap.add_argument("--excluded_studies", nargs="*", default=None, help="clinical table studies to exclude")
    ap.add_argument("--modalities", nargs="+", default=["WSI", "Genomics", "Clinical", "CT", "MRI"])
    ap.add_argument("--n_components", type=int, default=32)
    ap.add_argument("--penalizer", type=float, default=0.1)
    ap.add_argument("--cache_dir", default="/work/H2020DeciderFicarra/ccRCC/baseline_cache")
    args = ap.parse_args()
    base = munchify(yaml.safe_load(open(args.config)))
    if args.datasets:
        from make_modality_grid import dataset_configs   # same tumor -> yaml mapping as the grid
        base.data_loader.datasets_configs = dataset_configs(args.datasets, args.encoder)
    if args.excluded_studies is not None and base.data_loader.get("clinical"):
        base.data_loader.clinical.excluded_studies = list(args.excluded_studies)
    summary = []
    for modality in args.modalities:
        config = copy.deepcopy(base)
        config.model.kwargs.input_modalities = [modality]
        config.data_loader.modalities = munchify({"train": [modality], "val": [modality], "test": [[modality]]})
        config.data_loader.load_slides_in_RAM = False      # the WSI means are read once and cached
        config.data_loader.radiology_tokens = False        # CT / MRI: mean-pooled exam vectors
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = build_dataset(config)
        df = dataset.patient_df[dataset.patient_df[f"has_{modality}"]]
        if len(df) < 50:
            print(f"{modality}: {len(df)} patients, skipped")
            continue
        wsi_cache = wsi_means(dataset, args.cache_dir) if modality == "WSI" else None
        rows = []
        for k, fold_files in enumerate(kfold_split_files(config)):
            np.random.seed(config.seed)
            train, val, test = fold_patients(config, dataset, fold_files)
            train = [p for p in np.concatenate([x for x in (train, val) if x is not None]) if p in df.index]
            test = [p for p in test if p in df.index]
            if modality == "Clinical":
                tokens = dataset.clinical_tokens
                age_mean = [float(np.mean(tokens.num[[tokens.row[c] for c in train], 0][tokens.num_mask[[tokens.row[c] for c in train], 0]]))]
                X_tr, X_te = clinical_design(tokens, train, age_mean), clinical_design(tokens, test, age_mean)
                X_te = X_te.reindex(columns=X_tr.columns, fill_value=0.0)
            else:
                scaler = StandardScaler().fit(features(modality, dataset, train, wsi_cache))
                Z_tr = np.nan_to_num(scaler.transform(features(modality, dataset, train, wsi_cache)))
                Z_te = np.nan_to_num(scaler.transform(features(modality, dataset, test, wsi_cache)))
                pca = PCA(n_components=min(args.n_components, Z_tr.shape[1], len(train) - 1), random_state=0).fit(Z_tr)
                cols = [f"pc{i}" for i in range(pca.n_components_)]
                X_tr = pd.DataFrame(pca.transform(Z_tr), index=train, columns=cols)
                X_te = pd.DataFrame(pca.transform(Z_te), index=test, columns=cols)
            data = pd.concat([X_tr, df.loc[train, ["time"]].astype(float),
                              (df.loc[train, "censorship"] == 0).astype(int).rename("event")], axis=1)
            cox = CoxPHFitter(penalizer=args.penalizer).fit(data, duration_col="time", event_col="event")
            risk = cox.predict_partial_hazard(X_te).values
            times, cens = df.loc[test, "time"].values.astype(float), df.loc[test, "censorship"].values
            y_tr = Surv.from_arrays(event=df.loc[train, "censorship"].values == 0, time=df.loc[train, "time"].values.astype(float))
            y_te = Surv.from_arrays(event=cens == 0, time=times)
            t_max = min(df.loc[train, "time"].max(), np.percentile(times, 95))
            row = {"fold": k + 1, "patients": len(test), "c-index": safe_c_index(cens, times, risk)}
            try:
                row["c-index_uno"] = concordance_index_ipcw(y_tr, y_te, risk, tau=t_max)[0]
            except Exception:
                row["c-index_uno"] = np.nan
            try:
                grid = np.linspace(np.percentile(times, 5), t_max, 50, endpoint=False)
                row["IBS"] = integrated_brier_score(y_tr, y_te, cox.predict_survival_function(X_te, times=grid).T.to_numpy(), grid)
            except Exception:
                row["IBS"] = np.nan
            within = []
            for ds in sorted(df.loc[test, "dataset_name"].unique()):
                sel = (df.loc[test, "dataset_name"] == ds).values
                row[f"{ds}_c-index"] = safe_c_index(cens[sel], times[sel], risk[sel])
                within.append(row[f"{ds}_c-index"])
            row["mean_within_cohort_c-index"] = np.nanmean(within) if within else np.nan
            rows.append(row)
        folds = pd.DataFrame(rows)
        print(f"\n=== {modality}: {len(df)} patients, {df['dataset_name'].nunique()} cohorts")
        print(folds.round(3).to_string(index=False))
        out = {"modality": modality, "config": os.path.basename(args.config), "datasets": "+".join(dataset.datasets),
               "patients": len(df),
               "cohorts": df["dataset_name"].nunique(), "n_components": args.n_components,
               "End Time": datetime.datetime.now().isoformat(timespec="seconds")}
        for metric in [c for c in folds.columns if c not in ("fold", "patients")]:
            out[f"{metric}_mean"], out[f"{metric}_std"] = round(folds[metric].mean(), 3), round(folds[metric].std(ddof=0), 3)
        summary.append(out)
    summary = pd.DataFrame(summary)
    print("\n" + summary[[c for c in summary.columns if not c.startswith("TCGA_")]].to_string(index=False))
    csv = os.path.join(REPO, "experiments", "test_results_csv", "unimodal_baseline.csv")
    if os.path.exists(csv):
        summary = pd.concat([pd.read_csv(csv), summary], ignore_index=True)
    summary.to_csv(csv, index=False)
    print(f"-> {csv}")


if __name__ == "__main__":
    main()
