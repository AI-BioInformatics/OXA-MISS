"""
GMM prototypes of the WSI (PANTHER / MMP style, Song et al. CVPR 2024 / ICML 2024), one GMM per fold, for
data_loader.wsi_representation: gmm (model.kwargs.wsi_encoder: gmm).

Per fold k of the run's datasets (data_loader.datasets_configs of --config, folds = parameters.kfold_splits):
  1. fit: diagonal GMM with K components on up to --n_fit patches of the fold's NON-TEST internal patients
     (training + internal validation: label-free, the test fold and the external datasets are never used),
     the same number of patches per cancer type (sampled from --per_patient random patches of each patient);
  2. per patient (internal and external, all its patches, all its slides): responsibilities r (N, K) under the
     fold's GMM, then for each component pi_k = mean r, mu_k = weighted mean, var_k = weighted variance
     (a component with no mass: the GMM's own mean / variance), -> one (K, 1 + 2 D) matrix [pi | mu | var].
Two passes over the features (sampling, statistics): every slide is read twice, the five folds share pass 2.

Output (--out): gmm_fold<k>.npz (weights, means, variances of the fitted GMM), fold<k>.npy float16 memmap
(n_patients, K, 1 + 2 D), patients.csv (row order: dataset_name, case_id), meta.json.

Usage (GPU job, reads every slide twice: SLURM only, see utils/fit_wsi_gmm.sbatch):
    python utils/fit_wsi_gmm.py --config config/OXA_MISS_final_pancancer8_v2.yaml \
        --out /work/H2020DeciderFicarra/ccRCC/wsi_gmm_cache/pancancer8_UNIv2_K32 [--k 32] [--n_fit 200000]
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import yaml
from munch import munchify
from sklearn.mixture import GaussianMixture

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from dataloader.build import build_dataset  # noqa: E402
from dataloader.dataloader_multidataset import load_wsi_bag  # noqa: E402
from dataloader.kfold import kfold_split_files, read_fold  # noqa: E402


def wsi_dataset(config_path):
    """The run's dataset (internal + external) with WSI only, slides read from disk."""
    with open(config_path) as f:
        config = munchify(yaml.safe_load(f))
    config.model.kwargs.input_modalities = ["WSI"]
    config.data_loader.modalities = {"train": ["WSI"], "val": ["WSI"], "test": [["WSI"]]}
    config.data_loader.load_slides_in_RAM = False
    config.data_loader.missing_modalities_tables.active = False
    config.missing_modality_test.active = False
    config.data_loader.clinical = None
    return config, build_dataset(config)


def patient_patches(dataset, pid):
    """(N, D) float32 patches of all the slides of a patient."""
    name = dataset.patient_df.loc[pid, "dataset_name"]
    path = dataset.datasets[name].pt_files_path
    bags = [load_wsi_bag(dataset._slide_path(path, s)) for s in dataset.slides_on_disk[pid]]
    return torch.cat(bags, dim=0).float()


def responsibilities_stats(x, weights, means, variances):
    """Patient statistics under a diagonal GMM (torch, on x's device): (K, 1 + 2 D) [pi | mu | var]."""
    inv = 1.0 / variances                                                   # (K, D)
    # log N(x | mu_k, var_k) = -0.5 [sum_d x^2 / var - 2 x.mu / var + mu^2 / var + log var + log 2 pi]
    log_p = -0.5 * ((x * x) @ inv.T - 2 * x @ (means * inv).T + (means * means * inv).sum(1)
                    + torch.log(variances).sum(1) + x.shape[1] * np.log(2 * np.pi))
    r = torch.softmax(log_p + torch.log(weights), dim=1)                    # (N, K)
    nk = r.sum(0)                                                           # (K,)
    safe = nk.clamp(min=1e-6)[:, None]
    mu = (r.T @ x) / safe
    var = ((r.T @ (x * x)) / safe - mu * mu).clamp(min=1e-6)
    empty = nk < 1e-3
    mu[empty], var[empty] = means[empty], variances[empty]
    return torch.cat([(nk / x.shape[0])[:, None], mu, var], dim=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--n_fit", type=int, default=200_000)
    ap.add_argument("--per_patient", type=int, default=256, help="patches sampled per patient for the fit")
    ap.add_argument("--max_iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true", help="build the dataset and the folds, print the counts, stop")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    rng = np.random.default_rng(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    config, dataset = wsi_dataset(args.config)
    df = dataset.patient_df
    patients = [p for p in df.index if len(dataset.slides_on_disk.get(p, [])) > 0]
    external = set(df.index[df["dataset_name"].isin(dataset.external_datasets)])
    cancer = df.loc[patients, "dataset_name"].map(dataset.cancer_type)
    folds = []
    for fold_files in kfold_split_files(config):
        _, _, test = read_fold(fold_files, dataset)
        test = set(test)
        folds.append([p for p in patients if p not in test and p not in external])
    logging.info(f"{len(patients)} patients with WSI ({len(external & set(patients))} external), "
                 f"{len(folds)} folds, fit patients per fold {[len(f) for f in folds]}")
    if args.dry_run:
        print(cancer.value_counts().to_string())
        return

    old_umask = os.umask(0o007)
    os.makedirs(args.out, exist_ok=True)
    # pass 1: a random subset of each patient's patches (float16 in RAM)
    sample, t0 = {}, time.time()
    for i, pid in enumerate(patients):
        x = patient_patches(dataset, pid)
        idx = rng.choice(x.shape[0], size=min(args.per_patient, x.shape[0]), replace=False)
        sample[pid] = x[idx].numpy().astype(np.float16)
        if i % 500 == 0:
            logging.info(f"pass 1: {i}/{len(patients)} patients ({time.time() - t0:.0f}s)")
    dim = next(iter(sample.values())).shape[1]

    gmms = []
    for k, fit_patients in enumerate(folds):
        fit_cancer = cancer.loc[fit_patients]
        per_cancer = args.n_fit // fit_cancer.nunique()
        chunks = []
        for c in sorted(fit_cancer.unique()):
            pooled = np.concatenate([sample[p] for p in fit_cancer.index[fit_cancer == c]])
            chunks.append(pooled[rng.choice(len(pooled), size=min(per_cancer, len(pooled)), replace=False)])
        x_fit = np.concatenate(chunks).astype(np.float32)
        logging.info(f"fold {k}: fitting GMM K={args.k} on {len(x_fit)} patches ({fit_cancer.nunique()} cancer types)")
        gmm = GaussianMixture(args.k, covariance_type="diag", max_iter=args.max_iter, random_state=args.seed,
                              reg_covar=1e-6).fit(x_fit)
        logging.info(f"fold {k}: converged {gmm.converged_} after {gmm.n_iter_} iterations")
        np.savez(os.path.join(args.out, f"gmm_fold{k}.npz"), weights=gmm.weights_, means=gmm.means_,
                 variances=gmm.covariances_)
        gmms.append([torch.tensor(a, dtype=torch.float32, device=device)
                     for a in (gmm.weights_, gmm.means_, gmm.covariances_)])
    del sample

    # pass 2: per-patient statistics under every fold's GMM
    outs = [np.lib.format.open_memmap(os.path.join(args.out, f"fold{k}.npy"), mode="w+", dtype=np.float16,
                                      shape=(len(patients), args.k, 1 + 2 * dim)) for k in range(len(folds))]
    t0 = time.time()
    with torch.no_grad():
        for i, pid in enumerate(patients):
            x = patient_patches(dataset, pid).to(device)
            for k, (w, m, v) in enumerate(gmms):
                outs[k][i] = responsibilities_stats(x, w, m, v).cpu().numpy().astype(np.float16)
            if i % 500 == 0:
                logging.info(f"pass 2: {i}/{len(patients)} patients ({time.time() - t0:.0f}s)")
    for o in outs:
        o.flush()
    pd.DataFrame({"dataset_name": df.loc[patients, "dataset_name"].values, "case_id": patients}).to_csv(
        os.path.join(args.out, "patients.csv"), index=False)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump({"config": os.path.abspath(args.config), "k": args.k, "n_fit": args.n_fit, "per_patient": args.per_patient,
                   "seed": args.seed, "dim": dim, "n_patients": len(patients), "n_folds": len(folds),
                   "layout": "[pi | mu (dim) | var (dim)] per component", "fit": "non-test internal patients of the fold"}, f, indent=1)
    os.umask(old_umask)
    logging.info(f"written {args.out}")


if __name__ == "__main__":
    main()
