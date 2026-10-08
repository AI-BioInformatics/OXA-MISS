"""
Module B, measuring part: does the model (or the data alone) encode WHICH study a patient comes from and
WHICH modalities / clinical variables are missing, beyond the content of the available data?
Background: TCGA site signatures (Howard et al., Nat Commun 2021), informative missingness (Agniel et al.,
BMJ 2018), docs/design_rationale.md.

1) --missingness_baseline (CPU): survival from the availability pattern alone, same folds as main.py
   (CoxPH, ridge): which modalities the patient has, which clinical variables are recorded, the study, and both.
   A high c-index here = the pattern itself is prognostic; the model can use it as a shortcut.
2) --run_dir <results/run> (GPU): probes on the representations of a trained run. For each fold, the fold's
   checkpoint embeds its training and test patients (fused embedding and each modality's embedding); logistic
   probes are fitted on the training patients and evaluated (AUC) on the test patients, for:
     study            cohort / site (CPTAC vs TCGA-KIRC: same cancer, so it is site, not biology)
     cancer type      control (only with several tumors)
     has_<modality>   missingness pattern of the modalities
     <var>_missing    missingness of each clinical variable
   with a label-shuffled probe as chance level. Representations: fused, each modality after the cross-attention
   (not unimodal when the modalities interact), and <modality>_input (raw WSI features averaged over the
   patches, raw genes, clinical tokens before the cross-attention): what is decodable there is in the data. Linear probes; an adversary at chance during training does
   not prove the information is gone, so these probes are fitted from scratch on the frozen embeddings.

3) --counterfactual --run_dir <run> (GPU): does the model use the PRESENCE of a modality beyond its content?
   For every test patient with modality m, the risk with: the real data (full); m's content taken from a
   random other patient of the same cohort (shuffled: same pattern, random content); m hidden (removed:
   pattern changed). presence effect = mean(removed - shuffled) with a bootstrap 95% CI: systematically
   != 0 -> the risk moves because m is present / absent, not because of what m shows (shortcut). Also the
   content effect mean|shuffled - full| and the c-index of the three variants.
   (The has_<modality> probe on the fused representation is not such evidence: a representation built from
   the available modalities necessarily reveals which ones they are.)

Usage (SLURM, not on the login node):
    python utils/shortcut_analysis.py --missingness_baseline --config config/OXA_MISS_final_ccRCC.yaml
    python utils/shortcut_analysis.py --run_dir /work/.../results/<run> [--checkpoint last_epoch]
    python utils/shortcut_analysis.py --counterfactual --run_dir /work/.../results/<run>
"""
import argparse
import contextlib
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
from dataloader.build import build_dataset, fold_patients  # noqa: E402
from dataloader.clinical_tokens import NUM_COLS  # noqa: E402
from dataloader.dataloader_utils import make_dataloader  # noqa: E402
from dataloader.kfold import kfold_split_files  # noqa: E402
from experiments.utils import import_class_from_path, move_to_device, safe_c_index  # noqa: E402

warnings.filterwarnings("ignore")
MODALITIES = ["WSI", "Genomics", "CNV", "CT", "MRI", "Clinical"]
CHECKPOINTS = {"last_epoch": "model_last_epoch", "lowest_val_loss": "model_lowest_loss",
               "highest_val_metric": "model_highest_metric"}


def cancer_type(dataset_name):
    from dataloader.dataloader_multidataset import Multimodal_Bio_Dataset
    return Multimodal_Bio_Dataset.cancer_type(dataset_name)


def pattern_features(dataset, patients, modalities):
    """Availability pattern of each patient: has_<modality> and <clinical variable>_missing (0/1)."""
    df = dataset.patient_df.loc[patients]
    X = pd.DataFrame({f"has_{m}": df[f"has_{m}"].astype(float).values for m in modalities}, index=patients)
    tokens = dataset.clinical_tokens
    if tokens is not None and "Clinical" in modalities:
        n_real = len(tokens.cat_cols) - int(tokens.use_study_id)
        for p in patients:
            i = tokens.row.get(p)
            for j, col in enumerate(NUM_COLS):
                X.loc[p, f"{col}_missing"] = 1.0 if i is None else float(not tokens.num_mask[i, j])
            for j, col in enumerate(tokens.cat_cols[:n_real]):
                X.loc[p, f"{col}_missing"] = 1.0 if i is None else float(tokens.cat[i, j] == 0)
    return X


def missingness_baseline(config):
    from lifelines import CoxPHFitter
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = build_dataset(config)
    modalities = list(config.model.kwargs.input_modalities)
    rows = []
    for k, fold_files in enumerate(kfold_split_files(config)):
        np.random.seed(config.seed)
        train, val, test = fold_patients(config, dataset, fold_files)
        train = np.concatenate([p for p in (train, val) if p is not None])
        train, test = [p for p in train if p in dataset.patient_df.index], [p for p in test if p in dataset.patient_df.index]
        df = dataset.patient_df
        pattern_tr, pattern_te = pattern_features(dataset, train, modalities), pattern_features(dataset, test, modalities)
        study_tr = pd.get_dummies(df.loc[train, "dataset_name"]).astype(float)
        study_te = pd.get_dummies(df.loc[test, "dataset_name"]).reindex(columns=study_tr.columns, fill_value=0).astype(float)
        variants = {"pattern": (pattern_tr, pattern_te), "study": (study_tr, study_te),
                    "pattern+study": (pd.concat([pattern_tr, study_tr], axis=1), pd.concat([pattern_te, study_te], axis=1))}
        for name, (X_tr, X_te) in variants.items():
            keep = X_tr.columns[X_tr.std() > 0]
            row = {"features": name, "fold": k + 1, "n_features": len(keep)}
            if len(keep) == 0:
                row["c-index"] = np.nan
            else:
                data = pd.concat([X_tr[keep], df.loc[train, ["time"]].astype(float),
                                  (df.loc[train, "censorship"] == 0).astype(int).rename("event")], axis=1)
                cox = CoxPHFitter(penalizer=0.1).fit(data, duration_col="time", event_col="event")
                risk = cox.predict_partial_hazard(X_te[keep]).values
                row["c-index"] = safe_c_index(df.loc[test, "censorship"].values, df.loc[test, "time"].values, risk)
                for ds in sorted(df.loc[test, "dataset_name"].unique()):   # within-cohort: the pattern only
                    sel = (df.loc[test, "dataset_name"] == ds).values
                    row[f"{ds}_c-index"] = safe_c_index(df.loc[test, "censorship"].values[sel],
                                                        df.loc[test, "time"].values[sel], risk[sel])
            rows.append(row)
    folds = pd.DataFrame(rows)
    print(folds.round(3).to_string(index=False))
    metrics = [c for c in folds.columns if c.endswith("c-index")]
    summary = folds.groupby("features")[metrics].agg(["mean", "std"]).round(3)
    print("\nMissingness-only survival baseline (c-index mean, std over folds):\n" + summary.to_string())
    return summary


def embed(model, loader, device):
    """patient -> {fused / modality: embedding (D,)} with the model in eval mode."""
    out = {}
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            result = model(move_to_device(batch["input"], device))
            out[batch["patient_id"][0]] = {k: v[0].float().cpu().numpy() for k, v in result["embeddings"].items()}
    return out


def probe_auc(X_tr, y_tr, X_te, y_te, rng, shuffled=False):
    """AUC (macro one-vs-rest) of a logistic probe; NaN if a class is missing from either side."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if shuffled:
        y_tr = rng.permutation(y_tr)
    if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2 or not set(np.unique(y_te)) <= set(np.unique(y_tr)):
        return np.nan
    probe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0)).fit(X_tr, y_tr)
    proba = probe.predict_proba(X_te)
    if proba.shape[1] == 2:
        return roc_auc_score(y_te, proba[:, 1])
    return roc_auc_score(y_te, proba, multi_class="ovr", labels=probe.classes_)


def iterate_run_folds(run_dir, checkpoint):
    """For each fold of a trained k-fold run: (fold index, config, dataset with the fold's normalization,
    the fold's model in eval mode, train, val, test patients, device). Same data and validation split as
    main.py (config.yaml of the run)."""
    config = munchify(yaml.safe_load(open(os.path.join(run_dir, "config.yaml"))))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(io.StringIO()):
        dataset = build_dataset(config)
    config.data_loader.num_workers = 0
    dataset.set_sample(config.data_loader.test_sample)
    # the checkpoints are pickled models: their class module must be importable under its file name
    import_class_from_path(os.path.join(REPO, "experiments", "models", f"{config.model.name}.py"), config.model.name)
    for k, fold_files in enumerate(kfold_split_files(config)):
        np.random.seed(config.seed)   # same validation split as main.py
        train, val, test = fold_patients(config, dataset, fold_files)
        if "Genomics" in config.model.kwargs.input_modalities:
            dataset.normalize_genomics(train, val, test)
        path = os.path.join(run_dir, f"{CHECKPOINTS[checkpoint]}_Fold_{k + 1}.pt")
        model = torch.load(path, weights_only=False, map_location=device).to(device).eval()
        yield k, config, dataset, model, train, val, test, device


def run_probes(run_dir, checkpoint):
    rng = np.random.default_rng(0)
    rows = []
    for k, config, dataset, model, train, val, test, device in iterate_run_folds(run_dir, checkpoint):
        modalities = list(config.model.kwargs.input_modalities)
        with contextlib.redirect_stdout(io.StringIO()):
            loaders = {p: make_dataloader(dataset, p, ids, config.data_loader.modalities.test[0], config)
                       for p, ids in (("train", train), ("test", test))}
        emb = {p: embed(model, loader, device) for p, loader in loaders.items()}
        patterns = {p: pattern_features(dataset, list(emb[p]), modalities) for p in emb}
        df = dataset.patient_df

        def labels(target, ids, split):
            if target == "study":
                return df.loc[ids, "dataset_name"].values
            if target == "cancer_type":
                return np.array([cancer_type(d) for d in df.loc[ids, "dataset_name"]])
            return patterns[split].loc[ids, target].values

        targets = ["study", "cancer_type"] + list(patterns["train"].columns)
        input_level = [f"{m}_input" for m in ("WSI", "Genomics", "Clinical") if m in modalities]
        for representation in ["fused"] + modalities + input_level:
            train_ids = [p for p in emb["train"] if representation in emb["train"][p]]
            test_ids = [p for p in emb["test"] if representation in emb["test"][p]]
            if len(train_ids) < 20 or len(test_ids) < 10:
                continue
            X_tr = np.stack([emb["train"][p][representation] for p in train_ids])
            X_te = np.stack([emb["test"][p][representation] for p in test_ids])
            for target in targets:
                y_tr, y_te = labels(target, train_ids, "train"), labels(target, test_ids, "test")
                rows.append({"fold": k + 1, "representation": representation, "target": target,
                             "AUC": probe_auc(X_tr, y_tr, X_te, y_te, rng),
                             "AUC_shuffled": probe_auc(X_tr, y_tr, X_te, y_te, rng, shuffled=True),
                             "n_train": len(train_ids), "n_test": len(test_ids)})
        print(f"fold {k + 1}: {sum(r['fold'] == k + 1 for r in rows)} probes")
    probes = pd.DataFrame(rows)
    summary = probes.groupby(["representation", "target"])[["AUC", "AUC_shuffled"]].agg(["mean", "std"]).round(3)
    summary = summary.dropna(how="all")
    print("\nProbe AUC on the test patients (mean, std over folds; shuffled = chance):\n" + summary.to_string())
    out = os.path.join(run_dir, f"shortcut_probes_{checkpoint}.csv")
    probes.to_csv(out, index=False)
    print(f"-> {out}")


STATUS_KEYS = {"WSI": "WSI_status", "Genomics": "genomics_status", "CNV": "cnv_status",
               "CT": "ct_status", "MRI": "mri_status", "Clinical": "clinical_status"}
# input keys carrying the content of each modality (swapped between patients by the counterfactual)
CONTENT_KEYS = {"WSI": ["patch_features", "mask"], "Genomics": ["genomics"], "CNV": ["cnv"],
                "CT": ["ct_features"], "MRI": ["mri_features"],
                "Clinical": ["clinical_num", "clinical_num_mask", "clinical_cat", "clinical_features"]}


def _risk(model, data, device):
    logits = model(move_to_device(data, device))["output"][0].float().cpu().numpy()
    return float(-np.cumprod(1 - 1 / (1 + np.exp(-logits))).sum())


def run_counterfactual(run_dir, checkpoint, n_boot=2000):
    rng = np.random.default_rng(0)
    rows = []
    for k, config, dataset, model, train, val, test, device in iterate_run_folds(run_dir, checkpoint):
        modalities = list(config.model.kwargs.input_modalities)
        with contextlib.redirect_stdout(io.StringIO()):
            loader = make_dataloader(dataset, "test", test, modalities, config)
        patients = []
        for batch in loader:   # inputs kept on CPU, moved to the device for each forward
            data = batch["input"]
            patients.append({"id": batch["patient_id"][0], "dataset": batch["dataset_name"][0], "data": data,
                             "available": {m for m in modalities if STATUS_KEYS[m] in data and bool(data[STATUS_KEYS[m]].item())},
                             "censorship": float(batch["censorship"].item()), "time": float(batch["original_event_time"].item())})
        with torch.no_grad():
            for m in modalities:
                owners = [p for p in patients if m in p["available"]]
                for p in owners:
                    donors = [q for q in owners if q["dataset"] == p["dataset"] and q["id"] != p["id"]]
                    if not donors:
                        continue
                    donor = donors[rng.integers(len(donors))]
                    shuffled = dict(p["data"])
                    for key in CONTENT_KEYS[m]:
                        if key in donor["data"]:
                            shuffled[key] = donor["data"][key]
                    removed = dict(p["data"])
                    removed[STATUS_KEYS[m]] = torch.zeros_like(removed[STATUS_KEYS[m]])
                    rows.append({"fold": k + 1, "modality": m, "patient": p["id"], "dataset": p["dataset"],
                                 "censorship": p["censorship"], "time": p["time"],
                                 "full": _risk(model, p["data"], device), "shuffled": _risk(model, shuffled, device),
                                 "removed": _risk(model, removed, device)})
        print(f"fold {k + 1}: counterfactuals on {len(patients)} test patients")
    df = pd.DataFrame(rows)
    out = os.path.join(run_dir, f"counterfactual_{checkpoint}.csv")
    df.to_csv(out, index=False)
    summary = []
    for m, g in df.groupby("modality"):
        presence = (g["removed"] - g["shuffled"]).values
        boot = [rng.choice(presence, len(presence)).mean() for _ in range(n_boot)]
        row = {"modality": m, "patients": len(g),
               "presence_effect": presence.mean(), "CI_low": np.percentile(boot, 2.5), "CI_high": np.percentile(boot, 97.5),
               "content_effect": (g["shuffled"] - g["full"]).abs().mean()}
        for variant in ("full", "shuffled", "removed"):   # c-index pooled over the folds' test patients
            row[f"c-index_{variant}"] = safe_c_index(g["censorship"].values, g["time"].values, g[variant].values)
        summary.append(row)
    summary = pd.DataFrame(summary).round(4)
    print("\nCounterfactual masking (risk = -sum of the survival curve; presence effect = removed - shuffled, "
          "95% bootstrap CI; a CI excluding 0 = the presence of the modality moves the risk beyond its content):\n"
          + summary.to_string(index=False))
    print(f"-> {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--missingness_baseline", action="store_true")
    ap.add_argument("--config", help="main yaml (missingness baseline)")
    ap.add_argument("--run_dir", help="results folder of a trained k-fold run (probes)")
    ap.add_argument("--checkpoint", default="last_epoch", choices=sorted(CHECKPOINTS))
    ap.add_argument("--counterfactual", action="store_true", help="with --run_dir: counterfactual masking instead of probes")
    args = ap.parse_args()
    if args.missingness_baseline:
        if not args.config:
            raise SystemExit("--missingness_baseline needs --config")
        sys.path.insert(0, REPO)
        from main import resolve_modalities   # data_loader.modalities -> model.kwargs.input_modalities
        missingness_baseline(resolve_modalities(munchify(yaml.safe_load(open(args.config)))))
    if args.run_dir and args.counterfactual:
        run_counterfactual(args.run_dir, args.checkpoint)
    elif args.run_dir:
        run_probes(args.run_dir, args.checkpoint)
    if not (args.missingness_baseline or args.run_dir):
        raise SystemExit("nothing to do: --missingness_baseline and/or --run_dir")


if __name__ == "__main__":
    main()
