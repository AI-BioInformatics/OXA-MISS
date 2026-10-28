"""
Patient-level analysis of an ablation stage: every variant against the reference, on the SAME test patients.

Reads the run folders under --results_dir (results/ABL2/<stage>/<variant>/<run>/, or any folder tree with
config.yaml + test_df__Fold_<k>_<checkpoint>[_<scenario>].csv), groups the runs by variant (config.yaml
ablation.variant, else the name of the run's parent folder) and for each variant:

  1. seed ensemble: the risks of each (run, fold) model are z-scored over its test patients (the scale /
     offset of a Cox log-risk differs between models), then averaged per patient over the seeds. Internal test:
     out-of-fold (each patient is in one test fold). External test (scenario external*): every fold model
     predicts the same patients, so the average also runs over the fold models (ensemble);
  2. metrics of the ensemble: pooled Harrell c-index, tumour-stratified c-index (pairs only within the same
     harmonized cancer type), c-index per cancer type, the same on the patients with radiology (CT or MRI in
     avail_pattern), and one row per availability pattern (n, events, stratified c-index);
  3. paired bootstrap against the reference: the patients of both ensembles (inner join) are resampled with
     replacement within each cancer type, --n_boot times; on each sample delta = C(variant) - C(reference)
     (stratified c-index). No retraining: the predictions are fixed, the patient sample varies. Reported:
     delta, 95% percentile CI, two-sided p (2 x the smaller tail), Holm over the variants (per subset).
     It answers "would the gain hold on another sample of patients from the same population"; the training
     randomness is in the seed ensemble (and in utils/ablation_analysis.py, fold-level).

Runs without the avail_pattern column (written before 2026-10-27) have no radiology subset / pattern table.

Usage (CPU, ~20 s per variant with 2000 resamples and ~3.5k patients; inside a SLURM allocation, BLAS threads
capped to the allocated cores):
    OMP_NUM_THREADS=2 python utils/patient_level_analysis.py --results_dir /work/H2020DeciderFicarra/ccRCC/results/ABL2/S1_architecture \
        [--reference default] [--checkpoint Lowest_Validation_Loss_Model] [--scenario base] [--n_boot 2000]
    S6 + S6b (stage 2 primary checkpoint = last epoch):
        ... --results_dir <project_dir>/ABL2/S6_radiology \
            --checkpoint_of two_stage_frozen=Last_Epoch_Model two_stage_finetune=Last_Epoch_Model
Output: <results_dir>/patient_level_<checkpoint>_<scenario>_{metrics,patterns,bootstrap}.csv (+ .md).
"""
import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from experiments.utils import cancer_type_of, safe_c_index, stratified_c_index  # noqa: E402

RADIOLOGY = ("CT", "MRI")
KEY = ["dataset_name", "patient_ids"]


def run_variant(run_dir):
    """(variant, seed) of a run folder."""
    cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml"))) or {}
    variant = (cfg.get("ablation") or {}).get("variant") or os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    return variant, cfg.get("seed")


def load_run(run_dir, checkpoint, scenario):
    """Test predictions of all the folds of a run, one row per (fold, patient), risks z-scored per fold model."""
    suffix = "" if scenario == "base" else "_" + scenario.replace("/", "_")
    pattern = re.compile(rf"test_df__Fold_(\d+)_{re.escape(checkpoint)}{re.escape(suffix)}\.csv$")
    folds = []
    for path in sorted(glob.glob(os.path.join(run_dir, "test_df__Fold_*.csv"))):
        m = pattern.search(os.path.basename(path))
        if not m:
            continue
        df = pd.read_csv(path)
        r = df["all_risk_scores"]
        df["risk_z"] = (r - r.mean()) / (r.std() if r.std() > 0 else 1.0)
        folds.append(df.assign(fold=int(m.group(1))))
    return pd.concat(folds, ignore_index=True) if folds else None


def ensemble(runs):
    """Seed (and, external, fold) ensemble: mean z-scored risk per patient."""
    df = pd.concat(runs, ignore_index=True)
    agg = dict(risk=("risk_z", "mean"), censorship=("all_censorships", "first"),
               time=("all_original_event_times", "first"), n_models=("risk_z", "size"))
    if "avail_pattern" in df:
        agg["avail_pattern"] = ("avail_pattern", "first")
    out = df.groupby(KEY, as_index=False).agg(**agg)
    out["cancer"] = out["dataset_name"].map(cancer_type_of)
    if "avail_pattern" in out:
        out["has_radiology"] = out["avail_pattern"].fillna("").str.split("+").apply(
            lambda mods: any(m in RADIOLOGY for m in mods))
    return out


def c_stats(d):
    return {"n": len(d), "events": int((d["censorship"] == 0).sum()),
            "c_pooled": safe_c_index(d["censorship"], d["time"], d["risk"]),
            "c_stratified": stratified_c_index(d["censorship"], d["time"], d["risk"], d["cancer"])}


def subsets(d):
    yield "all", d
    if "has_radiology" in d:
        yield "radiology", d[d["has_radiology"]]


def _pair_matrices(time, event, risks, tied_tol=1e-8):
    """Harrell pairs of one stratum: comparable[i, j] = event_i and time_i < time_j (as sksurv); score[k][i, j] =
    1 concordant (risk_i > risk_j), 0.5 tied risk, 0 discordant, for each risk vector k."""
    comparable = (event[:, None] & (time[:, None] < time[None, :])).astype(float)
    scores = []
    for r in risks:
        diff = r[:, None] - r[None, :]
        scores.append(comparable * np.where(np.abs(diff) <= tied_tol, 0.5, (diff > 0).astype(float)))
    return comparable, scores


def bootstrap(ref, var, n_boot, rng):
    """Paired bootstrap of the stratified c-index difference on the common patients, resampled within cancer.
    A resample = multiplicity w_i of each patient: concordant = w' S w, comparable = w' C w per stratum (pairs
    of copies of the same patient are never comparable, as in the c-index of the resampled data)."""
    m = ref.merge(var[KEY + ["risk"]], on=KEY, suffixes=("_ref", "_var"))
    if len(m) < 2:
        return None
    strata = []
    for c in np.unique(m["cancer"].values):
        g = m[m["cancer"].values == c]
        comparable, (s_var, s_ref) = _pair_matrices(g["time"].values.astype(float), (g["censorship"].values == 0),
                                                    [g["risk_var"].values, g["risk_ref"].values])
        strata.append((len(g), comparable, s_var, s_ref))

    def delta_of(weights):
        num_v = num_r = den = 0.0
        for w, (_, comparable, s_var, s_ref) in zip(weights, strata):
            den += w @ comparable @ w
            num_v += w @ s_var @ w
            num_r += w @ s_ref @ w
        return (num_v - num_r) / den if den > 0 else np.nan

    delta = delta_of([np.ones(n) for n, *_ in strata])
    deltas = np.array([delta_of([np.bincount(rng.integers(0, n, n), minlength=n).astype(float) for n, *_ in strata])
                       for _ in range(n_boot)])
    deltas = deltas[~np.isnan(deltas)]
    p = min(1.0, 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())) if len(deltas) else np.nan
    lo, hi = np.percentile(deltas, [2.5, 97.5]) if len(deltas) else (np.nan, np.nan)
    return {"n_common": len(m), "events": int((m["censorship"] == 0).sum()), "delta": delta,
            "ci_low": lo, "ci_high": hi, "p": p, "n_boot": len(deltas)}


def holm(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = np.flatnonzero(~np.isnan(p))
    order = ok[np.argsort(p[ok])]
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(order) - rank) * p[i]))
        out[i] = running
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True)
    ap.add_argument("--reference", default="default")
    ap.add_argument("--checkpoint", default="Lowest_Validation_Loss_Model",
                    help="Lowest_Validation_Loss_Model | Highest_Validation_Metric_Model | Last_Epoch_Model")
    ap.add_argument("--checkpoint_of", nargs="*", default=[], metavar="VARIANT=CHECKPOINT",
                    help="per-variant checkpoint, e.g. two_stage_frozen=Last_Epoch_Model (S6b: fixed epochs)")
    ap.add_argument("--scenario", default="base", help="base | <missing scenario> | external[/...]")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="output folder (default: --results_dir)")
    args = ap.parse_args()

    checkpoint_of = dict(item.split("=", 1) for item in args.checkpoint_of)
    runs = {}
    for cfg_path in sorted(glob.glob(os.path.join(args.results_dir, "**", "config.yaml"), recursive=True)):
        run_dir = os.path.dirname(cfg_path)
        variant, seed = run_variant(run_dir)
        df = load_run(run_dir, checkpoint_of.get(variant, args.checkpoint), args.scenario)
        if df is None:
            continue
        runs.setdefault(variant, []).append(df.assign(seed=seed))
    if not runs:
        sys.exit(f"no test_df files for checkpoint {args.checkpoint} / scenario {args.scenario} under {args.results_dir}")
    if args.reference not in runs:
        sys.exit(f"reference {args.reference} not found; variants: {sorted(runs)}")
    print(f"{len(runs)} variants: " + ", ".join(f"{v} ({len(r)} runs)" for v, r in sorted(runs.items())))

    ens = {v: ensemble(r) for v, r in runs.items()}
    metric_rows, pattern_rows = [], []
    for v, d in ens.items():
        for name, s in subsets(d):
            metric_rows.append({"variant": v, "subset": name, "n_runs": len(runs[v]), **c_stats(s)})
        for cancer, s in d.groupby("cancer"):
            metric_rows.append({"variant": v, "subset": f"cancer:{cancer}", "n_runs": len(runs[v]), **c_stats(s)})
        if "avail_pattern" in d:
            for pattern, s in d.groupby("avail_pattern"):
                pattern_rows.append({"variant": v, "pattern": pattern, **c_stats(s)})

    rng = np.random.default_rng(args.seed)
    boot_rows = []
    for name in ("all", "radiology"):
        rows = []
        ref = dict(subsets(ens[args.reference])).get(name)
        if ref is None:
            continue
        for v, d in sorted(ens.items()):
            if v == args.reference:
                continue
            var = dict(subsets(d)).get(name)
            res = bootstrap(ref, var, args.n_boot, rng) if var is not None else None
            if res:
                rows.append({"variant": v, "subset": name, **res})
        for r, ph in zip(rows, holm([r["p"] for r in rows])):
            r["p_holm"] = ph
            r["effect"] = "n.s." if not ph < 0.05 else ("better" if r["delta"] > 0 else "worse")
        boot_rows += rows

    out_dir = args.out or args.results_dir
    stem = os.path.join(out_dir, f"patient_level_{args.checkpoint}_{args.scenario.replace('/', '_')}")
    old = os.umask(0o007)
    try:
        for tag, rows in (("metrics", metric_rows), ("patterns", pattern_rows), ("bootstrap", boot_rows)):
            if not rows:
                continue
            table = pd.DataFrame(rows).round(4)
            table.to_csv(f"{stem}_{tag}.csv", index=False)
            with open(f"{stem}_{tag}.md", "w") as f:
                f.write(table.to_markdown(index=False))
            if tag != "patterns":
                shown = table if tag == "bootstrap" else table[~table["subset"].str.startswith("cancer:")]
                print(f"\n== {tag} ==\n{shown.to_string(index=False)}")
    finally:
        os.umask(old)
    print(f"\nwritten {stem}_*.csv / .md")


if __name__ == "__main__":
    main()
