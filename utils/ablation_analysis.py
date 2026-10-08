"""
Statistics of an ablation grid (grid_search/make_ablation_grid.py): every variant against the default model,
paired on the same seed and fold.

Reads every run folder <results_dir>/<run>/ (results_dir = <project_dir>/ABL_<preset>): config.yaml (variant =
the "module:<variant>" wandb tag, seed) and results_folds.csv (per-fold test metrics of each checkpoint, written
at the end of the k-fold).

For each variant, checkpoint and metric:
    variant         mean +- std over (seed, fold) of the metric
    diff            mean paired difference variant - default (same seed, same fold)
    CI 95%          bootstrap over the (seed, fold) pairs
    p / p_holm      Wilcoxon signed-rank test on the paired differences; Holm correction over the variants
                    (per metric and checkpoint)
    effect          "better" / "worse" than the default when p_holm < --alpha (c-index / Uno: higher is better;
                    IBS / D-cal chi2: lower is better), else "n.s."
    n_pairs         number of (seed, fold) pairs (expected: seeds x folds)
Folds of the same seed share training patients, so the pairs are not fully independent: the CIs and p-values
are somewhat optimistic; the same holds for every variant, so the ranking is still fair.

Usage (CPU, seconds):
    python utils/ablation_analysis.py --results_dir /work/H2020DeciderFicarra/ccRCC/results/ABL_pancancer8 \
        [--checkpoints last_epoch lowest_val_loss highest_val_metric] [--metrics c-index c-index_uno IBS D-cal_stat]
    radiology input grid (new folder) against the default runs of the main grid:
    python utils/ablation_analysis.py --results_dir <project_dir>/ABL_ccRCC_radiology <project_dir>/ABL_ccRCC \
        --variants radiology_suprem_mrseg radiology_off --checkpoints last_epoch
Output: printed table + <results_dir>/ablation_statistics.csv (and .md).
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml

CHECKPOINTS = {"Last_Epoch_Model": "last_epoch", "Lowest_Validation_Loss_Model": "lowest_val_loss",
               "Highest_Validation_Metric_Model": "highest_val_metric"}
LOWER_IS_BETTER = {"IBS", "D-cal_stat"}


def load_runs(results_dir, scenario_suffix=""):
    """Long table: variant, seed, run, checkpoint, fold, metrics (one row per run, checkpoint and fold)."""
    rows, skipped = [], []
    for config_path in sorted(glob.glob(os.path.join(results_dir, "*", "config.yaml"))):
        run_dir = os.path.dirname(config_path)
        folds_path = os.path.join(run_dir, f"results_folds{scenario_suffix}.csv")
        if not os.path.exists(folds_path):
            skipped.append(os.path.basename(run_dir))   # unfinished / failed run
            continue
        with open(config_path) as f:
            config = yaml.safe_load(f)
        tags = (config.get("wandb") or {}).get("tags") or []
        variant = next((t.split(":", 1)[1] for t in tags if t.startswith("module:")), None)
        if variant is None:
            skipped.append(os.path.basename(run_dir) + " (no module tag)")
            continue
        folds = pd.read_csv(folds_path)
        folds["checkpoint"] = folds["model_version"].map(CHECKPOINTS).fillna(folds["model_version"])
        folds["variant"], folds["seed"], folds["run"] = variant, config.get("seed"), os.path.basename(run_dir)
        rows.append(folds)
    if skipped:
        print(f"[!] {len(skipped)} run folders without results_folds{scenario_suffix}.csv (unfinished?) or module tag: "
              f"{skipped[:5]}{' ...' if len(skipped) > 5 else ''}")
    if not rows:
        raise SystemExit(f"no finished ablation runs in {results_dir}")
    return pd.concat(rows, ignore_index=True)


def holm(pvalues):
    """Holm-Bonferroni adjusted p-values (NaN kept)."""
    p = np.asarray(pvalues, dtype=float)
    valid = ~np.isnan(p)
    adjusted = np.full_like(p, np.nan)
    order = np.argsort(p[valid])
    m = valid.sum()
    running = 0.0
    idx = np.flatnonzero(valid)[order]
    for rank, i in enumerate(idx):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def compare(data, baseline, checkpoints, metrics, n_boot, rng):
    from scipy.stats import wilcoxon
    out = []
    for checkpoint in checkpoints:
        d = data[data["checkpoint"] == checkpoint]
        base = d[d["variant"] == baseline]
        if base.empty:
            print(f"[!] no {baseline} runs for checkpoint {checkpoint}")
            continue
        for metric in metrics:
            if metric not in d.columns:
                continue
            rows = []
            base_values = base.set_index(["seed", "fold"])[metric]
            for variant, g in d.groupby("variant"):
                values = g.set_index(["seed", "fold"])[metric]
                row = {"checkpoint": checkpoint, "metric": metric, "variant": variant,
                       "runs": g["run"].nunique(), "mean": values.mean(), "std": values.std(ddof=0)}
                if variant != baseline:
                    paired = pd.concat([values.rename("v"), base_values.rename("b")], axis=1, join="inner").dropna()
                    diff = (paired["v"] - paired["b"]).to_numpy()
                    row["n_pairs"] = len(diff)
                    if len(diff):
                        boot = [rng.choice(diff, len(diff)).mean() for _ in range(n_boot)]
                        row.update(diff=diff.mean(), ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5))
                        try:
                            row["p"] = wilcoxon(diff).pvalue if np.any(diff != 0) else 1.0
                        except ValueError:
                            row["p"] = np.nan
                        row["better"] = bool(diff.mean() < 0) if metric in LOWER_IS_BETTER else bool(diff.mean() > 0)
                rows.append(row)
            table = pd.DataFrame(rows)
            mask = table["variant"] != baseline
            if "p" in table and mask.any():
                table.loc[mask, "p_holm"] = holm(table.loc[mask, "p"].to_numpy())
            out.append(table)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results_dir", required=True, nargs="+",
                    help="<project_dir>/ABL_<preset> (several: their runs are pooled, the output goes to the first)")
    ap.add_argument("--baseline", default="default")
    ap.add_argument("--variants", nargs="+", default=None, help="only these variants (+ the baseline); default all")
    ap.add_argument("--checkpoints", nargs="+", default=["last_epoch", "lowest_val_loss", "highest_val_metric"])
    ap.add_argument("--metrics", nargs="+", default=["c-index", "c-index_uno", "IBS", "D-cal_stat"])
    ap.add_argument("--scenario", default="base", help="test scenario (base, or e.g. test_WSI+Genomics)")
    ap.add_argument("--n_boot", type=int, default=10000)
    ap.add_argument("--alpha", type=float, default=0.05, help="significance level of the Holm-corrected p-values")
    args = ap.parse_args()
    suffix = "" if args.scenario == "base" else "_" + args.scenario.replace("/", "_")
    data = pd.concat([load_runs(d, suffix) for d in args.results_dir], ignore_index=True)
    if args.variants:
        data = data[data["variant"].isin([args.baseline, *args.variants])]
    counts = data.groupby("variant")["run"].nunique()
    print(f"{data['run'].nunique()} finished runs, {len(counts)} variants; runs per variant: {counts.to_dict()}")
    table = compare(data, args.baseline, args.checkpoints, args.metrics, args.n_boot, np.random.default_rng(0))
    if table.empty:
        raise SystemExit("nothing to compare")
    out = os.path.join(args.results_dir[0], f"ablation_statistics{suffix}.csv")
    table.to_csv(out, index=False)
    with open(out.replace(".csv", ".md"), "w") as f:
        for (checkpoint, metric), t in table.groupby(["checkpoint", "metric"], sort=False):
            f.write(f"\n### {metric} ({checkpoint})\n\n")
            f.write(fmt(t, args.baseline, args.alpha).to_markdown(index=False) + "\n")
    for (checkpoint, metric), t in table.groupby(["checkpoint", "metric"], sort=False):
        print(f"\n=== {metric} | checkpoint {checkpoint} (diff = variant - {args.baseline}, paired on seed x fold)")
        print(fmt(t, args.baseline, args.alpha).to_string(index=False))
    print(f"\n-> {out} (and .md)")


def fmt(t, baseline, alpha=0.05):
    """Readable table: baseline first, then variants sorted by difference."""
    t = t.copy()
    t["order"] = (t["variant"] != baseline).astype(int)
    t = t.sort_values(["order", "diff"], ascending=[True, False], na_position="first")
    show = pd.DataFrame({"variant": t["variant"], "runs": t["runs"], "mean ± std": [f"{m:.3f} ± {s:.3f}" for m, s in zip(t["mean"], t["std"])]})
    if "diff" in t:
        show["diff [95% CI]"] = [f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]" if pd.notna(d) else "" for d, lo, hi in
                                 zip(t["diff"], t.get("ci_low", np.nan), t.get("ci_high", np.nan))]
        show["p"] = [f"{p:.3g}" if pd.notna(p) else "" for p in t.get("p", np.nan)]
        show["p_holm"] = [f"{p:.3g}" if pd.notna(p) else "" for p in t.get("p_holm", np.nan)]
        show["effect"] = [("" if not isinstance(b, (bool, np.bool_)) else "n.s." if not (pd.notna(p) and p < alpha)
                           else "better" if b else "worse") for b, p in zip(t.get("better", ""), t.get("p_holm", np.nan))]
        show["pairs"] = [f"{int(n)}" if pd.notna(n) else "" for n in t.get("n_pairs", np.nan)]
    return show


if __name__ == "__main__":
    main()
