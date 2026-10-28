"""
Leakage check of the clinical data against the survival endpoint (time = FUT, event = Survival), per cohort
(CPTAC-CCRCC + the 8 TCGA cohorts of pancancer8; cohorts and columns as utils/clinical_survival_association.py,
but EVERY column is checked, also the outcome-related / calendar / administrative ones that are not tested there).

A column leaks when it carries the endpoint itself rather than information available at diagnosis:
    rho_time      Spearman correlation of a numerical column with the survival time (all patients)
    match_time    fraction of patients whose value equals the survival time (within 1 day, or in months
                  within 1 month): a copy of the label
    auc_event     how well the column alone separates dead from alive (ROC AUC, 0.5 = not at all; numerical:
                  the value; categorical: the in-sample event rate of its level, optimistic)
    miss_*        the column's MISSINGNESS against the endpoint (the model sees a "missing" token): event rate of
                  the patients with / without a value, Fisher p; median survival time with / without, Mann-Whitney p
Flags:
    LEAK          |rho_time| >= 0.8, match_time >= 0.5 or auc_event >= 0.9: encodes the label
    SUSPECT       auc_event >= 0.75, or missingness with an event-rate gap >= 0.15 and p < 0.001: check before
                  using it as an input
The model inputs (clean table: age, sex, stage, T, N, M) are reported in a separate table with their missingness.

Usage (CPU, about a minute):
    python utils/clinical_leakage_check.py [--out_dir /work/H2020DeciderFicarra/ccRCC/results/clinical_leakage_check]
Output: leakage_check.csv (every column x cohort), leakage_check.md (flagged columns, model inputs)
"""
import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import clinical_survival_association as csa  # noqa: E402

warnings.filterwarnings("ignore")
MODEL_INPUTS = [f"model: {c}" for c in csa.SELECTED]


def auc(y, score):
    if len(np.unique(y)) < 2 or len(np.unique(score)) < 2:
        return np.nan
    a = roc_auc_score(y, score)
    return max(a, 1 - a)


def check_column(values, time, event, min_patients):
    """Leakage metrics of one column of one cohort (None: too few values)."""
    v = csa.clean_values(values)
    present = v.notna()
    n = int(present.sum())
    if n < min_patients:
        return None
    row = {"n": n, "missing": round(1 - present.mean(), 3)}
    num = pd.to_numeric(v, errors="coerce")
    if num.notna().sum() >= 0.9 * n:
        ok = num.notna()                       # the few non-numerical values left out
        x, t, e = num[ok], time[ok], event[ok]
        row["kind"] = "numerical"
        row["rho_time"] = spearmanr(x, t).correlation if x.nunique() > 1 else np.nan
        row["match_time"] = float(np.mean((np.abs(x - t) <= 1) | (np.abs(x * 30.4375 - t) <= 31)))
        row["auc_event"] = auc(e, x)
    else:
        text = v[present].astype(str).str.strip().str.lower()
        if text.nunique() > 0.9 * n:
            row["kind"] = "identifier"
        else:
            row["kind"] = "categorical"
            rate = event[present].groupby(text).transform("mean")   # event rate of the patient's level
            row["auc_event"] = auc(event[present], rate)
    if 0.05 <= row["missing"] <= 0.95:
        e_in, e_out = event[present], event[~present]
        table = [[int(e_in.sum()), int((1 - e_in).sum())], [int(e_out.sum()), int((1 - e_out).sum())]]
        row["miss_event_rate_present"] = round(e_in.mean(), 3)
        row["miss_event_rate_missing"] = round(e_out.mean(), 3)
        row["miss_event_p"] = fisher_exact(table)[1]
        row["miss_time_median_present"] = float(time[present].median())
        row["miss_time_median_missing"] = float(time[~present].median())
        row["miss_time_p"] = mannwhitneyu(time[present], time[~present]).pvalue
    flags = []
    if (abs(row.get("rho_time", 0) or 0) >= 0.8 or row.get("match_time", 0) >= 0.5
            or (row.get("auc_event") or 0) >= 0.9):
        flags.append("LEAK")
    elif (row.get("auc_event") or 0) >= 0.75:
        flags.append("SUSPECT (separates the event)")
    gap = abs(row.get("miss_event_rate_present", 0) - row.get("miss_event_rate_missing", 0))
    if "miss_event_p" in row and gap >= 0.15 and row["miss_event_p"] < 0.001:
        flags.append("SUSPECT (missingness)")
    row["flag"] = "; ".join(flags)
    return row


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=f"{csa.ROOT}/results/clinical_leakage_check")
    ap.add_argument("--min_patients", type=int, default=30)
    args = ap.parse_args()

    outcome = csa.load_outcome()
    cases = outcome.index
    tcga, cptac = csa.load_raw_tcga(cases), csa.load_raw_cptac(cases)
    features = pd.concat([csa.load_model_table(cases), csa.harmonized(tcga, cptac), tcga, cptac], axis=1)
    rows = []
    for cohort in outcome.study.unique():
        mask = outcome.study.eq(cohort)
        time, event = outcome.time[mask], outcome.event[mask]
        for name in features.columns:
            source = name.split(": ")[0]
            if (source == "TCGA" and not cohort.startswith("TCGA_")) or (source == "CPTAC" and not cohort.startswith("CPTAC")):
                continue
            res = check_column(features.loc[mask, name], time, event, args.min_patients)
            if res is None:
                continue
            raw = name.split(": ", 1)[1]
            known = ("outcome-related" if csa.LEAK.search(raw) else "calendar" if csa.CALENDAR.search(raw)
                     else "administrative" if csa.ADMIN.search(raw) else "")
            rows.append({"cohort": cohort, "feature": name, "model_input": name in MODEL_INPUTS,
                         "excluded_before_as": known, **res})
    table = pd.DataFrame(rows)
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "leakage_check.csv"), index=False)

    fmt = lambda v, d=2: "" if pd.isna(v) else f"{v:.{d}f}"
    with open(os.path.join(args.out_dir, "leakage_check.md"), "w") as f:
        f.write("## Model inputs (clean table)\n\n")
        m = table[table.model_input]
        f.write(pd.DataFrame({
            "cohort": m.cohort, "feature": m.feature.str.replace("model: ", ""), "n": m.n,
            "missing": m.missing.map(lambda v: f"{v:.0%}"),
            "rho_time": m.rho_time.map(fmt), "auc_event": m.auc_event.map(fmt),
            "event rate present / missing": [
                "" if pd.isna(a) else f"{a:.2f} / {b:.2f} (p={p:.1g})"
                for a, b, p in zip(m.get("miss_event_rate_present"), m.get("miss_event_rate_missing"), m.get("miss_event_p"))],
            "flag": m.flag}).to_markdown(index=False) + "\n\n")
        f.write("## Flagged columns (all sources)\n\n")
        flagged = table[table.flag != ""].sort_values(["flag", "cohort"])
        f.write(pd.DataFrame({
            "cohort": flagged.cohort, "feature": flagged.feature, "excluded before as": flagged.excluded_before_as,
            "kind": flagged.kind, "n": flagged.n, "rho_time": flagged.rho_time.map(fmt),
            "match_time": flagged.match_time.map(fmt), "auc_event": flagged.auc_event.map(fmt),
            "event rate present / missing": [
                "" if pd.isna(a) else f"{a:.2f} / {b:.2f}"
                for a, b in zip(flagged.get("miss_event_rate_present"), flagged.get("miss_event_rate_missing"))],
            "flag": flagged.flag}).to_markdown(index=False) + "\n")
    print(open(os.path.join(args.out_dir, "leakage_check.md")).read())
    print(f"-> {args.out_dir}/leakage_check.csv, leakage_check.md")


if __name__ == "__main__":
    main()
