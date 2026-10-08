"""
Univariate association of every clinical feature with the survival outcome of the model (OS: time = FUT,
event = Survival, 1 = dead; label files of the dataset yamls), per cohort: CPTAC-CCRCC and the 8 TCGA cohorts of
pancancer8 (KIRC, BLCA, COADREAD, KIRP, LIHC, LUAD, LUSC, STAD).

Features:
    model table   the clean table read by the model (clinical_data_clean/all_studies_clinical_clean.csv):
                  age, sex, ajcc_stage, ajcc_t, ajcc_n, ajcc_m (selected, data_loader.clinical) + race, ethnicity
    harmonized    grade G1..G4 (TCGA Neoplasm Histologic Grade / CPTAC tumor_grade), comparable across cohorts
    raw TCGA      every column of TCGA_clinical_data_COMPLETE.tsv (cBioPortal; columns differ by cancer type)
    raw CPTAC     every column of CPTAC-3.clinical.tsv (GDC/Xena, several rows per case: collapsed to one row per
                  patient, tumour samples first, first non-missing value)
    Dropped: identifiers / dates / free text (almost one value per patient), constant columns, columns with
    < --min_patients patients or < --min_events events. Outcome-derived columns (vital status, death, follow-up,
    disease-free, progression, ...) would leak the outcome: listed as such, not tested.

Per feature and cohort (complete cases of that feature):
    numerical  Cox PH on the z-scored value: HR per 1 SD [95% CI], Wald p
    ordinal    stage / T / N / M / grade parsed to 1..4 (0..3 for N, 0..1 for M; X / unknown = missing):
               HR per step, Wald p
    categorical one-hot, reference = most frequent level, levels with < --min_level patients pooled into "other":
               likelihood-ratio p (all levels), HR of each level vs the reference
    C-index    Harrell's, of the fitted linear predictor (0.5 = no association)
    q          Benjamini-Hochberg over the tested features of that cohort
Pooled ccRCC (CPTAC + KIRC) and pooled pancancer8 (the 8 TCGA cohorts): model-table + harmonized features (the
raw columns differ between cohorts), Cox stratified by cohort (own baseline hazard per cohort).
summary_wide.md: features x cohorts, HR per step / SD (* = q < 0.05) for every feature tested in >= 3 cohorts.

Usage (CPU, about a minute):
    python utils/clinical_survival_association.py [--out_dir /work/H2020DeciderFicarra/ccRCC/results/clinical_survival_association]
Output: associations.csv (all rows), associations.md (tables per cohort), summary_wide.md / .csv,
        excluded_columns.csv
"""
import argparse
import os
import re
import warnings

import numpy as np
import pandas as pd
import yaml
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

warnings.filterwarnings("ignore")

ROOT = "/work/H2020DeciderFicarra/ccRCC"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# dataset yamls of the cohorts (ccRCC base config + pancancer8, grid_search/make_ablation_grid.py)
DATASET_YAMLS = [f"{REPO}/config/ccRCC.yaml"] + [f"{REPO}/config/TCGA_{c}_dataset_UNI.yaml" for c in
                                                 ("KIRC", "BLCA", "COAD", "KIRP", "LIHC", "LUAD", "LUSC", "STAD")]
PANCANCER8 = ["TCGA_KIRC", "TCGA_BLCA", "TCGA_COADRED", "TCGA_KIRP", "TCGA_LIHC", "TCGA_LUAD", "TCGA_LUSC",
              "TCGA_STAD"]
POOLS = {"pooled ccRCC (CPTAC + KIRC)": ["CPTAC", "TCGA_KIRC"], "pooled pancancer8": PANCANCER8}
CLEAN = f"{ROOT}/clinical_data_clean/all_studies_clinical_clean.csv"
RAW_TCGA = f"{ROOT}/TCGA_clinical_data_COMPLETE.tsv"
RAW_CPTAC = f"{ROOT}/CPTAC-3.clinical.tsv"
SELECTED = ["age", "sex", "ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]   # data_loader.clinical of the model

MISSING = {"", "nan", "none", "na", "n/a", "not reported", "unknown", "not available", "[not available]",
           "[unknown]", "[not applicable]", "[not evaluated]", "[discrepancy]", "[completed]", "not applicable",
           "unknown tumor status", "not allowed to collect", "--", "'--", "indeterminate", "[]"}
# outcome / follow-up information: associated with OS by construction
LEAK = re.compile(r"surviv|death|dead|vital|follow|last.?alive|last.?communication|last.?known|disease.?free|"
                  r"progress|recurren|neoplasm.?status|lost.?to|treatment.?outcome|therapy.?outcome|"
                  r"new.?neoplasm|cause_of", re.I)
# identifiers, administrative and technical fields: no clinical meaning
ADMIN = re.compile(r"(^|[ ._])(id|ids|uuid)($|[ ._])|submitter|datetime|^state|file.?name|report|barcode|"
                   r"consent|form.?completion|project|program|^sample$|sample.?id|patient.?id|other.?(patient|sample)|"
                   r"annotation|vial|oct.?embedded|freezing|preservation|is.?ffpe|composition|tumor_code|"
                   r"sample.?type|tissue.?type|specimen.?type|ordinal|study.?id|cancer.?type|oncotree|"
                   r"disease.?type|primary.?site|icd|somatic.?status|number.?of.?samples|collection.?indicator|"
                   r"procurement|sample.?collection|pathology_detail_id|treatment_id|days.?to.?consent|xena|obfuscated", re.I)
# calendar dates (birth / diagnosis / smoking years): tied to the length of follow-up, not to the patient
CALENDAR = re.compile(r"year_of|onset_year|quit_year|(started|stopped) smoking year|year cancer initial|"
                      r"days_to_(birth|sample)|date", re.I)
# codes that look numeric in some cohorts (site codes, category codes): categorical
CODES = re.compile(r"tissue.?source.?site|history.?category|status.?category|timepoint", re.I)
# what a feature is, for the table (selected = data_loader.clinical of the model)
GROUPS = [
    (re.compile(r"treatment|therapy|adjuvant|radiotherapy", re.I), "treatment (after diagnosis)"),
    (re.compile(r"tissue.?source.?site|publication.?version|edition", re.I), "site / calendar (non-clinical)"),
    (re.compile(r"mutation|tmb|fraction.?genome", re.I), "molecular summary"),
]


def feature_group(name):
    source, raw = name.split(": ", 1)
    if source == "model":
        return "selected" if raw in SELECTED else "clean table, not selected"
    if source == "harmonized":
        return "clinical, not selected"
    for pattern, group in GROUPS:
        if pattern.search(raw):
            return group
    return "clinical, not selected"
ORDINAL = [  # (column-name pattern, value parser)
    (re.compile(r"stage(?!.*(version|edition|system))|group.?stage", re.I), "stage"),
    (re.compile(r"grade", re.I), "grade"),
]


def parse_ordinal(value, kind):
    """stage I..IV -> 1..4; T1..T4 -> 1..4; N0..N3 -> 0..3; M0/M1 -> 0/1; G1..G4 -> 1..4; else NaN."""
    if not isinstance(value, str):
        return np.nan
    v = value.strip().lower()
    if kind == "grade":
        m = re.search(r"g\s*([1-4])", v)
        return float(m.group(1)) if m else np.nan
    m = re.search(r"stage\s*(iv|iii|ii|i)(?![iv])", v)
    if m:
        return float({"i": 1, "ii": 2, "iii": 3, "iv": 4}[m.group(1)])
    m = re.match(r"^[pcy]?t\s*([0-4])", v)
    if m:
        return float(m.group(1)) if m.group(1) != "0" else np.nan
    m = re.match(r"^[pc]?n\s*([0-3])", v)
    if m:
        return float(m.group(1))
    m = re.match(r"^[pc]?m\s*([01])", v)
    if m:
        return float(m.group(1))
    return np.nan


def clean_values(s):
    s = s.astype(object).where(s.notna(), None)
    return s.map(lambda v: np.nan if v is None or str(v).strip().lower() in MISSING else v)


def load_outcome():
    rows = []
    for path in DATASET_YAMLS:
        with open(path) as f:
            y = yaml.safe_load(f)
        par = y["parameters"]
        d = pd.read_csv(par["dataframe_path"], sep="\t").drop_duplicates(par.get("case_id_name", "case_id"))
        rows.append(pd.DataFrame({"case_id": d[par.get("case_id_name", "case_id")], "study": y["name"],
                                  "time": d[par["label_name"]].astype(float),
                                  "event": d[par["event_name"]].astype(int)}))
    out = pd.concat(rows).set_index("case_id")
    return out[out.time > 0]


def load_model_table(cases):
    d = pd.read_csv(CLEAN).drop_duplicates("case_id").set_index("case_id")
    d = d.reindex(cases)[["age", "sex", "race", "ethnicity", "ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]]
    return d.add_prefix("model: ")


def load_raw_tcga(cases):
    t = pd.read_csv(RAW_TCGA, sep="\t", low_memory=False)
    t = t.drop_duplicates("Patient ID")             # TCGA patient ids are unique across the studies
    return t.set_index("Patient ID").reindex(cases).add_prefix("TCGA: ")


def load_raw_cptac(cases):
    c = pd.read_csv(RAW_CPTAC, sep="\t", low_memory=False)
    c = c[c.submitter_id.isin(cases)].copy()
    c["_tumour"] = c.get("tissue_type.samples", pd.Series(index=c.index, dtype=object)).astype(str).str.lower().eq("tumor")
    c = c.sort_values("_tumour", ascending=False).drop(columns="_tumour")
    for col in c.columns:
        c[col] = clean_values(c[col])
    c = c.groupby("submitter_id").first()
    return c.reindex(cases).add_prefix("CPTAC: ")


def harmonized(tcga, cptac):
    """Features comparable across the cohorts, beyond the model table."""
    grade = tcga["TCGA: Neoplasm Histologic Grade"].where(tcga["TCGA: Neoplasm Histologic Grade"].notna(),
                                                          cptac["CPTAC: tumor_grade.diagnoses"])
    return pd.DataFrame({"harmonized: grade": grade})


def feature_kind(name, s, min_level):
    """('numerical' | 'ordinal' | 'categorical', values) or (None, reason)."""
    raw = name.split(": ", 1)[1]
    if LEAK.search(raw):
        return None, "outcome-related"
    if ADMIN.search(raw):
        return None, "identifier / administrative"
    s = clean_values(s)
    nonnull = s.dropna()
    if nonnull.empty:
        return None, "empty"
    for pattern, kind in ORDINAL:
        if pattern.search(raw) or (kind == "stage" and re.search(r"ajcc_(pathologic_|clinical_)?[tnm]($|\.)", raw, re.I)):
            parsed = s.map(lambda v: parse_ordinal(v, kind))
            if parsed.notna().sum() >= 0.8 * len(nonnull) and parsed.nunique() >= 2:
                return "ordinal", parsed.astype(float)
    numeric = pd.to_numeric(nonnull, errors="coerce")
    if numeric.notna().mean() >= 0.9 and not CODES.search(raw):
        values = pd.to_numeric(s, errors="coerce")
        if values.nunique() < 2:
            return None, "constant"
        if CALENDAR.search(raw):
            return None, "date / calendar"
        return "numerical", values
    text = s.map(lambda v: str(v).strip().lower() if isinstance(v, str) or not pd.isna(v) else np.nan)
    if text.dropna().nunique() > 0.5 * len(text.dropna()) and text.dropna().nunique() > 20:
        return None, "free text / identifier"
    counts = text.value_counts()
    rare = counts.index[counts < min_level]
    text = text.where(~text.isin(rare), "other")
    if text.value_counts().ge(min_level).sum() < 2:
        return None, "constant"
    return "categorical", text


def fit_cox(df, strata=None):
    for penalizer in (0.0, 0.01, 0.1):
        try:
            cph = CoxPHFitter(penalizer=penalizer)
            cph.fit(df, duration_col="time", event_col="event", strata=strata)
            return cph
        except Exception:
            continue
    return None


def associate(values, kind, outcome, strata=None):
    df = outcome.join(values.rename("x")).dropna(subset=["x"])
    n, events = len(df), int(df.event.sum())
    base = {"n": n, "events": events}
    if kind == "numerical":
        sd = df.x.std()
        df["x"] = (df.x - df.x.mean()) / sd
    if kind in ("numerical", "ordinal"):
        cols = ["time", "event", "x"] + ([strata] if strata else [])
        cph = fit_cox(df[cols], strata)
        if cph is None:
            return {**base, "note": "Cox did not converge"}
        s = cph.summary.loc["x"]
        risk = df.x * s["coef"]
        unit = "per 1 SD" if kind == "numerical" else "per step"
        return {**base, "effect": f"HR {s['exp(coef)']:.2f} [{s['exp(coef) lower 95%']:.2f}, "
                                  f"{s['exp(coef) upper 95%']:.2f}] {unit}",
                "HR": s["exp(coef)"], "p": s["p"],
                "c_index": concordance_index(df.time, -risk, df.event)}
    ref = df.x.value_counts().index[0]
    dummies = pd.get_dummies(df.x).drop(columns=ref).astype(float)
    names = {f"lvl_{i}": level for i, level in enumerate(dummies.columns)}
    dummies.columns = list(names)
    data = pd.concat([df[["time", "event"] + ([strata] if strata else [])], dummies], axis=1)
    cph = fit_cox(data, strata)
    if cph is None:
        return {**base, "note": "Cox did not converge"}
    lr = cph.log_likelihood_ratio_test()                # all levels vs the null model
    risk = cph.predict_log_partial_hazard(data)
    hrs = ", ".join(f"{names[c]} {cph.summary.loc[c, 'exp(coef)']:.2f}" for c in dummies.columns)
    return {**base, "effect": f"HR vs '{ref}': {hrs}", "p": lr.p_value,
            "c_index": concordance_index(df.time, -risk, df.event)}


def bh(p):
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum():
        q = p[ok]
        order = np.argsort(q)
        ranked = q[order] * ok.sum() / np.arange(1, ok.sum() + 1)
        ranked = np.minimum.accumulate(ranked[::-1])[::-1]
        res = np.empty_like(q)
        res[order] = np.minimum(ranked, 1)
        out[ok] = res
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=f"{ROOT}/results/clinical_survival_association")
    ap.add_argument("--min_patients", type=int, default=30, help="patients with the feature, per cohort")
    ap.add_argument("--min_events", type=int, default=10, help="events among them")
    ap.add_argument("--min_level", type=int, default=10, help="patients per categorical level (else 'other')")
    args = ap.parse_args()

    outcome = load_outcome()
    cases = outcome.index
    tcga, cptac = load_raw_tcga(cases), load_raw_cptac(cases)
    features = pd.concat([load_model_table(cases), harmonized(tcga, cptac), tcga, cptac], axis=1)
    groups = {study: outcome.study.eq(study) for study in outcome.study.unique()}
    groups.update({pool: outcome.study.isin(members) for pool, members in POOLS.items()})
    sizes = {g: (int(m.sum()), int(outcome.event[m].sum())) for g, m in groups.items()}
    for g, (n, ev) in sizes.items():
        print(f"{g}: {n} patients, {ev} events")

    rows, excluded = [], []
    for group, mask in groups.items():
        pooled = group in POOLS
        strata = "study" if pooled else None
        shared = {}   # model-table / harmonized values: raw columns that only repeat one are not tested again
        for name in features.columns:
            source = name.split(": ")[0]
            if pooled and source in ("TCGA", "CPTAC"):
                continue                      # raw columns differ between the cohorts
            if (source == "TCGA" and not group.startswith("TCGA_")) or (source == "CPTAC" and not group.startswith("CPTAC")):
                continue
            s = features.loc[mask, name]
            kind, values = feature_kind(name, s, args.min_level)
            if kind is None:
                if s.notna().any():
                    excluded.append({"cohort": group, "feature": name, "reason": values})
                continue
            if source in ("model", "harmonized"):
                shared[name] = values
            else:
                same = next((m for m, v in shared.items() if v.dropna().index.equals(values.dropna().index)
                             and (v.dropna().astype(str).values == values.dropna().astype(str).values).all()), None)
                if same:
                    excluded.append({"cohort": group, "feature": name, "reason": f"duplicate of {same}"})
                    continue
            present = values.notna()
            n, ev = int(present.sum()), int(outcome.event[mask][present].sum())
            if n < args.min_patients or ev < args.min_events:
                excluded.append({"cohort": group, "feature": name, "reason": f"too few ({n} patients, {ev} events)"})
                continue
            res = associate(values, kind, outcome[mask][["time", "event"] + (["study"] if pooled else [])], strata)
            rows.append({"cohort": group, "feature": name, "group": feature_group(name), "type": kind, **res})

    table = pd.DataFrame(rows)
    table["q"] = table.groupby("cohort")["p"].transform(bh)
    order = {g: i for i, g in enumerate(groups)}
    table = table.sort_values(["cohort", "p"], key=lambda c: c.map(order) if c.name == "cohort" else c)
    os.makedirs(args.out_dir, exist_ok=True)
    table.to_csv(os.path.join(args.out_dir, "associations.csv"), index=False)
    pd.DataFrame(excluded).to_csv(os.path.join(args.out_dir, "excluded_columns.csv"), index=False)
    with open(os.path.join(args.out_dir, "associations.md"), "w") as f:
        for cohort, t in table.groupby("cohort", sort=False):
            n, ev = sizes[cohort]
            f.write(f"\n### {cohort} ({n} patients, {ev} events)\n\n")
            show = t.assign(p=t.p.map(lambda v: f"{v:.2g}"), q=t.q.map(lambda v: f"{v:.2g}"),
                            c_index=t.c_index.map(lambda v: f"{v:.3f}"))
            f.write(show[["feature", "group", "type", "n", "events", "effect", "c_index", "p", "q"]]
                    .to_markdown(index=False) + "\n")

    # features x cohorts: HR per step / SD (ordinal, numerical) or the p of the levels (categorical)
    def cell(r):
        star = "*" if r.q < 0.05 else ""
        if r.type == "categorical":
            return f"p={r.p:.1g}{star} (C {r.c_index:.2f})"
        return f"{r.HR:.2f}{star} (C {r.c_index:.2f})"
    counts = table.groupby("feature").cohort.nunique()
    common = table[table.feature.isin(counts.index[counts >= 3])]
    wide = common.assign(cell=common.apply(cell, axis=1)).pivot(index="feature", columns="cohort", values="cell")
    wide = wide[[g for g in groups if g in wide.columns]]
    lead = [f for f in [f"model: {c}" for c in SELECTED] + ["harmonized: grade"] if f in wide.index]
    wide = wide.loc[lead + sorted(set(wide.index) - set(lead))].fillna("")
    wide.to_csv(os.path.join(args.out_dir, "summary_wide.csv"))
    with open(os.path.join(args.out_dir, "summary_wide.md"), "w") as f:
        f.write("HR per step (ordinal) / per SD (numerical), or p of the levels (categorical); * = q < 0.05 "
                "(BH within the cohort); C = Harrell's C-index of the feature alone.\n\n")
        f.write(wide.to_markdown() + "\n")
    print(open(os.path.join(args.out_dir, "summary_wide.md")).read())
    print(f"-> {args.out_dir}/associations.csv, associations.md, summary_wide.md/.csv, excluded_columns.csv")


if __name__ == "__main__":
    main()
