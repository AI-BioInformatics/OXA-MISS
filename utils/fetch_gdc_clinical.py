"""
Clinical table of a TCGA project from the GDC API, in the format of ccRCC/clinical_data/<STUDY>_clinical.csv
(tab-separated: case_id, age, sex, race, ethnicity, ajcc_stage, ajcc_t, ajcc_n, ajcc_m), then cleaned like
new_files/clean_clinical_data.py (lower case, empty / NX / MX / TX / T0 / Tis / Stage X -> NaN).

Conventions of the existing tables:
    age          demographic.age_at_index (years); fallback diagnoses.age_at_diagnosis / 365.25
    sex          demographic.sex_at_birth (gender in older GDC releases)
    stage / TNM  AJCC pathologic of the primary diagnosis, sub-stages collapsed (Stage IIIA -> Stage III,
                 T4a -> T4, N2b -> N2, M1a -> M1)
Options (study-specific, off by default):
    --m_fallback_clinical  ajcc_m from ajcc_clinical_m when the pathologic M is missing (e.g. HNSC: pathologic M
                           missing for 63% of the patients)
    --figo_to_tnm          stage / T / N / M from the FIGO stage when there is no AJCC staging (OV: FIGO only).
                           AJCC 8th / FIGO 2014 ovarian correspondence: FIGO I / II / III -> T1 / T2 / T3
                           (FIGO IV: T unknown); FIGO IV -> M1, FIGO I-III -> M0; N unknown (only FIGO IIIA1 implies
                           N1, and TCGA uses the older IIIA / IIIB / IIIC); stage I-IV collapsed as for AJCC.
Patients: --cases (a file with a case_id column, e.g. the existing raw table) or every case of the project.

Usage (a few API calls, runs on the login node):
    python utils/fetch_gdc_clinical.py --project TCGA-HNSC --cases /work/.../clinical_data/TCGA_HNSC_clinical.csv \
        --out_dir /work/H2020DeciderFicarra/ccRCC/clinical_data_clean_v2
"""
import argparse
import json
import os
import re
import urllib.request

import numpy as np
import pandas as pd

GDC_CASES = "https://api.gdc.cancer.gov/cases"
FIELDS = ["submitter_id", "demographic.sex_at_birth", "demographic.gender", "demographic.race", "demographic.ethnicity", "demographic.age_at_index",
          "diagnoses.age_at_diagnosis", "diagnoses.ajcc_pathologic_stage", "diagnoses.ajcc_pathologic_t",
          "diagnoses.ajcc_pathologic_n", "diagnoses.ajcc_pathologic_m", "diagnoses.ajcc_clinical_m",
          "diagnoses.classification_of_tumor", "diagnoses.diagnosis_is_primary_disease", "diagnoses.figo_stage"]
UNKNOWN = {"nx", "mx", "tx", "t0", "tis", "stage x"}
MISSING = {"", "nan", "none", "na", "n/a", "not reported", "unknown"}


def fetch(project):
    payload = {"filters": {"op": "in", "content": {"field": "project.project_id", "value": [project]}},
               "fields": ",".join(FIELDS), "format": "JSON", "size": 5000}
    req = urllib.request.Request(GDC_CASES, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)["data"]["hits"]


def collapse(value, kind):
    """GDC AJCC value -> the collapsed form of the existing tables."""
    if not isinstance(value, str) or value.strip().lower() in MISSING:
        return ""
    v = value.strip()
    pattern = {"stage": r"^(Stage (?:IV|III|II|I))", "t": r"^(T\d)", "n": r"^(N\d)", "m": r"^(M\d)"}[kind]
    match = re.match(pattern, v)
    return match.group(1) if match else v      # TX, NX, MX, Tis, T0, Stage X unchanged (NaN after cleaning)


def primary(diagnoses):
    for d in diagnoses or []:
        if str(d.get("classification_of_tumor", "")).lower() == "primary" or d.get("diagnosis_is_primary_disease") is True:
            return d
    return (diagnoses or [{}])[0]


FIGO_T = {"Stage I": "T1", "Stage II": "T2", "Stage III": "T3"}


def figo_to_tnm(figo):
    """(stage, T, N, M) from a FIGO stage (AJCC 8th / FIGO 2014 ovarian correspondence)."""
    stage = collapse(figo, "stage")
    if stage not in ("Stage I", "Stage II", "Stage III", "Stage IV"):
        return "", "", "", ""
    if stage == "Stage IV":
        return stage, "", "", "M1"
    return stage, FIGO_T[stage], "", "M0"


def to_table(hits, m_fallback_clinical=False, figo=False):
    rows = []
    for h in hits:
        demo, diag = h.get("demographic") or {}, primary(h.get("diagnoses"))
        age = demo.get("age_at_index")
        if age is None and diag.get("age_at_diagnosis") is not None:
            age = round(diag["age_at_diagnosis"] / 365.25)
        clean = lambda v: "" if v is None or str(v).strip().lower() in MISSING else str(v)
        rows.append({"case_id": h["submitter_id"], "age": "" if age is None else str(int(age)),
                     "sex": clean(demo.get("sex_at_birth") or demo.get("gender")),   # renamed in recent GDC releases
                     "race": clean(demo.get("race")),
                     "ethnicity": clean(demo.get("ethnicity")),
                     "ajcc_stage": collapse(diag.get("ajcc_pathologic_stage"), "stage"),
                     "ajcc_t": collapse(diag.get("ajcc_pathologic_t"), "t"),
                     "ajcc_n": collapse(diag.get("ajcc_pathologic_n"), "n"),
                     "ajcc_m": collapse(diag.get("ajcc_pathologic_m"), "m"),   # pathologic only, as the other tables
                     "ajcc_m_clinical": collapse(diag.get("ajcc_clinical_m"), "m"),
                     "figo_stage": clean(diag.get("figo_stage"))})
    df = pd.DataFrame(rows)
    if m_fallback_clinical:
        # a missing / MX pathologic M is taken from the clinical M
        missing = df["ajcc_m"].str.lower().isin({"", "mx"})
        df.loc[missing, "ajcc_m"] = df.loc[missing, "ajcc_m_clinical"]
        print(f"M from the clinical M for {int((missing & df['ajcc_m'].ne('')).sum())} of {int(missing.sum())} cases without pathologic M")
    if figo:
        no_ajcc = df[["ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]].eq("").all(axis=1)
        derived = df.loc[no_ajcc, "figo_stage"].map(figo_to_tnm)
        df.loc[no_ajcc, ["ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]] = list(derived)
        print(f"stage / T / M derived from FIGO for {int(no_ajcc.sum())} cases without AJCC staging")
    return df


def clean_table(raw, study):
    """Same cleaning as new_files/clean_clinical_data.py."""
    df = raw.copy()
    df.insert(0, "study_id", study)
    for c in df.columns.drop(["study_id", "case_id"]):
        s = df[c].astype(str).str.strip().str.lower()
        df[c] = s.where(~s.isin(MISSING | UNKNOWN), np.nan)
    df["age"] = pd.to_numeric(df["age"], errors="coerce")
    return df


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project", required=True, help="e.g. TCGA-HNSC")
    ap.add_argument("--cases", default=None, help="table with the case_id column to keep (tab or comma separated)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--compare", default=None, help="existing raw table to compare with (validation)")
    ap.add_argument("--m_fallback_clinical", action="store_true", help="M from the clinical M when the pathologic is missing")
    ap.add_argument("--figo_to_tnm", action="store_true", help="stage / T / N / M from FIGO when there is no AJCC staging")
    args = ap.parse_args()
    study = args.project.replace("-", "_")
    table = to_table(fetch(args.project), args.m_fallback_clinical, args.figo_to_tnm)
    if args.cases:
        cases = pd.read_csv(args.cases, sep=None, engine="python", dtype=str)["case_id"].str.strip()
        missing = sorted(set(cases) - set(table["case_id"]))
        if missing:
            print(f"[!] {len(missing)} cases not found on GDC, e.g. {missing[:5]}")
        table = table.set_index("case_id").reindex(cases).fillna("").rename_axis("case_id").reset_index()
    if args.compare:
        old = pd.read_csv(args.compare, sep="\t", dtype=str, keep_default_na=False).set_index("case_id")
        new = table.set_index("case_id").reindex(old.index).fillna("")
        print(f"agreement with {args.compare} ({len(old)} cases):")
        for c in old.columns:
            both = (old[c] != "") | (new[c] != "")
            print(f"  {c:10s} {np.mean(old[c] == new[c]):.1%} (on values present in either: {np.mean(old.loc[both, c] == new.loc[both, c]):.1%})")
    os.makedirs(args.out_dir, exist_ok=True)
    raw_cols = ["case_id", "age", "sex", "race", "ethnicity", "ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]
    table[raw_cols].to_csv(os.path.join(args.out_dir, f"{study}_clinical.csv"), sep="\t", index=False)
    cleaned = clean_table(table[raw_cols], study)
    cleaned.to_csv(os.path.join(args.out_dir, f"{study}_clinical_clean.csv"), index=False, na_rep="NaN")
    print(f"{study}: {len(cleaned)} cases -> {args.out_dir}; NaN per column: "
          + ", ".join(f"{c}={int(cleaned[c].isna().sum())}" for c in cleaned.columns if c not in ("study_id", "case_id")))
    if table["figo_stage"].ne("").any():
        print(f"FIGO stage available for {int(table['figo_stage'].ne('').sum())} cases: {table['figo_stage'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
