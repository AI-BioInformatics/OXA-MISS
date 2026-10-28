"""
Clean clinical tables (format of ccRCC/clinical_data_clean: study_id, case_id, age, sex, race, ethnicity, ajcc_stage,
ajcc_t, ajcc_n, ajcc_m) for the datasets added after the first 12: CPTAC LUAD / LSCC / PDA / STAD / UCEC, TCGA PAAD / UCEC.

Sources
    CPTAC LUAD / LSCC / PDA / UCEC  ccRCC/docs/CPTAC-3.clinical.tsv, with the recipe of CPTAC_CCRCC_clinical_clean.csv
                                    (age = age_at_diagnosis / 365 in years, not rounded; gender; AJCC pathologic;
                                    M from the clinical M when the pathologic M is MX / missing), checked by rebuilding
                                    CPTAC_CCRCC from the same file: identical except ajcc_m, 203/208 (the other 5 do
                                    not follow any rule on the docs columns, i.e. they came from another source).
    CPTAC STAD                      not in docs/CPTAC-3.clinical.tsv (older export) -> GDC API, same recipe; the API
                                    path is checked against docs on the CPTAC LUAD cases.
    TCGA PAAD / UCEC                not in docs/TCGA_clinical_data_COMPLETE.tsv; TCGA-CDR has them but without T / N / M
                                    and ethnicity -> GDC API with utils/fetch_gdc_clinical.py (recipe of the TCGA tables).
    TCGA UCEC has FIGO staging only. Endometrial FIGO 2009 <-> AJCC 7th correspondence:
        stage I-IV collapsed; T: I -> T1, II -> T2, IIIA / IIIB -> T3, IVA -> T4 (IIIC, IVB: T unknown);
        N: IIIC1 -> N1, IIIC2 -> N2 (otherwise unknown, as the OV FIGO conversion); M: I-III and IVA -> M0, IVB -> M1.
CPTAC cohorts = ccRCC/Gene_expression_CPTAC/cases_<T>.csv (GDC site + diagnosis U patients with WSI, from
scripts/make_gene_expression_manifests.py); TCGA = every case of the project.

Usage (a few API calls, runs on the login node):
    python utils/build_clinical_new_datasets.py --out_dir /work/H2020DeciderFicarra/ccRCC/clinical_data_clean_v2
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fetch_gdc_clinical import MISSING, clean_table, collapse, fetch, primary, to_table  # noqa: E402

DOCS_CPTAC = "/work/H2020DeciderFicarra/ccRCC/docs/CPTAC-3.clinical.tsv"
CLEAN_DIR = "/work/H2020DeciderFicarra/ccRCC/clinical_data"
COHORTS = "/work/H2020DeciderFicarra/ccRCC/Gene_expression_CPTAC/cases_{}.csv"
CPTAC_DOCS = ["LUAD", "LSCC", "PDA", "UCEC"]
CPTAC_API = ["STAD"]
TCGA = ["PAAD", "UCEC"]
RAW_COLS = ["case_id", "age", "sex", "race", "ethnicity", "ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]
DOCS_COLS = {"gender.demographic": "sex", "race.demographic": "race", "ethnicity.demographic": "ethnicity",
             "ajcc_pathologic_stage.diagnoses": "ajcc_stage", "ajcc_pathologic_t.diagnoses": "ajcc_t",
             "ajcc_pathologic_n.diagnoses": "ajcc_n", "ajcc_pathologic_m.diagnoses": "ajcc_m",
             "ajcc_clinical_m.diagnoses": "ajcc_m_clinical"}
M_MIN_AGREEMENT = 0.97   # CPTAC_CCRCC rebuild: ajcc_m 203/208


def raw_value(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip().lower() in MISSING else str(v)


def cptac_from_docs(docs, cases):
    """One row per case from the sample-level docs export (primary diagnosis row first)."""
    d = docs[docs["submitter_id"].isin(cases)].copy()
    d["_primary"] = d["diagnosis_is_primary_disease.diagnoses"].astype(str).str.lower().ne("false")
    conflicts = d.groupby("submitter_id")[list(DOCS_COLS) + ["age_at_diagnosis.diagnoses"]].nunique().gt(1).any(axis=1)
    if conflicts.any():
        print(f"  [!] {int(conflicts.sum())} cases with conflicting rows: the primary-diagnosis row is used")
    d = d.sort_values(["submitter_id", "_primary"], ascending=[True, False]).drop_duplicates("submitter_id")
    out = pd.DataFrame({"case_id": d["submitter_id"],
                        "age": (d["age_at_diagnosis.diagnoses"] / 365).map(lambda a: "" if np.isnan(a) else repr(float(a)))})
    for src, dst in DOCS_COLS.items():
        kind = {"ajcc_stage": "stage", "ajcc_t": "t", "ajcc_n": "n", "ajcc_m": "m", "ajcc_m_clinical": "m"}.get(dst)
        out[dst] = d[src].map(lambda v: collapse(v, kind) if kind else raw_value(v))
    return m_fallback(out).reset_index(drop=True)


def m_fallback(df):
    """ajcc_m from the clinical M when the pathologic M is MX / missing."""
    missing = df["ajcc_m"].str.lower().isin({"", "mx"}) & df["ajcc_m_clinical"].ne("")
    df.loc[missing, "ajcc_m"] = df.loc[missing, "ajcc_m_clinical"]
    print(f"  M from the clinical M for {int(missing.sum())} cases")
    return df


def cptac_from_api(hits, cases):
    rows = []
    for h in hits:
        if h["submitter_id"] not in cases:
            continue
        demo, diag = h.get("demographic") or {}, primary(h.get("diagnoses"))
        age = diag.get("age_at_diagnosis")
        rows.append({"case_id": h["submitter_id"], "age": "" if age is None else repr(age / 365),
                     "sex": raw_value(demo.get("gender") or demo.get("sex_at_birth")),
                     "race": raw_value(demo.get("race")), "ethnicity": raw_value(demo.get("ethnicity")),
                     "ajcc_stage": collapse(diag.get("ajcc_pathologic_stage"), "stage"),
                     "ajcc_t": collapse(diag.get("ajcc_pathologic_t"), "t"),
                     "ajcc_n": collapse(diag.get("ajcc_pathologic_n"), "n"),
                     "ajcc_m": collapse(diag.get("ajcc_pathologic_m"), "m"),
                     "ajcc_m_clinical": collapse(diag.get("ajcc_clinical_m"), "m")})
    return m_fallback(pd.DataFrame(rows, columns=RAW_COLS + ["ajcc_m_clinical"]))


def figo_endometrial(figo):
    """(stage, T, N, M) from an endometrial FIGO stage."""
    f = raw_value(figo).replace("Stage ", "")
    if not f:
        return "", "", "", ""
    stage = collapse(f"Stage {f}", "stage")
    t = {"I": "T1", "II": "T2", "IIIA": "T3", "IIIB": "T3", "IVA": "T4"}.get(
        "I" if f in ("I", "IA", "IB", "IC") else "II" if f in ("II", "IIA", "IIB") else f, "")
    n = {"IIIC1": "N1", "IIIC2": "N2"}.get(f, "")
    m = "M1" if f == "IVB" else "M0" if stage in ("Stage I", "Stage II", "Stage III") or f == "IVA" else ""
    return stage, t, n, m


def compare(a, b, label):
    """Agreement of two cleaned tables on their common cases."""
    a, b = a.set_index("case_id"), b.set_index("case_id")
    common = a.index.intersection(b.index)
    print(f"  check {label}: {len(common)} common cases")
    agreement = {}
    for c in RAW_COLS[1:]:
        x, y = a.loc[common, c], b.loc[common, c]
        same = (x == y) | (x.isna() & y.isna())
        if c == "age":
            same = same | np.isclose(pd.to_numeric(x), pd.to_numeric(y), atol=1e-9)
        print(f"    {c:10s} {same.mean():.1%} ({int(same.sum())}/{len(common)})")
        agreement[c] = same.mean()
    return agreement


def save(raw, study, out_dir):
    cleaned = clean_table(raw[RAW_COLS], study)
    cleaned.to_csv(os.path.join(out_dir, f"{study}_clinical_clean.csv"), index=False, na_rep="NaN")
    print(f"{study}: {len(cleaned)} cases; NaN: " + ", ".join(f"{c}={int(cleaned[c].isna().sum())}" for c in RAW_COLS[1:]))
    return cleaned


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    docs = pd.read_csv(DOCS_CPTAC, sep="\t", low_memory=False)

    # recipe check: CPTAC_CCRCC rebuilt from docs must match the existing table
    old = pd.read_csv(os.path.join(CLEAN_DIR, "CPTAC_CCRCC_clinical_clean.csv"))
    rebuilt = clean_table(cptac_from_docs(docs, set(old.case_id)), "CPTAC_CCRCC")
    agreement = compare(rebuilt, old, "CPTAC_CCRCC docs rebuild vs existing table")
    if len(rebuilt) != len(old) or agreement.pop("ajcc_m") < M_MIN_AGREEMENT or min(agreement.values()) < 1:
        sys.exit("CPTAC recipe does not reproduce CPTAC_CCRCC_clinical_clean.csv")

    new = []
    for t in CPTAC_DOCS:
        cases = set(pd.read_csv(COHORTS.format(t)).case_id)
        raw = cptac_from_docs(docs, cases)
        print(f"CPTAC_{t}: {len(raw)}/{len(cases)} cohort cases in docs ({len(cases) - len(raw)} without clinical data)")
        new.append(save(raw, f"CPTAC_{t}", args.out_dir))

    hits = fetch("CPTAC-3")
    luad_docs = new[CPTAC_DOCS.index("LUAD")]
    luad_api = clean_table(cptac_from_api(hits, set(luad_docs.case_id)), "CPTAC_LUAD")
    compare(luad_api, luad_docs, "CPTAC_LUAD GDC API vs docs")
    for t in CPTAC_API:
        cases = set(pd.read_csv(COHORTS.format(t)).case_id)
        raw = cptac_from_api(hits, cases)
        print(f"CPTAC_{t}: {len(raw)}/{len(cases)} cohort cases on GDC ({len(cases) - len(raw)} without clinical data)")
        new.append(save(raw, f"CPTAC_{t}", args.out_dir))

    for t in TCGA:
        raw = to_table(fetch(f"TCGA-{t}"))
        if t == "UCEC":
            no_ajcc = raw[["ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]].eq("").all(axis=1)
            raw.loc[no_ajcc, ["ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]] = list(raw.loc[no_ajcc, "figo_stage"].map(figo_endometrial))
            print(f"TCGA_UCEC: stage / T / N / M from endometrial FIGO for {int(no_ajcc.sum())} cases")
        new.append(save(raw, f"TCGA_{t}", args.out_dir))

    # combined table: existing 12 studies (unchanged) + the new ones
    allold = pd.read_csv(os.path.join(CLEAN_DIR, "all_studies_clinical_clean.csv"))
    allnew = pd.concat([allold] + new, ignore_index=True)
    dup = allnew.case_id.duplicated(keep=False)
    if dup.any():
        sys.exit(f"case_id in more than one study: {sorted(allnew.case_id[dup].unique())[:10]}")
    allnew.to_csv(os.path.join(args.out_dir, "all_studies_clinical_clean.csv"), index=False, na_rep="NaN")
    print(f"all_studies_clinical_clean.csv: {len(allold)} existing + {len(allnew) - len(allold)} new = {len(allnew)} cases")


if __name__ == "__main__":
    main()
