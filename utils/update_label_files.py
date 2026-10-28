"""
Build the OS label files of every cohort (dataloader/dataset_updated/<DATASET>_labels.csv) from scratch.

A patient is in the label file if it has a valid OS (event known, follow-up > 0 days, possibly from OS_OVERRIDES)
and at least one available modality of the dataset yaml (config/<DATASET>_dataset.yaml):
    WSI       UNIv2 feature files (.h5 / .pt) of the yaml's pt_files_path: diagnostic FFPE slides only
              (TCGA DX slides at 20x equivalent, CPTAC FFPE tumour slides)
    Genomics  rows of the yaml's genomics_path
    CT / MRI  token features of config/radiology.yaml (default encoders suprem / mrseg)
    Clinical  rows of the clean clinical table with at least one observed variable
Rows: one per WSI of the patient, or a single row with an empty slide_id when the patient has no WSI.

OS sources (ccRCC/docs):
    TCGA   TCGA-CDR Supplemental Table S1 (OS, OS.time) or cBioPortal PanCanAtlas (Overall Survival months *
           30.4375), per cohort the source that reproduces the earlier label files (TCGA_COHORTS); PAAD, UCEC: CDR
    CPTAC  CPTAC-3.clinical.tsv (STAD: the same GDC fields, CPTAC_CLINICAL_FILES): days_to_death if dead, else the
           max days_to_last_follow_up; patients restricted to the cohort's primary site / disease type
           (CPTAC_COHORTS); CCRCC also to ccRCC histology
Patients without a valid OS in the source keep the OS of the earlier label files (ccRCC/docs/OS_previous_label_files_*.csv,
curated from the original label files); OS_OVERRIDES always apply. The OS provenance is in the audit (os_source), and
the agreement with the earlier label files is printed per cohort.

Label file convention: Survival = 1 -> Dead (event); `event_name: "Survival"` in the dataset yaml.

Usage (reads the genomics and feature folders of every cohort: run it on a compute node):
    python utils/update_label_files.py [--out dataloader/dataset_updated] [--datasets TCGA_PAAD CPTAC_UCEC]
"""
import argparse
import glob
import os
import re

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS = '/work/H2020DeciderFicarra/ccRCC/docs'
CPTAC3_CLINICAL = f'{DOCS}/CPTAC-3.clinical.tsv'
CBIO_CLINICAL = f'{DOCS}/TCGA_clinical_data_COMPLETE.tsv'
TCGA_CDR = f'{DOCS}/TCGA-CDR-SupplementalTableS1.xlsx'
# OS of the label files before the 2026-10-26 rebuild (curated from the original label files + OS_OVERRIDES)
PREVIOUS_OS = f'{DOCS}/OS_previous_label_files_2026-10-26.csv'
CLINICAL_TABLE = '/work/H2020DeciderFicarra/ccRCC/clinical_data/all_studies_clinical_clean.csv'
CLINICAL_VARS = ['age', 'sex', 'race', 'ethnicity', 'ajcc_stage', 'ajcc_t', 'ajcc_n', 'ajcc_m']
RADIOLOGY_REGISTRY = os.path.join(REPO, 'config', 'radiology.yaml')
RADIOLOGY_ENCODERS = {'CT': 'suprem', 'MRI': 'mrseg'}
CASE_ID_PATTERN = re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|C3[LN]-\d{5})')

# TCGA cohort -> OS source reproducing the earlier label files
TCGA_COHORTS = {
    'BLCA': 'cdr', 'BRCA': 'cdr', 'COAD': 'cdr', 'HNSC': 'cdr', 'OV': 'cdr', 'STAD': 'cdr', 'PAAD': 'cdr', 'UCEC': 'cdr',
    'KIRC': 'cbio', 'KIRP': 'cbio', 'LIHC': 'cbio', 'LUAD': 'cbio', 'LUSC': 'cbio',
}
# CPTAC cohort -> (primary_site, disease_type) of CPTAC-3.clinical.tsv; None = any disease type of the site.
CPTAC_COHORTS = {
    'CCRCC': ('Kidney', None),
    'LSCC': ('Bronchus and lung', 'Squamous Cell Neoplasms'),
    'LUAD': ('Bronchus and lung', 'Adenomas and Adenocarcinomas'),
    'PDA': ('Pancreas', None),
    'UCEC': ('Uterus, NOS', None),
    'STAD': ('Stomach', None),
}
# CPTAC cohort -> clinical file, when not CPTAC3_CLINICAL. STAD is not in CPTAC-3.clinical.tsv: same GDC fields
# (CPTAC-3, primary site stomach, 165 cases) downloaded from the GDC API on 2026-10-26, same columns.
CPTAC_CLINICAL_FILES = {'STAD': f'{DOCS}/CPTAC-3_STAD.clinical_GDC_2026-10-26.tsv'}
# CPTAC-3 kidney cohort also contains non-ccRCC tumours: keep ccRCC (8312/3) and unknown histology
CCRCC_MORPHOLOGY_KEEP = {'8312/3', 'Unknown'}

LABEL_COLS = ['case_id', 'slide_id', 'True_Label', 'FUT', 'Survival']

# Patients whose follow-up in the sources is <= 0 days but a usable time was found in another source
# (search of 2026-09-30 over local tables, GDC API release 46, cBioPortal API, PDC API, LinkedOmics and the
#  CPTAC ccRCC papers: dataloader/dataset_updated/audit/patients_without_valid_OS.csv). case_id -> (event, days, source)
# Rule agreed with the user: alive patients are censored at the latest date they were known alive;
# dead patients without a death date stay dead, with the latest date they were known alive as FUT.
OS_OVERRIDES = {
    # alive: censored at the latest date known alive
    'TCGA-BH-A0B2': (0, 1242, 'cBioPortal brca_tcga_pan_can_atlas_2018: LIVING, 40.81 months (label said Dead at 0, '
                              'not in TCGA-CDR, no clinical data in GDC r46)'),
    'TCGA-BW-A5NP': (0, 289, 'GDC r46: Alive, treatment end at 289 d (last follow-up 102 d)'),
    'TCGA-BR-A4CR': (0, 51, 'cBioPortal stad_tcga_pub: LIVING, 1.68 months'),
    'TCGA-BR-4369': (0, 20, 'GDC r46: Alive, days_to_consent 20 (all follow-ups 0)'),
    'TCGA-BR-8679': (0, 5, 'cBioPortal stad_tcga_pub: LIVING, 0.16 months'),
    # dead, no death date: latest date known alive
    'TCGA-24-0968': (1, 598, 'GDC r46 + BCR: Dead, last contact 598 d (ov_tcga_pub: DECEASED 19.61 months)'),
    'TCGA-63-A5MU': (1, 420, 'GDC r46: Dead (no death date), progression at 420 d; PanCanAtlas PFS 420 d'),
    'TCGA-3M-AB47': (1, 395, 'GDC r46: Dead (no death date, cause: stomach cancer), recurrence at 395 d'),
    'TCGA-IN-A6RO': (1, 200, 'GDC r46 + BCR: Dead (death date [Discrepancy] in BCR), treatment end at 200 d'),
    'TCGA-GV-A3QG': (1, 84, 'GDC r46: Dead (death date [Discrepancy] in BCR), progression at 84 d; PanCanAtlas PFS 84 d '
                            '(blca_tcga_pub OS 0.36 months contradicts the progression)'),
    'TCGA-6A-AB49': (1, 31, 'GDC r46: Dead (no death date), days_to_consent 31'),
    'TCGA-56-6546': (1, 26, 'GDC r46: Dead (no death date), days_to_consent 26'),
    'C3N-02723': (1, 2, 'Li et al. Cancer Cell 2023 Table S1: dead, OS 2 d (GDC r46: days_to_death -6; '
                        'cause of death: surgical complications)'),
}


def read_ids(path):
    return set(pd.read_csv(path, usecols=[0]).iloc[:, 0].astype(str))


def case_of(name):
    m = CASE_ID_PATTERN.match(name)
    return m.group(1) if m else None


# ------------------------------------------------------------------ modalities
def wsi_on_disk(folder):
    """slide_id -> case_id of every WSI feature file (.npy / .h5 / .pt) of the folder."""
    slides = {}
    if os.path.isdir(folder):
        for f in os.listdir(folder):
            if f.endswith(('.npy', '.h5', '.pt')):
                stem = os.path.splitext(f)[0]
                if case_of(stem):
                    slides[stem] = case_of(stem)
    return pd.Series(slides, dtype=str)


def radiology_on_disk(dataset_name):
    """modality -> case_ids with a token file of the default encoder (config/radiology.yaml)."""
    reg = yaml.safe_load(open(RADIOLOGY_REGISTRY))
    tumor = reg['tumor_folders'].get(dataset_name)
    out = {}
    for m, enc in RADIOLOGY_ENCODERS.items():
        folder = os.path.join(reg['root'], reg['encoders'][m][enc], tumor) if tumor else None
        ids = set()
        if folder and os.path.isdir(folder):
            ids = {case_of(f) for f in os.listdir(folder) if f.endswith('.pt') and case_of(f)}
        out[m] = ids
    return out


def clinical_on_disk(table, dataset_name):
    t = table[table.study_id == dataset_name]
    return set(t.loc[t[CLINICAL_VARS].notna().any(axis=1), 'case_id'].astype(str))


# ------------------------------------------------------------------ OS sources
def cptac_os(cohort):
    site, disease = CPTAC_COHORTS[cohort]
    c = pd.read_csv(CPTAC_CLINICAL_FILES.get(cohort, CPTAC3_CLINICAL), sep='\t', low_memory=False)
    c = c[c.primary_site == site]
    if disease is not None:
        c = c[c.disease_type == disease]
    g = c.groupby('submitter_id').agg(
        vital=('vital_status.demographic', lambda s: set(s.dropna())),
        dtd=('days_to_death.demographic', lambda s: pd.to_numeric(s, errors='coerce').max()),
        dlf=('days_to_last_follow_up.diagnoses', lambda s: pd.to_numeric(s, errors='coerce').max()),
        morphology=('morphology.diagnoses', 'first'))
    g['event'] = g.vital.map(lambda v: 1.0 if 'Dead' in v else (0.0 if 'Alive' in v else np.nan))
    g['time'] = np.where(g.event == 1, g.dtd, g.dlf)
    keep = g.morphology.isin(CCRCC_MORPHOLOGY_KEEP) if cohort == 'CCRCC' else pd.Series(True, index=g.index)
    return g[['time', 'event']], keep


def tcga_os(cohort, source, cdr, cbio):
    if source == 'cdr':
        d = cdr[(cdr.type == cohort) & cdr.Redaction.isna()].drop_duplicates('bcr_patient_barcode')
        d = d.set_index('bcr_patient_barcode')
        return pd.DataFrame({'time': pd.to_numeric(d['OS.time'], errors='coerce'),
                             'event': pd.to_numeric(d['OS'], errors='coerce')})
    c = cbio[cbio['Study ID'].str.startswith(cohort.lower() + '_tcga')].drop_duplicates('Patient ID').set_index('Patient ID')
    return pd.DataFrame({'time': (c['Overall Survival (Months)'] * 30.4375).round(1),
                         'event': c['Overall Survival Status'].astype(str).str[0].map({'1': 1.0, '0': 0.0})})


def apply_overrides(os_table):
    os_table = os_table.copy()
    used = sorted(set(OS_OVERRIDES) & set(os_table.index))
    for pid in used:
        event, days, _ = OS_OVERRIDES[pid]
        os_table.loc[pid, ['event', 'time']] = [float(event), float(days)]
    missing = {p: v for p, v in OS_OVERRIDES.items() if p not in os_table.index}
    for pid, (event, days, _) in missing.items():
        os_table.loc[pid, ['event', 'time']] = [float(event), float(days)]
    return os_table


# ------------------------------------------------------------------ per dataset
def previous_os(previous_table, name):
    """OS of the earlier label files of the dataset (curated from the original label files), None if absent."""
    if not previous_table or not os.path.exists(previous_table):
        return None
    old = pd.read_csv(previous_table, dtype=str)
    old = old[old.dataset == name].drop_duplicates('case_id').set_index('case_id')
    if old.empty:
        return None
    return pd.DataFrame({'time': pd.to_numeric(old.FUT, errors='coerce'), 'event': pd.to_numeric(old.Survival, errors='coerce')})


def build(name, os_table, slides, modalities, out_dir, previous_dir, keep=None):
    # OS: the cohort's source, then the earlier label files (PREVIOUS_OS) for the patients the source has no valid OS for,
    # then OS_OVERRIDES (always). os_source records where each patient's OS comes from.
    os_table = os_table.copy()
    os_table['os_source'] = 'source'
    valid = lambda t: t.time.notna() & t.event.notna() & (t.time > 0)
    prev = previous_os(previous_dir, name)
    if prev is not None:
        fill = prev.index[valid(prev)].difference(os_table.index[valid(os_table)])
        os_table = os_table.reindex(os_table.index.union(fill))
        os_table.loc[fill, ['time', 'event']] = prev.loc[fill, ['time', 'event']].values
        os_table.loc[fill, 'os_source'] = 'previous label file'
    os_table = apply_overrides(os_table)
    os_table.loc[os_table.index.isin(list(OS_OVERRIDES)), 'os_source'] = 'override'
    universe = sorted(set(slides.values) | set().union(*modalities.values()))
    audit = pd.DataFrame(index=pd.Index(universe, name='case_id'))
    audit['WSI'] = audit.index.isin(set(slides.values))
    for m, ids in modalities.items():
        audit[m] = audit.index.isin(ids)
    mod_cols = ['WSI'] + list(modalities)
    audit['n_modalities'] = audit[mod_cols].sum(axis=1)
    audit = audit.join(os_table)
    audit['has_OS'] = audit.time.notna() & audit.event.notna() & (audit.time > 0)
    status = pd.Series('included', index=audit.index)
    status[~audit.has_OS] = 'no valid OS'
    if keep is not None:
        histology_ok = keep.reindex(audit.index).astype(object).where(lambda x: x.notna(), True).astype(bool)
        status[(status == 'included') & ~histology_ok] = 'excluded histology'
    audit['status'] = status
    included = audit.index[status == 'included']

    rows = []
    for pid in included:
        r = audit.loc[pid]
        event = int(r.event)
        base = {'case_id': pid, 'True_Label': 'Dead' if event == 1 else 'Alive',
                'FUT': f'{float(r.time):.1f}', 'Survival': str(event)}
        for s in sorted(slides[slides == pid].index) or ['']:
            rows.append(dict(base, slide_id=s))
    labels = pd.DataFrame(rows, columns=LABEL_COLS)
    label_file = f'{name}_labels.csv'
    labels.to_csv(os.path.join(out_dir, label_file), sep='\t', index=False)
    audit.to_csv(os.path.join(out_dir, 'audit', f'{name}_audit.csv'))

    inc = audit.loc[included]
    print(f'\n### {name}: {len(included)} patients ({int(inc.event.sum())} deaths), {len(labels)} rows, '
          f'{int((labels.slide_id != "").sum())} WSI')
    print('  patients with modality: ' + ', '.join(f'{m} {int(inc[m].sum())}' for m in mod_cols))
    print(f'  not included: {status[status != "included"].value_counts().to_dict()} | OS from: '
          f'{inc.os_source.value_counts().to_dict()}')
    old = previous_os(previous_dir, name)
    if old is not None:
        common = old.index.intersection(inc.index)
        t_ok = int(((old.loc[common, 'time'] - inc.loc[common, 'time']).abs() <= 1).sum())
        e_ok = int((old.loc[common, 'event'] == inc.loc[common, 'event']).sum())
        print(f'  vs previous label file: {len(old)} -> {len(inc)} patients (kept {len(common)}, dropped '
              f'{len(old.index.difference(inc.index))}, new {len(inc.index.difference(old.index))}); '
              f'OS equal on kept: time {t_ok}/{len(common)}, event {e_ok}/{len(common)}')
    return audit


def dataset_params(name):
    return yaml.safe_load(open(os.path.join(REPO, 'config', f'{name}_dataset.yaml')))['parameters']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'dataloader', 'dataset_updated'))
    ap.add_argument('--previous', default=PREVIOUS_OS,
                    help='OS of the earlier label files (dataset, case_id, FUT, Survival): fallback OS and agreement report')
    ap.add_argument('--datasets', nargs='+', default=None, help='default: every TCGA and CPTAC cohort with an OS source')
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'audit'), exist_ok=True)
    names = args.datasets or [f'TCGA_{c}' for c in TCGA_COHORTS] + [f'CPTAC_{c}' for c in CPTAC_COHORTS]
    clinical = pd.read_csv(CLINICAL_TABLE, low_memory=False)
    cdr = cbio = None
    for name in names:
        p = dataset_params(name)
        mods = {'Genomics': read_ids(p['genomics_path']), **radiology_on_disk(name),
                'Clinical': clinical_on_disk(clinical, name)}
        slides = wsi_on_disk(p['pt_files_path'])
        if name.startswith('TCGA_'):
            cohort = name[len('TCGA_'):]
            if cdr is None:
                cdr = pd.read_excel(TCGA_CDR, sheet_name=0)
                cbio = pd.read_csv(CBIO_CLINICAL, sep='\t', low_memory=False)
            build(name, tcga_os(cohort, TCGA_COHORTS[cohort], cdr, cbio), slides, mods, args.out, args.previous)
        else:
            os_table, keep = cptac_os(name[len('CPTAC_'):])
            build(name, os_table, slides, mods, args.out, args.previous, keep=keep)


if __name__ == '__main__':
    main()
