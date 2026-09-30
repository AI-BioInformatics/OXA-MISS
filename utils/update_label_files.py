"""
Build updated OS label files that include every patient with an OS label and at least
one available modality (WSI, genomics, CT, MRI, clinical), not only patients with a WSI.

Existing rows of the label files are kept unchanged, except: CPTAC patients with a known non-ccRCC
histology are removed; patients with follow-up <= 0 days get the OS found in another source
(OS_OVERRIDES) or are removed. New patients are appended with:
  - one row per WSI found on disk (union of all feature extractors), or
  - a single row with an empty slide_id when the patient has no WSI.

OS for new patients is taken from the same source that reproduces the existing labels
of each cohort (checked and printed at run time):
  - CPTAC:            CPTAC-3.clinical.tsv (days_to_death if dead, else max days_to_last_follow_up)
  - TCGA, CDR cohorts: TCGA-CDR Supplemental Table S1 (OS, OS.time)
  - TCGA, cBio cohorts: cBioPortal PanCancer clinical (Overall Survival months * 30.4375)

Label file convention (same as dataloader/dataset/): Survival = 1 -> Dead (event).
Use `event_name: "Survival"` in the dataset yaml.

Usage:
    python utils/update_label_files.py [--out dataloader/dataset_updated]
"""
import argparse
import glob
import os

import numpy as np
import pandas as pd
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CCRCC = '/work/H2020DeciderFicarra/ccRCC'
TCGA_SHARED = '/work/h2020deciderficarra_shared/TCGA'
LABELS_DIR = os.path.join(REPO, 'dataloader', 'dataset')

CT_MAPPING = f'{CCRCC}/CT_mapping.csv'
MRI_MAPPING = f'{CCRCC}/MRI_mapping.csv'
CPTAC3_CLINICAL = f'{CCRCC}/CPTAC-3.clinical.tsv'
CBIO_CLINICAL = f'{CCRCC}/TCGA_clinical_data_COMPLETE.tsv'
TCGA_CDR = f'{CCRCC}/TCGA-CDR-SupplementalTableS1.xlsx'

# cohort -> OS source reproducing the existing label file
TCGA_COHORTS = {
    'BLCA': 'cdr', 'BRCA': 'cdr', 'COAD': 'cdr', 'HNSC': 'cdr', 'OV': 'cdr', 'STAD': 'cdr',
    'KIRC': 'cbio', 'KIRP': 'cbio', 'LIHC': 'cbio', 'LUAD': 'cbio', 'LUSC': 'cbio',
}
# CPTAC-3 kidney cohort also contains non-ccRCC tumours: keep ccRCC (8312/3) and unknown histology
CPTAC_MORPHOLOGY_KEEP = {'8312/3', 'Unknown'}

LABEL_COLS = ['case_id', 'slide_id', 'True_Label', 'FUT', 'Survival']

# Patients whose follow-up in the label files is <= 0 days but a usable time was found in another source
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


def wsi_on_disk(pt_dirs, case_id_len):
    """slide_id -> case_id for every .pt file in the given folders."""
    slides = {}
    for d in pt_dirs:
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            if f.endswith('.pt'):
                slides[f[:-3]] = f[:case_id_len]
    return pd.Series(slides, dtype=str)


def imaging_on_disk(mapping, folder):
    if folder is None or not os.path.isdir(folder):
        return set()
    return {r.case_id for r in mapping.itertuples() if os.path.exists(os.path.join(folder, r.chosen_exam))}


def yaml_params(pattern):
    """Merge modality paths of all dataset yamls matching pattern (e.g. several extractors)."""
    params = {}
    for f in sorted(glob.glob(os.path.join(REPO, 'config', pattern))):
        p = yaml.safe_load(open(f))['parameters']
        for k in ['pt_files_path', 'genomics_path', 'cnv_path', 'ct_path', 'mri_path', 'clinical_path']:
            if k in p:
                params.setdefault(k, set()).add(p[k])
    return params


# ------------------------------------------------------------------ OS sources
def cptac_os():
    c = pd.read_csv(CPTAC3_CLINICAL, sep='\t', low_memory=False)
    c = c[c.primary_site == 'Kidney']
    g = c.groupby('submitter_id').agg(
        vital=('vital_status.demographic', lambda s: set(s.dropna())),
        dtd=('days_to_death.demographic', 'max'),
        dlf=('days_to_last_follow_up.diagnoses', 'max'),
        morphology=('morphology.diagnoses', 'first'),
        diagnosis=('primary_diagnosis.diagnoses', 'first'))
    g['event'] = g.vital.map(lambda v: 1.0 if 'Dead' in v else (0.0 if 'Alive' in v else np.nan))
    g['time'] = np.where(g.event == 1, g.dtd, g.dlf)
    return g[['time', 'event', 'morphology', 'diagnosis']]


def tcga_os(cohort, source, cdr, cbio):
    if source == 'cdr':
        d = cdr[(cdr.type == cohort) & cdr.Redaction.isna()].drop_duplicates('bcr_patient_barcode')
        d = d.set_index('bcr_patient_barcode')
        return pd.DataFrame({'time': pd.to_numeric(d['OS.time'], errors='coerce'),
                             'event': pd.to_numeric(d['OS'], errors='coerce')})
    c = cbio[cbio['Study ID'].str.startswith(cohort.lower() + '_tcga')].drop_duplicates('Patient ID').set_index('Patient ID')
    return pd.DataFrame({'time': (c['Overall Survival (Months)'] * 30.4375).round(1),
                         'event': c['Overall Survival Status'].astype(str).str[0].map({'1': 1.0, '0': 0.0})})


# ------------------------------------------------------------------ per dataset
def build(name, label_file, os_table, slides, modalities, out_dir, keep_mask=None):
    labels = pd.read_csv(os.path.join(LABELS_DIR, label_file), sep='\t', dtype=str)
    removed = []
    if keep_mask is not None:
        # existing patients with a histology different from the cohort's one are removed too
        # (patients not found in the histology source are kept)
        drop = keep_mask.reindex(labels.case_id.unique())
        removed = sorted(drop[drop == False].index)
        labels = labels[~labels.case_id.isin(removed)]
    # follow-up <= 0 days is not a valid survival time: use the OS found elsewhere, otherwise remove the patient
    overridden = sorted(set(labels.case_id) & set(OS_OVERRIDES))
    for pid in overridden:
        event, days, _ = OS_OVERRIDES[pid]
        rows = labels.case_id == pid
        labels.loc[rows, 'FUT'] = f'{float(days):.1f}'
        labels.loc[rows, 'Survival'] = str(event)
        labels.loc[rows, 'True_Label'] = 'Dead' if event == 1 else 'Alive'
    non_positive = sorted(labels.loc[pd.to_numeric(labels.FUT, errors='coerce') <= 0, 'case_id'].unique())
    labels = labels[~labels.case_id.isin(non_positive)]
    labeled = set(labels.case_id)

    universe = sorted(set(slides.values) | set().union(*modalities.values()) | labeled)
    audit = pd.DataFrame(index=pd.Index(universe, name='case_id'))
    audit['in_labels'] = audit.index.isin(labeled)
    audit['WSI'] = audit.index.isin(set(slides.values))
    for m, ids in modalities.items():
        audit[m] = audit.index.isin(ids)
    mod_cols = ['WSI'] + list(modalities)
    audit['n_modalities'] = audit[mod_cols].sum(axis=1)
    audit = audit.join(os_table)
    audit['has_OS'] = audit.time.notna() & audit.event.notna() & (audit.time > 0)

    # check that the OS source reproduces the existing labels
    first = labels.drop_duplicates('case_id').set_index('case_id')
    chk = first.join(os_table, how='inner')
    t_ok = ((chk.FUT.astype(float) - chk.time).abs() <= 1).sum()
    e_ok = (chk.Survival.astype(float) == chk.event).sum()

    new = audit[~audit.in_labels]
    reason = pd.Series('added', index=new.index)
    reason[new.n_modalities == 0] = 'no modality'
    reason[~new.has_OS] = 'no valid OS'
    if keep_mask is not None:
        reason[~keep_mask.reindex(new.index).fillna(False).astype(bool) & (reason == 'added')] = 'excluded histology'
    audit.loc[new.index, 'status'] = reason
    audit.loc[audit.in_labels, 'status'] = 'existing'
    added = reason[reason == 'added'].index

    rows = []
    for pid in added:
        r = audit.loc[pid]
        event = int(r.event)
        base = {'case_id': pid, 'True_Label': 'Dead' if event == 1 else 'Alive',
                'FUT': f'{float(r.time):.1f}', 'Survival': str(event)}
        pid_slides = sorted(slides[slides == pid].index)
        for s in pid_slides or ['']:
            rows.append(dict(base, slide_id=s))
    updated = pd.concat([labels[LABEL_COLS], pd.DataFrame(rows, columns=LABEL_COLS)], ignore_index=True)

    os.makedirs(out_dir, exist_ok=True)
    updated.to_csv(os.path.join(out_dir, label_file), sep='\t', index=False)
    audit.to_csv(os.path.join(out_dir, 'audit', label_file.replace('_labels.csv', '_audit.csv')))

    added_df = audit.loc[added]
    print(f'\n### {name}  ({label_file})')
    print(f'  OS source check on existing patients: time {t_ok}/{len(chk)}, event {e_ok}/{len(chk)} '
          f'({len(first) - len(chk)} existing patients not in source)')
    if removed:
        print(f'  existing patients removed (histology): {removed}')
    if overridden:
        print(f'  OS replaced (follow-up was <= 0 days): ' + ', '.join(f'{p} -> {OS_OVERRIDES[p][2]}' for p in overridden))
    if non_positive:
        print(f'  existing patients removed (follow-up <= 0 days, no OS found elsewhere): {len(non_positive)}')
    print(f'  patients: existing {len(labeled)} + added {len(added)} = {len(labeled) + len(added)}'
          f'  | rows {len(labels)} -> {len(updated)}')
    print(f'  added patients by modality: ' +
          ', '.join(f'{m} {int(added_df[m].sum())}' for m in mod_cols) +
          f'  | without WSI: {int((~added_df.WSI).sum())}')
    skipped = reason[reason != 'added'].value_counts().to_dict()
    if skipped:
        print(f'  not added: {skipped}')
    return audit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default=os.path.join(REPO, 'dataloader', 'dataset_updated'))
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, 'audit'), exist_ok=True)

    ct_map, mri_map = pd.read_csv(CT_MAPPING), pd.read_csv(MRI_MAPPING)
    ccrcc_clinical = read_ids(f'{CCRCC}/multi-modal-ist.github.io/datasets/ccRCC/Code/clinical+genomic_split.csv')

    # ---------------- CPTAC ccRCC
    p = yaml_params('ccRCC.yaml')
    cos = cptac_os()
    mods = {
        'Genomics': set().union(*[read_ids(x) for x in p['genomics_path']]),
        'CT': set().union(*[imaging_on_disk(ct_map, x) for x in p.get('ct_path', [])]),
        'MRI': set().union(*[imaging_on_disk(mri_map, x) for x in p.get('mri_path', [])]),
        'Clinical': {x for x in ccrcc_clinical if x.startswith('C3')} | set(cos.index),
    }
    build('CPTAC ccRCC', 'CPTAC_CCRCC_labels.csv', cos[['time', 'event']],
          wsi_on_disk(p['pt_files_path'], 9), mods, args.out,
          keep_mask=cos.morphology.isin(CPTAC_MORPHOLOGY_KEEP))

    # ---------------- TCGA
    cdr = pd.read_excel(TCGA_CDR, sheet_name=0)
    cbio = pd.read_csv(CBIO_CLINICAL, sep='\t', low_memory=False)
    for cohort, source in TCGA_COHORTS.items():
        p = yaml_params(f'TCGA_{cohort}_dataset*.yaml')
        pt_dirs = set(glob.glob(f'{TCGA_SHARED}/{cohort}/features_*/pt_files')) | p.get('pt_files_path', set())
        ge = p.get('genomics_path', {f'{TCGA_SHARED}/{cohort}/gene_expression/fpkm_unstranded.csv'})
        mods = {'Genomics': set().union(*[read_ids(x) for x in ge])}
        if 'cnv_path' in p:
            mods['CNV'] = set().union(*[read_ids(x) for x in p['cnv_path']])
        if 'ct_path' in p:
            mods['CT'] = set().union(*[imaging_on_disk(ct_map, x) for x in p['ct_path']])
        if 'mri_path' in p:
            mods['MRI'] = set().union(*[imaging_on_disk(mri_map, x) for x in p['mri_path']])
        if 'clinical_path' in p:
            mods['Clinical'] = {x for x in set().union(*[read_ids(x) for x in p['clinical_path']]) if x.startswith('TCGA')}
        build(f'TCGA {cohort} (OS: {source})', f'TCGA_{cohort}_labels.csv', tcga_os(cohort, source, cdr, cbio),
              wsi_on_disk(pt_dirs, 12), mods, args.out)


if __name__ == '__main__':
    main()
