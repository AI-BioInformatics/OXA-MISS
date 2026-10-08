"""
Clinical data as tokens (OXA_MISS_final), from the clean table of all the studies
(clinical_data_clean/all_studies_clinical_clean.csv: one row per case_id, NaN for every missing value).

  numerical:   age, given RAW to the model: its z-score uses the training patients' mean / std of each fold,
               stored in the model (set_clinical_normalization), so a checkpoint is self-contained
  categorical: fixed vocabulary (same in every fold, no fit), index 0 = missing
  study_id:    optional categorical token (use_study_id): the study / cancer type of the patient

Configured by data_loader.clinical in the main yaml; with it, Clinical availability (has_Clinical) comes from
this table instead of the clinical_path of the dataset yamls.
"""
import numpy as np
import pandas as pd
import torch

CLEAN_CSV = "/work/H2020DeciderFicarra/ccRCC/clinical_data_clean/all_studies_clinical_clean.csv"

# Fixed vocabulary. After the cleaning every category has >= 100 patients over the 9 studies.
# race / ethnicity are not in the default: missing for 64% (CPTAC ethnicity) / 28% (KIRC) with a pattern that
# changes from study to study -> the model would learn "missing token = study".
CAT_VOCAB = {
    "sex":        ["female", "male"],
    "ajcc_stage": ["stage i", "stage ii", "stage iii", "stage iv"],
    "ajcc_t":     ["t1", "t2", "t3", "t4"],
    "ajcc_n":     ["n0", "n1", "n2", "n3"],
    "ajcc_m":     ["m0", "m1"],
    "race":       ["white", "black or african american", "asian",
                   "american indian or alaska native",
                   "native hawaiian or other pacific islander", "other"],
    "ethnicity":  ["not hispanic or latino", "hispanic or latino"],
}
DEFAULT_CAT_COLS = ["sex", "ajcc_stage", "ajcc_t", "ajcc_n", "ajcc_m"]
NUM_COLS = ["age"]
# BRCA, OV and HNSC excluded (HNSC empty, OV without T/N/M, BRCA out of the project)
DEFAULT_EXCLUDED_STUDIES = ["TCGA_BRCA", "TCGA_OV", "TCGA_HNSC"]


class ClinicalTokens:
    """case_id -> raw numerical values, their mask and the categorical indices of a patient."""
    DEFAULT_CATEGORICAL = DEFAULT_CAT_COLS

    def __init__(self, table=CLEAN_CSV, categorical=DEFAULT_CAT_COLS, use_study_id=False,
                 excluded_studies=DEFAULT_EXCLUDED_STUDIES):
        df = pd.read_csv(table, dtype={"case_id": str})
        if not df["case_id"].is_unique:
            raise ValueError(f"{table}: duplicated case_id")
        # study vocabulary from the whole table (before the exclusion): the same for every run
        studies = sorted(df["study_id"].dropna().unique())
        df = df[~df["study_id"].isin(excluded_studies)].reset_index(drop=True)
        self.cat_cols = list(categorical)
        unknown = [c for c in self.cat_cols if c not in CAT_VOCAB]
        if unknown:
            raise ValueError(f"clinical categorical columns {unknown} have no vocabulary (known {sorted(CAT_VOCAB)})")
        vocab = {c: CAT_VOCAB[c] for c in self.cat_cols}
        if use_study_id:
            self.cat_cols.append("study_id")
            vocab["study_id"] = studies
        self.cat_cardinalities = [len(vocab[c]) for c in self.cat_cols]
        self.use_study_id = use_study_id

        num = df[NUM_COLS].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float32)
        num_mask = ~np.isnan(num)
        cat = np.zeros((len(df), len(self.cat_cols)), dtype=np.int64)
        for j, c in enumerate(self.cat_cols):
            mapped = df[c].map({v: i + 1 for i, v in enumerate(vocab[c])})
            out_of_vocab = df[c].notna() & mapped.isna()
            if out_of_vocab.any():  # fail instead of silently treating them as missing
                raise ValueError(f"{c}: values out of vocabulary {sorted(df.loc[out_of_vocab, c].unique())}")
            cat[:, j] = mapped.fillna(0).astype(np.int64).to_numpy()
        # a patient has Clinical if at least one real clinical value is observed (the study token is not one)
        n_real = len(self.cat_cols) - int(use_study_id)
        observed = num_mask.any(axis=1) | (cat[:, :n_real] > 0).any(axis=1)
        self.num = np.where(num_mask, num, 0.0).astype(np.float32)
        self.num_mask, self.cat = num_mask, cat
        self.row = {cid: i for i, cid in enumerate(df["case_id"]) if observed[i]}
        self.token_names = NUM_COLS + self.cat_cols  # order of the model's clinical tokens

    def has(self, case_id):
        return case_id in self.row

    def num_stats(self, case_ids):
        """(mean, std) of each numerical column over the observed values of case_ids (the fold's training
        patients): the only statistics fitted on data, stored in the model."""
        rows = [self.row[c] for c in case_ids if c in self.row]
        mean, std = [], []
        for j in range(len(NUM_COLS)):
            values = self.num[rows, j][self.num_mask[rows, j]]
            if len(values) < 2:
                raise ValueError(f"clinical {NUM_COLS[j]}: {len(values)} training values, cannot standardize")
            mean.append(float(values.mean()))
            std.append(float(values.std()) or 1.0)
        return mean, std

    def cat_modes(self, case_ids):
        """Most frequent observed category (1-based index) of each categorical column over case_ids (the
        fold's training patients): the imputation of clinical_missing=impute (ablation)."""
        rows = [self.row[c] for c in case_ids if c in self.row]
        modes = []
        for j, col in enumerate(self.cat_cols):
            values = self.cat[rows, j]
            values = values[values > 0]
            modes.append(int(np.bincount(values).argmax()) if len(values) else 1)
        return modes

    def get(self, case_id):
        """Tensors of a patient without batch dimension (the DataLoader adds it); status False if missing."""
        i = self.row.get(case_id)
        if i is None:
            return self.missing()
        return {"clinical_status": True,
                "clinical_num": torch.from_numpy(self.num[i]),
                "clinical_num_mask": torch.from_numpy(self.num_mask[i]),
                "clinical_cat": torch.from_numpy(self.cat[i])}

    def missing(self):
        return {"clinical_status": False,
                "clinical_num": torch.zeros(len(NUM_COLS)),
                "clinical_num_mask": torch.zeros(len(NUM_COLS), dtype=torch.bool),
                "clinical_cat": torch.zeros(len(self.cat_cols), dtype=torch.long)}
