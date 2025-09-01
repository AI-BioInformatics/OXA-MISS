import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

# Percorsi file
genomic_path = "/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/Chemorefractory_bulk_counts.tsv"
labels_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/dataloader/dataset/Decider_chemorefractory_all_tissues_labels.csv"

# Caricamento file
genomic_df = pd.read_csv(genomic_path, sep="\t")
labels_df = pd.read_csv(labels_path, sep="\t")

# Pazienti unici nel file genomico
genomic_patients = genomic_df["patient"].unique()

# Filtra i pazienti comuni nei label
filtered_labels = labels_df[labels_df["case_id"].isin(genomic_patients)].copy()

# Mapping: patient ID e relativa label
patients = filtered_labels[["case_id", "Treatment_Response"]].drop_duplicates()

# Reset index
patients = patients.reset_index(drop=True)

# Prepara Stratified KFold
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Creazione fold
for i, (train_idx, val_idx) in enumerate(skf.split(patients["case_id"], patients["Treatment_Response"])):
    train_patients = patients.loc[train_idx, "case_id"].tolist()
    val_patients = patients.loc[val_idx, "case_id"].tolist()

    # Riduco a 20 pazienti in validazione (come richiesto)
    if len(val_patients) > 20:
        val_patients = val_patients[:20]

    # Rimuovo i 20 selezionati da val dal training set
    train_patients = [p for p in train_patients if p not in val_patients]

    # Costruisco il dataframe per l’output
    max_len = max(len(train_patients), len(val_patients))
    train_col = train_patients + [None] * (max_len - len(train_patients))
    val_col = val_patients + [None] * (max_len - len(val_patients))
    df_out = pd.DataFrame({"train": train_col, "val": val_col})

    # Salvo
    output_path = f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/Treatment_Response/chemorefractory/split_{i}.csv"
    df_out.to_csv(output_path, index=False)

    print(f"Fold {i+1} salvato: {output_path}")
