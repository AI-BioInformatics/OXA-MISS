import pandas as pd

# Percorso file labels
labels_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/dataloader/dataset/Decider_chemorefractory_all_tissues_labels.csv"
labels_df = pd.read_csv(labels_path, sep="\t")

# Crea dizionario con case_id -> Treatment_Response
label_map = labels_df.drop_duplicates(subset="case_id").set_index("case_id")["Treatment_Response"].to_dict()

# Percorso base per i fold
base_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/Treatment_Response/chemorefractory"

# Itera i tre fold
for split_index in range(3):
    fold_path = f"{base_path}/split_{split_index}.csv"
    fold_df = pd.read_csv(fold_path)

    print(f"\n--- Fold {split_index} ---")

    for col in ["train", "val"]:
        # Rimuove NaN e cerca la label
        patients = fold_df[col].dropna().astype(str).tolist()
        count_label_1 = sum(label_map.get(pid, 0) == 1 for pid in patients)
        percentuale = count_label_1 / len(patients) * 100 if patients else 0
        print(f"{col}: {count_label_1} ({round(percentuale, 2)} %) pazienti con Treatment_Response == 1")
