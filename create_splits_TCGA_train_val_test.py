import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# --- CONFIG -------------------------------------------------------------
#tumors = ["BLCA", "BRCA", "COAD", "HNSC", "KIRC","KIRP", "LUAD", "LUSC","OV","STAD"]
tumors= ["NACT_interval_PFI_cohort"] #"NACT_primary"
features_dir_base = "/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts" #"/work/h2020deciderficarra_shared/TCGA"
label_dir = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/dataloader/dataset"
out_base = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/utils/splits_train_val_test"
random_seed = 42
# ------------------------------------------------------------------------


def save_split_csv(train_ids, val_ids, test_ids, out_path):
    df = pd.DataFrame({
        "train": pd.Series(train_ids),
        "val": pd.Series(val_ids),
        "test": pd.Series(test_ids)
    })
    df.to_csv(out_path, index=False)


def at_least_one_event(idx_set, y, wanted_event=0):
    return np.any(y[idx_set] == wanted_event)


def generate_splits_for_tumor(tumor):
    # --- STEP 1: check pt_files and filter label file -------------------
    feature_dir = os.path.join(features_dir_base, tumor.split('PFI')[0].replace("_", "/"), "features_UNI", "pt_files")  #
    label_path = os.path.join(label_dir, f"Decider_{tumor}_all_tissues_labels.csv") #os.path.join(label_dir, f"TCGA_{tumor}_labels.csv")
    if not os.path.exists(feature_dir):
        print(f"❌ Directory mancante per {tumor}: {feature_dir}")
        return

    pt_files = os.listdir(feature_dir)
    valid_slide_ids = set([f.split(".pt")[0] for f in pt_files])

    df = pd.read_csv(label_path, sep="\t")
    df = df[df["slide_id"].isin(valid_slide_ids)]
    # df = df.dropna(subset=["FUT", "Survival"])
    df =df.dropna(subset=['PFI','Treatment_Response'])
    df = df.drop_duplicates(subset="case_id")

    # Sovrascrive il file delle label filtrato
    # df.to_csv(label_path, sep="\t", index=False)

    X = df["case_id"].values
    # y = df["Survival"].values.astype(int)  # 0 = morto, 1 = vivo
    y= df["Treatment_Response"].values.astype(int)  
    N = len(X)

    test_fraction = 0.10
    val_fraction = 0.10
    test_size = int(round(test_fraction * N))
    val_size = int(round(val_fraction * N))

    outer_split = StratifiedShuffleSplit(
        n_splits=5,
        test_size=test_fraction,
        random_state=random_seed
    )

    out_dir = os.path.join(out_base, tumor)
    os.makedirs(out_dir, exist_ok=True)

    for fold, (train_val_idx, test_idx) in enumerate(outer_split.split(X, y)):
        X_rem, y_rem = X[train_val_idx], y[train_val_idx]
        inner_val_fraction = val_size / (N - test_size)

        inner_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=inner_val_fraction,
            random_state=random_seed + fold
        )
        train_idx, val_idx = next(inner_split.split(X_rem, y_rem))

        # Garantisci almeno un morto se presenti nel dataset
        if (y == 0).sum() > 0:
            if not at_least_one_event(val_idx, y_rem, wanted_event=0):
                dead_in_train = train_idx[y_rem[train_idx] == 0]
                if len(dead_in_train):
                    swap = dead_in_train[0]
                    alive_in_val = val_idx[y_rem[val_idx] == 1][0]
                    train_idx = train_idx[train_idx != swap]
                    val_idx = np.where(val_idx == alive_in_val, swap, val_idx)

            if not at_least_one_event(test_idx, y, wanted_event=0):
                dead_global = train_val_idx[y[train_val_idx] == 0]
                dead_global = np.setdiff1d(dead_global, val_idx)
                if len(dead_global):
                    swap = dead_global[0]
                    alive_in_test = test_idx[y[test_idx] == 1][0]
                    test_idx = np.where(test_idx == alive_in_test, swap, test_idx)
                    train_idx = train_idx[train_idx != swap]

        train_ids = X_rem[train_idx]
        val_ids = X_rem[val_idx]
        test_ids = X[test_idx]

        out_file = os.path.join(out_dir, f"splits_{fold}.csv")
        save_split_csv(train_ids, val_ids, test_ids, out_file)

        def proportions(ids):
            sub_y = y[np.isin(X, ids)]
            c = Counter(sub_y)
            tot = len(sub_y)
            return {
                "dead": c.get(0, 0)/tot if tot else 0,
                "alive": c.get(1, 0)/tot if tot else 0
            }

        print(f"[{tumor}] Fold {fold}:")
        print(f"  Train: {proportions(train_ids)}")
        print(f"  Val:   {proportions(val_ids)}")
        print(f"  Test:  {proportions(test_ids)}\n")


# ----------- MAIN -------------------------------------------------------
for tumor in tumors:
    generate_splits_for_tumor(tumor)
