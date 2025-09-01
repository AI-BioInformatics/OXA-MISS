import pandas as pd 
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold
from collections import Counter
# --- CONFIG -------------------------------------------------------------
#tumors = ["BLCA", "BRCA", "COAD", "HNSC", "KIRC","KIRP", "LUAD", "LUSC","OV","STAD"]
tumors= ["NACT_interval_PFI_cohort"] #"NACT_primary"
features_dir_base = "/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts" #"/work/h2020deciderficarra_shared/TCGA"
label_dir = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/dataloader/dataset"
out_base = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/utils/splits"
random_seed = 42
# ------------------------------------------------------------------------


def save_split_csv(train_ids, val_ids, out_path):
    df = pd.DataFrame({
        "train": pd.Series(train_ids),
        "val": pd.Series(val_ids),
    })
    df.to_csv(out_path, index=False)
    
    
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
    
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_seed)


    out_dir = os.path.join(out_base, tumor)
    os.makedirs(out_dir, exist_ok=True)

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        train_ids = X[train_idx]
        val_ids = X[val_idx]

        out_file = os.path.join(out_dir, f"splits_{fold}.csv")
        save_split_csv(train_ids, val_ids, out_file)

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


# ----------- MAIN -------------------------------------------------------
for tumor in tumors:
    generate_splits_for_tumor(tumor)
