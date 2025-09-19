import pandas as pd
import os

tumors = ['BLCA', 'BRCA', 'COAD', 'HNSC', 'KIRC', 'KIRP','LUAD', 'LUSC', 'OV','STAD']
base_dir = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider"
output_dir = os.path.join(base_dir, "splits", "aug_splits_train_val_test")
ok = True

os.makedirs(output_dir, exist_ok=True)

for tumor in tumors:
    print(f"\n### Tumor: {tumor}")
    tumor_output_dir = os.path.join(output_dir, tumor)
    os.makedirs(tumor_output_dir, exist_ok=True)

    for fold in range(5):
        aug_file = os.path.join(base_dir, f"splits/aug_splits_train_val_test/{tumor}/splits_{fold}.csv")
        base_file = os.path.join(base_dir, f"utils/splits_train_val_test/{tumor}/splits_{fold}.csv")

        # Leggi i CSV
        aug_df = pd.read_csv(aug_file)
        base_df = pd.read_csv(base_file)

        ok = True
        for split in ["val", "test"]:
            aug_patients = set(aug_df[split].dropna())
            base_patients = set(base_df[split].dropna())

            if aug_patients != base_patients:
                print(f"❌ Mismatch in {split} set - fold {fold}")
                print(f"aug only: {aug_patients - base_patients}")
                print(f"base only: {base_patients - aug_patients}")
                ok = False

        # Check train include at least all base train patients
        aug_train = set(aug_df['train'].dropna())
        base_train = set(base_df['train'].dropna())
        if not base_train.issubset(aug_train):
            print(f"❌ Mismatch in train set - fold {fold}")
            print(f"Missing patients in aug train set: {base_train - aug_train}")
            ok = False

        # Identifica pazienti nuovi in aug (su tutto il fold)
        all_aug = aug_train | set(aug_df['val'].dropna()) | set(aug_df['test'].dropna())
        all_base = base_train | set(base_df['val'].dropna()) | set(base_df['test'].dropna())
        new_patients = all_aug - all_base

        if new_patients:
            print(f"ℹ️ New patients in aug not in base (fold {fold}): {new_patients}")

        # Costruisci nuovo split
        new_train = base_train | new_patients
        new_val = set(base_df['val'].dropna())
        new_test = set(base_df['test'].dropna())

        # Prepara il nuovo dataframe per lo split
        max_len = max(len(new_train), len(new_val), len(new_test))
        new_split_df = pd.DataFrame({
            'train': list(new_train) + [None] * (max_len - len(new_train)),
            'val': list(new_val) + [None] * (max_len - len(new_val)),
            'test': list(new_test) + [None] * (max_len - len(new_test)),
        })

        # Salva il nuovo split
        new_split_file = os.path.join(tumor_output_dir, f"splits_{fold}.csv")
        new_split_df.to_csv(new_split_file, index=False)
        print(f"✅ Saved new split to: {new_split_file}")

    if ok:
        print(f"\n✅ All folds for tumor {tumor} passed the checks.")
    else:
        print(f"\n⚠️ There were mismatches for tumor {tumor}, review logs above.")
