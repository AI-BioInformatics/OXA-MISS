import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
from collections import Counter
# Define the missing rates (from 0% to 100% with 25% gaps)
missing_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
modalities = ['genomics', 'wsi']
missing_all_rates = [0.3, 0.6]
type_dataset = "TCGA" # Decider or TCGA
# Set random seed for reproducibility
np.random.seed(42)
# Determine True_Label, FUT, and Survival
def get_true_label(row):
    if 'demographic.vital_status' in row:
        return 'Alive' if row['demographic.vital_status'] == 'Alive' else 'Dead'
    else:
        return 'Alive' if row['vital_status'] == 'Alive' else 'Dead'
def safe_binning(series, bins):
    binned = pd.cut(series, bins=bins, labels=False, include_lowest=True)
    binned_filled = binned.copy()
    binned_filled[series < bins[0]] = 0
    binned_filled[series > bins[-1]] = len(bins) - 2
    return binned_filled.astype(int)

def get_fut(row):
    if 'demographic.vital_status' in row:
        if row['demographic.vital_status'] == 'Alive':
            return row['days_to_last_follow_up']
        else:
            return row['demographic.days_to_death']
    else:
        if row['vital_status'] == 'Alive':
            return row['last_contact_days_to']
        else:
            return row['death_days_to']

def get_survival(row):
    if 'demographic.vital_status' in row:
        return 1 if row['demographic.vital_status'] == 'Alive' else 0
    else:
        return 1 if row['vital_status'] == 'Alive' else 0
def at_least_one_event(idx_set, y, wanted_event=0):
    return np.any(y[idx_set] == wanted_event)

def save_split_csv(train_ids, val_ids, test_ids, out_path):
    df = pd.DataFrame({
        "train": pd.Series(train_ids),
        "val": pd.Series(val_ids),
        "test": pd.Series(test_ids)
    })
    df.to_csv(out_path, index=False)

rename_mapping = {
'submitter_id': 'case_id',
'demographic.vital_status': 'vital_status',
'demographic.days_to_death': 'death_days_to',
'days_to_last_follow_up': 'last_contact_days_to'}
out_base =     "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits"
    
for df_name in  [ 
                 "BLCA", 
                    "BRCA", 
                    "COAD", 
                    "HNSC", 
                    "KIRC", 
                    "KIRP", 
                    "LUAD", 
                    "LUSC",
                    "OV", 
                    "STAD"
                    # #"NACT_interval_PFI_cohort",
                   
                    ]: 
    os.makedirs(out_base, exist_ok=True)
    if type_dataset == "Decider":
        metadata_path = f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/dataloader/dataset/{type_dataset}_{df_name}_all_tissues_labels.csv"
    else:
        metadata_path = f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/dataloader/dataset/{type_dataset}_{df_name}_labels.csv"
    metadata_df = pd.read_csv(metadata_path, sep="\t")

    clinical_path = f"/work/H2020DeciderFicarra/D2_4/docs/clinical_UCSC_updated/{df_name}_survival.csv"
    clinical_updated = pd.read_csv(clinical_path, sep="\t") 
    clinical_updated = clinical_updated.rename(columns={'_PATIENT': 'case_id', 'OS' : 'Survival', 'OS.time': 'FUT'})
    inverted_survival = clinical_updated['Survival'].map({1: 0, 0: 1})
    clinical_updated['Survival'] = inverted_survival
    clinical_updated = clinical_updated[['case_id','Survival', 'FUT']]
    clinical_updated.drop_duplicates(inplace=True)
    # Aggiorna i dati di clinical_missing con clinical_updated
    survival_dict = clinical_updated.set_index('case_id')['Survival'].to_dict()
    fut_dict = clinical_updated.set_index('case_id')['FUT'].to_dict()

    # Aggiorna direttamente i campi in metadata_df
    for _, row in clinical_updated.iterrows():
        case_id = row['case_id']
        if case_id in metadata_df['case_id'].values:
            idxs = metadata_df.index[metadata_df['case_id'] == case_id].to_list()
            for idx in idxs:
                if pd.notna(row['FUT']) and row['FUT'] != metadata_df.at[idx, 'FUT']:
                    metadata_df.at[idx, 'Survival'] = row['Survival']
                    metadata_df.at[idx, 'FUT'] = row['FUT']
                    metadata_df.at[idx, 'True_Label'] = 'Alive' if row['Survival'] == 1 else 'Dead'
    # Drop samples where "FUT" or "Survival" is NaN
    metadata_df.to_csv(metadata_path, sep="\t", index=False)
    metadata_df = metadata_df.drop_duplicates("case_id")
    metadata_df = metadata_df[metadata_df["FUT"] != "[Discrepancy]"]
    metadata_df = metadata_df.dropna()
    metadata_df["FUT"] = metadata_df["FUT"].astype(float)
    metadata_df = metadata_df.dropna(subset=["FUT", "Survival"])
    if type_dataset == "TCGA":
        df_base_path = f'/work/h2020deciderficarra_shared/TCGA/{df_name}'
    elif type_dataset == "Decider":
        if "NACT_interval" in df_name:
            df_base_path = f'/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/NACT/interval'
        elif df_name == "PDS":
            df_base_path = f'/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/{df_name}'
        
    df_wsi_path = os.path.join(df_base_path, 'features_UNI/pt_files')
    if type_dataset == "Decider":
        if 'PFI' in df_name:
            df_genomics_path = f"/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/{df_name.split('_PFI')[0]}_bulk_counts.tsv"
        else:
            df_genomics_path = f"/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/{df_name}_bulk_counts.tsv"
    elif type_dataset == "TCGA":
        df_genomics_path = os.path.join(df_base_path, 'gene_expression/fpkm_unstranded.csv', )
    
    # ex:
    # TCGA-CF-A3MF-01A-01-TSA.47D11180-87D1-468A-8D8...
    #         A3MF --> pid, pid_group_index = 2
    
    if type_dataset == "Decider":
        separator= '_'
        case_id_group_index = 0
    elif type_dataset == "TCGA":
        case_id_group_index = 2
        separator = '-'
    df_wsi = pd.DataFrame.from_dict({'name': list(os.listdir(df_wsi_path))})
    df_wsi['case_id'] = df_wsi['name'].str.split(separator, n=case_id_group_index+1).str[:case_id_group_index+1].str.join(separator)
    if type_dataset == "Decider":
        df_genomics = pd.read_csv(df_genomics_path, sep="\t")
    elif type_dataset == "TCGA":
        df_genomics = pd.read_csv(df_genomics_path)
    df_genomics = df_genomics.rename(columns={'patient': 'case_id'})
    missing_pats = list(set(df_genomics['case_id']) - set(metadata_df['case_id']))

    if missing_pats != 0:
        print(f"WARNING: {df_name} has missing case_ids in gene expression data")
        basepath='/work/H2020DeciderFicarra/D2_4/docs'
        clinical_file = os.path.join(basepath,f'clinical_data_{df_name}.csv')
        clinical_df = pd.read_csv(clinical_file,sep='\t')
        if 'submitter_id' in clinical_df.columns:
            clinical_df.rename(columns={'submitter_id': 'case_id'}, inplace=True)
        else:   
            clinical_df.rename(columns={'bcr_patient_barcode': 'case_id'}, inplace=True)
        if os.path.exists(os.path.join(basepath,f'clinical_data_{df_name}_onlywsi.csv')):
            clinical_doubled=pd.read_csv(os.path.join(basepath,f'clinical_data_{df_name}_onlywsi.csv'))
            diff = pd.Index(clinical_doubled['submitter_id']).difference(clinical_df['case_id'])
            if not diff.empty:
                print("Ci sono case_id in clinical_doubled non presenti in clinical_df")
                # Filtra i case_id mancanti
                to_add = clinical_doubled[clinical_doubled['submitter_id'].isin(diff)].copy()

                # Rinomina le colonne
                to_add = to_add.rename(columns=rename_mapping)

                # Se vuoi puoi ridurre solo alle colonne esistenti in clinical_df:
                common_cols = [col for col in clinical_df.columns if col in to_add.columns]
                to_add = to_add[common_cols]

                # Concatena i dataframe
                clinical_df = pd.concat([clinical_df, to_add], ignore_index=True)
        clinical_missing = clinical_df[clinical_df['case_id'].isin(missing_pats)].copy()
        if not clinical_missing.empty:
            clinical_missing['True_Label'] = clinical_missing.apply(get_true_label, axis=1)
            clinical_missing['FUT'] = clinical_missing.apply(get_fut, axis=1)
            clinical_missing['Survival'] = clinical_missing.apply(get_survival, axis=1)
            
            cols_to_keep = ['case_id', 'True_Label', 'FUT', 'Survival']
            clinical_missing = clinical_missing[cols_to_keep]
            
            clinical_missing['Survival'] = clinical_missing['case_id'].map(survival_dict).combine_first(clinical_missing['Survival'])
            clinical_missing['FUT'] = clinical_missing['case_id'].map(fut_dict).combine_first(clinical_missing['FUT'])
            clinical_missing['True_Label'] = clinical_missing['Survival'].map({1: 'Alive', 0: 'Dead'}).combine_first(clinical_missing['True_Label'])
            metadata_df = pd.concat([metadata_df, clinical_missing], ignore_index=True)
            print(f"Added {len(clinical_missing)} missing case_ids from clinical data to metadata_df")
        else:
            print(f"No missing case_ids found in clinical data for {df_name}")
                

    # df_available_modalities = pd.DataFrame({'case_id': pd.concat([df_wsi['case_id'], df_genomics['case_id']]).drop_duplicates().reset_index(drop=True)})
    metadata_df['wsi_available'] = metadata_df['case_id'].isin(df_wsi.case_id)
    metadata_df['genomics_available'] = metadata_df['case_id'].isin(df_genomics.case_id)
    metadata_df['complete'] = metadata_df['wsi_available'] & metadata_df['genomics_available']
    print(metadata_df)
    metadata_df = metadata_df.dropna(subset=["FUT", "Survival"])
    df_filtered = metadata_df[metadata_df['complete']==True].copy()
    df_incomplete = metadata_df[metadata_df['complete'] == False].copy()    
    df_incomplete['fold'] = -1    
    # df_filtered['time_bin'] = pd.qcut(df_filtered['FUT'], q=4, labels=False, duplicates='drop').astype(int)
    # _, bins = pd.qcut(df_filtered['FUT'], q=4, retbins=True, duplicates='drop')
    df_filtered['FUT'] = pd.to_numeric(df_filtered['FUT'], errors='coerce')
    df_filtered['time_bin'], bins = pd.qcut(df_filtered['FUT'], q=4, labels=False, retbins=True, duplicates='drop')
    # df_filtered['time_bin'] = df_filtered['time_bin'].astype(int)
    df_filtered['time_bin'] = df_filtered['time_bin'].fillna(-1).astype(int)
    
    df_incomplete['FUT'] = pd.to_numeric(df_incomplete['FUT'], errors='coerce')
    # df_incomplete['time_bin'] = pd.cut(df_incomplete['FUT'], bins=bins, labels=False, include_lowest=True).astype(int)
    df_incomplete['time_bin'] = safe_binning(df_incomplete['FUT'], bins)

    df_incomplete['time_bin'] = df_incomplete['time_bin'].fillna(-1).astype(int)
    df_filtered['strata'] = df_filtered['Survival'].astype(str) + "_" + df_filtered['time_bin'].astype(str)
    df_incomplete['strata'] = df_incomplete['Survival'].astype(str) + "_" + df_incomplete['time_bin'].astype(str)
    
    df_filtered['fold'] = -1
    df_incomplete['fold'] = -1
    X = df_filtered["case_id"].values
    y = df_filtered["Survival"].values.astype(int)
    
    N_complete = len(X)
    N_incomplete = len(df_incomplete)
    N_total = N_complete + N_incomplete
    outer_split = StratifiedShuffleSplit(
        n_splits=5,
        test_size=0.2,
        random_state=42
    )
    for fold, (train_idx_full, val_test_idx) in enumerate(outer_split.split(X, y)):
        rng = np.random.RandomState(seed=42 + fold)
        # Split val/test
        y_val_test = y[val_test_idx]
        val_test_ids = X[val_test_idx]

        val_test_split = StratifiedShuffleSplit(
            n_splits=1,
            test_size=0.5,  # metà a val, metà a test
            random_state=42 + fold
        )
        val_idx_local, test_idx_local = next(val_test_split.split(val_test_ids, y_val_test))

        val_ids = val_test_ids[val_idx_local]
        test_ids = val_test_ids[test_idx_local]
        train_ids = X[train_idx_full]
        
   # --- STEP 5: Garantisci almeno un morto in val/test ---
        if (y == 0).sum() > 0:
            if not at_least_one_event(val_idx_local, y_val_test, wanted_event=0):
                dead_in_train = train_ids[y[train_idx_full] == 0]
                if len(dead_in_train):
                    swap = dead_in_train[0]
                    alive_in_val = val_ids[y[np.isin(X, val_ids)] == 1][0]
                    train_ids = train_ids[train_ids != swap]
                    val_ids = np.where(val_ids == alive_in_val, swap, val_ids)

            if not at_least_one_event(test_idx_local, y_val_test, wanted_event=0):
                dead_in_train = train_ids[y[train_idx_full] == 0]
                if len(dead_in_train):
                    swap = dead_in_train[0]
                    alive_in_test = test_ids[y[np.isin(X, test_ids)] == 1][0]
                    train_ids = train_ids[train_ids != swap]
                    test_ids = np.where(test_ids == alive_in_test, swap, test_ids)
         # --- STEP 6: Salva versione base (solo completi) ---
        # df_filtered.loc[df_filtered.iloc[val_ids].index,'fold'] = fold
        df_filtered=df_filtered.set_index('case_id', drop=False)
        df_filtered.loc[val_ids, 'fold'] = fold
        df_filtered =  df_filtered.drop(columns=['case_id']).reset_index()
        out_dir = os.path.join(out_base,'splits_train_val_test', df_name)
        os.makedirs(out_dir, exist_ok=True)
        out_file_base = os.path.join(out_dir, f"splits_{fold}.csv")
        save_split_csv(train_ids, val_ids, test_ids, out_file_base)

        # --- STEP 7: Crea versione con training espanso ---
        train_augmented = np.concatenate([train_ids, df_incomplete["case_id"].values])
        print(f'Added {len(train_augmented)- len(train_ids)} to the training set')
        def proportions(ids):
            sub_y = y[np.isin(X, ids)]
            c = Counter(sub_y)
            tot = len(sub_y)
            return {
                "dead": c.get(0, 0)/tot if tot else 0,
                "alive": c.get(1, 0)/tot if tot else 0
            }

        print(f"[{df_name}] Fold {fold}:")
        print(f"  Train: {proportions(train_ids)}")
        print(f"  Val:   {proportions(val_ids)}")
        print(f"  Test:  {proportions(test_ids)}\n")
        print(f"  Train augmented: {proportions(train_augmented)}")
        
        
        out_dir_aug=os.path.join(out_base,'aug_splits_train_val_test', df_name)
        os.makedirs(out_dir_aug, exist_ok=True)
        out_file_aug = os.path.join(out_dir_aug, f"splits_{fold}.csv")
        save_split_csv(train_augmented, val_ids, test_ids, out_file_aug)


    # skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # for fold, (train_idx, val_idx) in enumerate(skf.split(df_filtered, df_filtered['strata'])):
    #     df_filtered.loc[df_filtered.iloc[val_idx].index,'fold'] = fold
    # print(df_filtered)
    #df_filtered= pd.concat([df_filtered, df_incomplete], ignore_index=True)
    for rate in missing_rates:
        for modality in modalities:
            colname = f'{modality}_miss_{int(rate * 100)}'
            # Start by copying the original genomic information
            df_filtered[colname] = df_filtered[f'{modality}_available']
            df_incomplete[colname] = df_incomplete[f'{modality}_available']
            # Apply the missing simulation within each fold
            for fold in df_filtered['fold'].unique():
                fold_idx = df_filtered[df_filtered['fold'] == fold].index
                n_samples = len(fold_idx)
                # Determine the number of samples to drop for this fold
                n_missing = int(round(rate * n_samples))
                # Randomly select indices within the fold to set as missing
                missing_indices = np.random.choice(fold_idx, size=n_missing, replace=False)
                df_filtered.loc[missing_indices, colname] = False
    # Add missing columns for incomplete data
    ######################################################################################################################
    # MISSING ALL PART
    ######################################################################################################################
    '''
    PROMPT: (CLAUDE 3.5 SONNET)
        ho un dataframe pandas df_filtered. ho due liste:
        modalities = ['genomics', 'wsi']
        missing_all_rates = [0.3, 0.6]
        per ogni modalita e per ogni rate deve essere creata una colonna missing_all_modality_ratio (in ratio riporta solo la parte dopo il punto)
        una volta che ho queste colonne, per ogni rate bisogna fare questa operazione:
        se ad esempio il rate è 0.3, ovvero 30%, devo calcolare la parte complementare, ovvero: 100-30 =70%, 
        quindi seleziono il 70% di tutte le righe presenti nel dataset e per ognuna delle colonne create in precedenza (con rate=30%) assegno valore = True.
        Per il restante 30% delle righe invece devo fare in modo che ogni modalità abbia probabilità 1/N di essere False. 
        L'unico vincolo è che per ongi riga almeno una modalità per ogni rate sia == True.
    '''

    # Create new columns for each modality and rate combination
    for rate in missing_all_rates:
        for modality in modalities:
            # Create column name using rate without decimal point
            col_name = f'missing_all_{modality}_{str(rate).split(".")[1]}'
            # Initialize all values to False
            df_filtered[col_name] = False
            df_incomplete[col_name] = True

    # Process each missing rate
    for rate in missing_all_rates:
        # Calculate complementary percentage (e.g., 0.3 -> 0.7)
        keep_rate = 1 - rate
        
        # Get number of rows to keep (non-missing)
        n_rows = len(df_filtered)
        n_keep = int(n_rows * keep_rate)
        
        # Randomly select indices for non-missing rows
        # keep_indices = np.random.choice(n_rows, size=n_keep, replace=False)
        keep_indices = np.random.choice(df_filtered.index, size=n_keep, replace=False)

        # Get column names for current rate
        rate_cols = [f'missing_all_{mod}_{str(rate).split(".")[1]}' for mod in modalities]

        # Set True for kept indices (non-missing data)
        for col in rate_cols:
            df_filtered.loc[keep_indices, col] = True
        
        # Process remaining indices (missing data)
        missing_indices = list(set(df_filtered.index) - set(keep_indices))
        
        # Ensure at least one modality is True for each row
        for idx in missing_indices:
            # Randomly select at least one modality to be True
            n_true = np.random.randint(1, len(modalities))
            true_modalities = np.random.choice(rate_cols, size=n_true, replace=False)
            
            for col in rate_cols:
                if col in true_modalities:
                    df_filtered.loc[idx, col] = True
                else:
                    df_filtered.loc[idx, col] = False

    # Verify that each row has at least one True for each rate
    for rate in missing_all_rates:
        rate_cols = [f'missing_all_{mod}_{str(rate).split(".")[1]}' for mod in modalities]
        has_true = df_filtered[rate_cols].any(axis=1)
        if not has_true.all():
            print(f"Warning: Some rows have all False values for rate {rate}")
        assert has_true.all(), f"Some rows have all False values for rate {rate}"

    for rate in missing_all_rates:
        str_rate = str(rate).split(".")[1]
        if len(str_rate) == 1:
            for modality in modalities:
                col_name = f'missing_all_{modality}_{str_rate}'
                df_filtered = df_filtered.rename(columns={col_name: col_name + '0'})
                df_incomplete = df_incomplete.rename(columns={col_name: col_name + '0'})
            
    ######################################################################################################################
    # 
    ######################################################################################################################
    
    print(df_filtered)
    df_filtered = pd.concat([df_filtered, df_incomplete], ignore_index=True)
    # out_path_splits = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits"
    missing_mod_table = df_filtered.drop("slide_id", axis=1)
    os.makedirs("/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/aug_missing_mod", exist_ok=True)
    missing_mod_table.to_csv(os.path.join("/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/splits/aug_missing_mod", f"{df_name}_missing_modality_table.csv"), index=False)
    
    # # Save train and validation splits for each fold
    # for fold in df_filtered['fold'].unique():
    #     train_df = df_filtered[df_filtered['fold'] != fold][['case_id']].copy()
    #     val_df = df_filtered[df_filtered['fold'] == fold][['case_id']].copy()
    #     # Convert the 'case_id' columns to lists
    #     train_list = train_df['case_id'].tolist()
    #     val_list = val_df['case_id'].tolist()
    #     max_len = max(len(train_list), len(val_list))
    #     train_list.extend([None] * (max_len - len(train_list)))
    #     val_list.extend([None] * (max_len - len(val_list)))
        
    #     # Create a DataFrame with train and val columns
    #     split_df = pd.DataFrame({
    #         'train': train_list,
    #         'val': val_list
    #     })
        
    #     # Save the split DataFrame to a CSV file
    #     os.makedirs(os.path.join(out_path_splits, df_name), exist_ok=True)
    #     split_df.to_csv(os.path.join(out_path_splits, df_name, f"splits_{fold}.csv"), index=False)


    # df_available_modalities.to_csv(f'/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/TCGA_available_modalities/{df_name}_available_modalities.csv', index=False)



