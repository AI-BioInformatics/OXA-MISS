import os 
import pandas as pd

data_csv="/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/experiments/tests_results_Surv.csv"
df= pd.read_csv(data_csv)
col = 'modality_setting'
modalities_to_check = ["complete","missing_all_60","missing_all_30","missing_all_wsi_60", "missing_all_genomics_60"]

for mod in modalities_to_check:
    df_mod = df[df[col] == mod]
    if not df_mod.empty:
        print(f"Results for {mod}:")
        df_mod=df_mod[df_mod["model_version"] =='Lowest_Validation_Loss_Model']
        df_mod = df_mod[['dataset_name','seed','c-index_mean', 'c-index_std']]
        print(df_mod['dataset_name'].value_counts())
        if df_mod['dataset_name'].value_counts()['OV'].item() > 3:
            df_mod=df_mod.iloc[:-3]
        summary = df_mod.groupby('dataset_name').agg(['mean','std']).reset_index()
        df_mod.to_csv(f'/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/csv_results/OUR_MODEL_{mod}_results_OS_TCGA.csv', index=False)
        summary.to_csv(f'/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/csv_results/OUR_MODEL_{mod}_results_OS_TCGA_grouped.csv', index=False)
    else:
        print(f"No results found for {mod}")