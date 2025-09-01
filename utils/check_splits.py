import pandas as pd

missing_rates = [0.0, 0.25, 0.5, 0.75, 1.0]
for df_name in  ["BLCA", "BRCA", "COAD", "HNSC", "OV", "STAD","KIRC", "KIRP", "LUAD", "LUSC"]: # "KIRC", "KIRP", "LUAD", "LUSC"
    print(df_name)
    df = pd.read_csv(f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/utils/missing_mod/{df_name}_missing_modality_table.csv")
    tot_patients = df.shape[0]
    accum_patients = 0
    for fold in df["fold"].unique():
        print("---Fold: ", fold)                
        print(f"--- ---num patients per fold: {df[df['fold'] == fold].shape[0]}")
        accum_patients += df[df['fold'] == fold].shape[0]
        print(f"--- ---")
        for rate in missing_rates:
            colname = f'genomics_miss_{int(rate * 100)}'
            print(f"--- ---Expected missing Rate: {rate}")
            print(f"--- ---Actual missing rate: {round(1-df[df['fold'] == fold][colname].astype(float).mean(), 2)}")
        

    if accum_patients != tot_patients:
        print(f"---Error: {accum_patients} != {tot_patients}")
    else:
        print(f"---OK: {accum_patients} == {tot_patients}")
