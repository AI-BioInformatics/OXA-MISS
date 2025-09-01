import os
import re
import csv
import glob
import pandas as pd

# Carica una volta il file di riferimento
reference_csv_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/experiments/test_results_csv/TS_Custom_Multimodal_XA.csv"
reference_df = pd.read_csv(reference_csv_path)


# Assumiamo che ci sia una colonna chiamata 'ID' con dentro i run_id
# E che le colonne seed e modality_setting esistano
def process_file(file_path):
    try:
        with open(file_path, 'r') as f:
            content = f.read()

            # Check if file contains the required string
            if "The program took (DD--HH:MM:SS)" not in content:
                return None

            # Find results path
            results_pattern = r'INFO:root:project directory: (.*?)\n'
            results_match = re.search(results_pattern, content)
            if not results_match:
                return None
            results_path = results_match.group(1)

            # Find wandb run ID
            wandb_pattern = r'https://wandb\.ai/multimodal_decider/multimodal_decider/runs/([a-zA-Z0-9]+)'
            wandb_match = re.search(wandb_pattern, content)
            if not wandb_match:
                return None
            run_id = wandb_match.group(1)

            # Match nel reference_df
            match = reference_df[reference_df["ID"] == run_id]
            if match.empty:
                print(f"run_id {run_id} not found in reference CSV.")
                return None

            seed = match.iloc[0]["seed"]
            modality_setting = match.iloc[0]["modality_setting"]

            return {
                'results_path': results_path,
                'run_id': run_id,
                'seed': seed,
                'modality_setting': modality_setting
            }

    except Exception as e:
        print(f"Error processing file {file_path}: {str(e)}")
        return None

def main(folders):
    results = []
    
    # Process each folder
    
    for folder in folders:
        # Walk through all files in the folder
        for root, _, files in os.walk(folder):
            files = [f for f in files if 'chemorefractory' in f]
            for file in files:
                file_path = os.path.join(root, file)
                result = process_file(file_path)
                if result:
                    results.append(result)
    
    # Write results to CSV
    if results:
    # Converte i risultati attuali in DataFrame
        new_df = pd.DataFrame(results)
        output_file= '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/chemorefractory_TS.csv'
        if os.path.exists(output_file):
            # Legge il CSV esistente
            existing_df = pd.read_csv(output_file)

            # Concatena evitando duplicati (opzionale: per esempio su 'run_id')
            combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=["run_id"])
        else:
            combined_df = new_df
        updated_df = filter_runs(combined_df)
        # Scrive il CSV aggiornato
        updated_df.to_csv(output_file, index=False)
        print(f"Results written to {output_file} ({len(combined_df)} entries)")
    else:
        print("No matching files found")
# Estrai tumor_type dal path (tra '_' e '_kfold_5')
def extract_tumor(path):
    match = re.search(r'XA_(.*?)_kfold_5', path)
    return match.group(1) if match else None

def filter_runs(df):
    
    results_df= df.copy()
    results_df["tumor_type"] = results_df["results_path"].apply(extract_tumor)
    # merged = results_df.merge(reference_df[["ID", "c-index_mean"]], left_on="run_id", right_on="ID", how="left")
    merged =results_df.merge(
        reference_df[["ID",  "F1-Score_mean", "F1-Score_std", "AUC_mean", "AUC_std", "Accuracy_mean", "Accuracy_std"]],
        left_on="run_id",
        right_on="ID",
        how="left"
    )
    #F1-Score_mean,F1-Score_std,AUC_mean,AUC_std,Accuracy_mean,Accuracy_std
    # Rimuovi duplicati tenendo quello con c_index_mean più alto per ogni combo
    best_df = (
        #merged.sort_values("c-index_mean", ascending=False)
        merged.sort_values("F1-Score_mean", ascending=False)
        .drop_duplicates(subset=["tumor_type", "seed", "modality_setting"])
        .reset_index(drop=True)
    )

    # Salva se vuoi
    #best_df.to_csv("/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/runs_mapping_withseed.csv", index=False)
    return best_df    
if __name__ == "__main__":
    # Example usage
    folders = [
        # '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/slurm_out/TCGA_multimodal_completemodalities',
        # '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/slurm_out/TCGA_multimodal_missingall30',
        # '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/slurm_out/TCGA_multimodal_missingall60',
        '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/slurm_out/chemorefractory/'
    ]
    main(folders)
