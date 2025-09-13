import os
import shutil
import pandas as pd

def filtra_csv_pazienti(csv_path_in, csv_path_out, csv_dir):
    pazienti = set()

    # leggi tutti i csv nella cartella
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            for col in ['train', 'val', 'test']:
                if col in df.columns:
                    pazienti.update(df[col].dropna().tolist())

    # leggi csv di input
    df_in = pd.read_csv(csv_path_in)

    # filtra in base alla colonna "patient"
    df_out = df_in[df_in['patient'].isin(pazienti)]

    # salva il risultato
    df_out.to_csv(csv_path_out, index=False)

if __name__ == '__main__':
    filtra_csv_pazienti(
        csv_path_in='/work/h2020deciderficarra_shared/TCGA/KIRC/gene_expression/fpkm_unstranded.csv',
        csv_path_out='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo_training/data/KIRC/gene_expression/fpkm_unstranded.csv',
        csv_dir='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo_training/splits/KIRC'
    )