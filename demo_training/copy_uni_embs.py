import os
import shutil
import pandas as pd


def copia_file_tcga(path_in, path_out, csv_dir):
    pazienti = set()

    # leggi tutti i csv nella cartella
    for csv_file in os.listdir(csv_dir):
        if csv_file.endswith(".csv"):
            df = pd.read_csv(os.path.join(csv_dir, csv_file))
            for col in ['train', 'val', 'test']:
                if col in df.columns:
                    pazienti.update(df[col].dropna().tolist())

    # print('pazienti: \n', pazienti)
    os.makedirs(path_out, exist_ok=True)

    # copia i file.pt corrispondenti
    for fname in os.listdir(path_in):
        if fname.endswith(".pt"):
            paz_id = "-".join(fname.split("-")[:3])  # es: TCGA-A3-331
            if paz_id in pazienti:
                # print(f'{paz_id} in pazienti')
                shutil.copy(os.path.join(path_in, fname),
                            os.path.join(path_out, fname))
            # else:
            #     print(f'{paz_id} NOT in pazienti')

if __name__ == '__main__':
    copia_file_tcga(
        path_in='/work/h2020deciderficarra_shared/TCGA/KIRC/features_UNI/pt_files',
        path_out='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo_training/data/KIRC/features_UNI/pt_files',
        csv_dir='/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo_training/splits/KIRC'
    )
    print('finished')