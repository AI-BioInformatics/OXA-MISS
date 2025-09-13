import os,json
import pandas as pd

split_dir = '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo/splits/KIRC'

pt_files_source = '/work/h2020deciderficarra_shared/TCGA/KIRC/features_UNI/pt_files'
pt_files_dest = '/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo/data/KIRC/features_UNI/pt_files'
genomics_path_source = "/work/h2020deciderficarra_shared/TCGA/KIRC/gene_expression/fpkm_unstranded.csv"
genomics_path_dest = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/demo/data/KIRC/gene_expression/fpkm_unstranded.csv" 

total_test_patients = set()

for fold in os.listdir(split_dir):
    fold_path = os.path.join(split_dir, fold)
    # leggo il csv
    df = pd.read_csv(fold_path)
    test_patients = df['test'].head(10).tolist()

    total_test_patients.update(test_patients)
    
    # copio in pt_files_dest i file da pt_files_source che inziano con l'id di uno dei test patients
    # for patient in test_patients:
    #     for file in os.listdir(pt_files_source):
    #         if file.startswith(patient):
    #             src_file = os.path.join(pt_files_source, file)
    #             dest_file = os.path.join(pt_files_dest, file)
    #             print(f"Copying {src_file} to {dest_file}")
    #             os.system(f"cp {src_file} {dest_file}")

# creo il csv della genomica copiando dal csv source tutte le righe che hanno un valore nella colonna 'patient' incluso in test_patients
genes_list = []
genes_groups = json.load(open('genes_groups/pathways_ensg.json', 'r'))
for group_id, group in genes_groups.items():
    genes_list += group['ensg_gene_id']
print(len(genes_list), "genes in the group")

genomics_df = pd.read_csv(genomics_path_source)

genes_list = [gene for gene in genes_list if gene in genomics_df.columns]

columns_of_interest = ['patient'] + list(set(genes_list))
genomics_df = genomics_df[columns_of_interest]
genomics_df = genomics_df[genomics_df['patient'].isin(list(total_test_patients))]
genomics_df.to_csv(genomics_path_dest, index=False)