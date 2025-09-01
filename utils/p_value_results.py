import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

# Carica i due file CSV per i due modelli (ad esempio model_1.csv e model_2.csv)
df_model_1 = pd.read_csv('path_to_model_1.csv')
df_model_2 = pd.read_csv('path_to_model_2.csv')

# Unisci i due dataframe sui tumori e sui seeds
df = pd.merge(df_model_1[['dataset_name', 'seed', 'c-index_mean']],
              df_model_2[['dataset_name', 'seed', 'c-index_mean']],
              on=['dataset_name', 'seed'],
              suffixes=('_model_1', '_model_2'))

# Lista dei tumori unici nel dataset
tumors = df['dataset_name'].unique()

# Crea dizionari per memorizzare i p-value dei test
wilcoxon_p_values = {}
friedman_p_values = {}

# Itera attraverso ogni tumore
for tumor in tumors:
    # Filtra i dati per il tumore corrente
    tumor_data = df[df['dataset_name'] == tumor]
    
    # Assicurati di avere 3 valori di c-index per ogni seed
    if len(tumor_data) == 3:
        # Estrai i valori di c-index per i 2 modelli (model 1 e model 2) per i 3 seeds
        c_index_model_1 = tumor_data['c-index_mean_model_1'].values
        c_index_model_2 = tumor_data['c-index_mean_model_2'].values
        
        # Applica il test di Wilcoxon (test appaiato tra i 2 modelli)
        stat_wilcoxon, p_value_wilcoxon = wilcoxon(c_index_model_1, c_index_model_2)
        wilcoxon_p_values[tumor] = p_value_wilcoxon
        
        # Applica il test di Friedman (per confrontare tutti i 3 seed)
        stat_friedman, p_value_friedman = friedmanchisquare(c_index_model_1[0], c_index_model_1[1], c_index_model_1[2],
                                                            c_index_model_2[0], c_index_model_2[1], c_index_model_2[2])
        friedman_p_values[tumor] = p_value_friedman

# Visualizza i p-value per ogni tumore
for tumor in tumors:
    print(f"p-value (Wilcoxon) per {tumor}: {wilcoxon_p_values[tumor]}")
    print(f"p-value (Friedman) per {tumor}: {friedman_p_values[tumor]}")

