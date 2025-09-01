
import h5py

# Apri il file in modalità lettura
with h5py.File('/work/h2020deciderficarra_shared/TCGA/BRCA/features_UNI/h5_files/TCGA-Z7-A8R6-01Z-00-DX1.CE4ED818-D762-4324-9DEA-2ACB38B9B0B9.h5', 'r') as f:
    
    coords = f['coords'][:]
    features = f['features'][:]
    

    print('debug')