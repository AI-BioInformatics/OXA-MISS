#%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#%%%%%%%%%%%%%%%%%%%%%
results_path = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/abmilTweak_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_V_chemorefractory_kfold_3_all_tissues_YY2024-MM10-DD24-HH19-MM07_E47501EAE1"
fold_1 = pd.read_hdf(results_path + "/last_epoch_test_df_Fold_1.h5", key="df", mode="r")
fold_2 = pd.read_hdf(results_path + "/last_epoch_test_df_Fold_2.h5", key="df", mode="r")
fold_3 = pd.read_hdf(results_path + "/last_epoch_test_df_Fold_3.h5", key="df", mode="r")
#%%%%%%%%%%%%%%%%%%%%%

#%%%%%%%%%%%%%%%%%%%%%