#%%%%%%%%%%%%%%%%%%%%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, confusion_matrix,f1_score
from PIL import Image
import io
#%%%%%%%%%%%%%%%%%%%%%
# path_1 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_kfold_3_all_tissues_YY2025-MM01-DD08-HH14-MM20_0910D0D7FE/last_epoch_test_df_Fold_1.h5"
# path_2 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_kfold_3_all_tissues_YY2025-MM01-DD08-HH14-MM20_0910D0D7FE/last_epoch_test_df_Fold_2.h5"
# path_3 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_kfold_3_all_tissues_YY2025-MM01-DD08-HH14-MM20_0910D0D7FE/last_epoch_test_df_Fold_3.h5"
# paths = [path_1, path_2, path_3]

# template = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_chemorefractory_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_EFFDB88502/last_epoch_test_df_Fold_"
# paths = [template+f"{i}.h5" for i in range(1, 79)]
# template = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_chemorefractory_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_2E899A0282/last_epoch_test_df_Fold_"
# paths = [template+f"{i}.h5" for i in range(1, 76)]
# template = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_chemorefractory_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_0AAA4B1894/last_epoch_test_df_Fold_"
# paths = [template+f"{i}.h5" for i in range(1, 78)]
# template = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_B01564EE98/last_epoch_test_df_Fold_"
# paths = [template+f"{i}.h5" for i in range(1, 34)]

# template = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_B537E2C811/last_epoch_test_df_Fold_"
# paths = [template+f"{i}.h5" for i in range(1, 34)]

# template = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_3A700F6999/last_epoch_test_df_Fold_"
# paths = [template+f"{i}.h5" for i in range(1, 33)]



template1 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD17-HH17-MM06_EB0781F329/last_epoch_test_df_Fold_"
paths1 = [template1+f"{i}.h5" for i in range(1, 49)]

template2 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_loo_all_tissues_YY2025-MM01-DD16-HH11-MM08_B01564EE98/last_epoch_test_df_Fold_"
paths2 = [template2+f"{i}.h5" for i in range(1, 34)]

paths = paths1 + paths2

# df_1 = pd.read_hdf(path_1).set_index('patient_ids')
# df_2 = pd.read_hdf(path_2).set_index('patient_ids')
# df_3 = pd.read_hdf(path_3).set_index('patient_ids')

dfs = [pd.read_hdf(path).set_index('patient_ids') for path in paths]
#%%%%%%%%%%%%%%%%%%%%%
# df = pd.concat([df_1, df_2, df_3])
df = pd.concat(dfs)
#%%%%%%%%%%%%%%%%%%%%%
def accuracy_confusionMatrix_plot(log_dict, metrics_df):
    # Clear any existing plot
    plt.clf()
    # Extract labels and predictions from log_dict

    all_labels = np.array(log_dict["all_labels"])
    treatment_response_predictions = np.array(log_dict["treatment_response_predictions"])
    f1_score = round(metrics_df["F1-Score"].values[0],3)
    AUC = round(metrics_df["AUC"].values[0],3)
    
    # Calculate the confusion matrix
    cm = confusion_matrix(all_labels, treatment_response_predictions)
    # Plot the confusion matrix using seaborn heatmap
    sns.set(font_scale=2)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, 
                xticklabels=np.unique(all_labels), 
                yticklabels=np.unique(all_labels),
                annot_kws={"size": 45}, ax=ax)
    # Set labels and title
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title('Confusion Matrix')
    # Add F1-Score and AUC to the plot
    plt.figtext(0.5, -0.05, f'F1-Score: {f1_score} | AUC: {AUC}', ha="center", fontsize=18, fontweight='bold')

    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    # Close the figure after saving it to avoid memory issues
    plt.close(fig)
    image = Image.open(buffer)
    #image.save("/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/cm.png")
    
    # Return the PIL Image object to be logged later
    return image
#%%%%%%%%%%%%%%%%%%%%%
all_labels = df["all_labels"].values
all_predictions = df["treatment_response_predictions"].values
all_logits = torch.tensor(df["treatment_response_logits"].tolist())
# Calculate AUC
logits_for_auc = torch.softmax(all_logits, dim=1).numpy()[:, 1]
auc = roc_auc_score(all_labels, logits_for_auc)    
f1 = f1_score(all_labels, all_predictions, average='macro')
accuracy = np.mean(all_labels == all_predictions) 
metrics_df = pd.DataFrame({"AUC": [auc], "F1-Score": [f1], "Accuracy": [accuracy]})
#%%%%%%%%%%%%%%%%%%%%%
log_dict = {"all_labels": all_labels, "treatment_response_predictions": all_predictions}
image = accuracy_confusionMatrix_plot(log_dict, metrics_df)
image
#%%%%%%%%%%%%%%%%%%%%%
