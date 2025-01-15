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
path_1 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_kfold_3_all_tissues_YY2025-MM01-DD08-HH14-MM20_0910D0D7FE/last_epoch_test_df_Fold_1.h5"
path_2 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_kfold_3_all_tissues_YY2025-MM01-DD08-HH14-MM20_0910D0D7FE/last_epoch_test_df_Fold_2.h5"
path_3 = "/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/custom_Treatment_Response_T_PTRC-HGSOC_chemorefractory_OBR_expandedv2_V_chemorefractory_kfold_3_all_tissues_YY2025-MM01-DD08-HH14-MM20_0910D0D7FE/last_epoch_test_df_Fold_3.h5"

df_1 = pd.read_hdf(path_1).set_index('patient_ids')
df_2 = pd.read_hdf(path_2).set_index('patient_ids')
df_3 = pd.read_hdf(path_3).set_index('patient_ids')
#%%%%%%%%%%%%%%%%%%%%%
df = pd.concat([df_1, df_2, df_3])
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
