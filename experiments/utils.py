import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, torch, os
from PIL import Image
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score, confusion_matrix,f1_score

# def KaplanMeier_plot_deprecated(log_dict):        
#     all_event_times = np.array(log_dict["all_event_times"])
#     all_censorships = np.array(log_dict["all_censorships"])
#     all_risk_scores = np.array(log_dict["all_risk_scores"])
#     mean_risk_score = np.mean(all_risk_scores)
#     high_risk_events = all_event_times[all_risk_scores >= mean_risk_score]
#     low_risk_events = all_event_times[all_risk_scores < mean_risk_score]

#     kmf = KaplanMeierFitter()
#     kmf.fit(high_risk_events, event_observed=(1 - all_censorships[all_risk_scores >= mean_risk_score]))
#     kmf.plot(label='High Risk')
#     kmf.fit(low_risk_events, event_observed=(1 - all_censorships[all_risk_scores < mean_risk_score]))
#     kmf.plot(label='Low Risk')

#     plt.title('Kaplan-Meier Survival Curve')
#     plt.xlabel('Time')
#     plt.ylabel('Survival Probability')
#     plt.legend()
#     plt.savefig('/work/H2020DeciderFicarra/D2_4/chemorefractory/MultimodalDecider/km.png') 
def move_to_device(data, device):
    if isinstance(data, dict):
        # Recursively call for each value in the dictionary
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        # Recursively call for each item in the list
        return [move_to_device(value, device) for value in data]
    elif isinstance(data, torch.Tensor):
        # Move tensor to the device
        return data.to(device)
    else:
        # If not a tensor or a collection, return the value as is
        return data 


def KaplanMeier_plot(log_dict):
    all_event_times = np.array(log_dict["all_event_times"])
    all_censorships = np.array(log_dict["all_censorships"])
    all_risk_scores = np.array(log_dict["all_risk_scores"])

    df = pd.DataFrame({
        "time": all_event_times,
        "event": all_censorships,
        "risk_score": all_risk_scores
    })
    # Categorize risk scores into quantiles for simplicity
    df['risk_group'] = pd.qcut(df['risk_score'], 2, labels=["Low", "High"])
    df['time'] = df['time'] / 365
    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(13, 13), dpi=300)
    colors = {'Low': '#364C83', 'Medium': '#88CCEE', 'High': '#C1423A'}
    risk_function = {}
    for name, grouped_df in df.groupby('risk_group'):
        kmf.fit(grouped_df['time'], event_observed=grouped_df['event'], label=name)
        risk_function[name] = kmf.survival_function_
        # Change color based on risk group and set ci_alpha for transparency of CI
        kmf.plot(ax=ax, color=colors[name], ci_alpha=0.075, linewidth=3, marker='+', markersize=8)

    # plt.title('Kaplan-Meier Curves for {}'.format(exp_name))
    plt.xlabel('Time (years)')
    plt.ylabel('Proportion surviving')
    plt.show()
    # For the logrank test, compare each group against each other
    p_values = []
    groups = df['risk_group'].unique()
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:  # Avoid repeating comparisons
                data1 = df[df['risk_group'] == group1]
                data2 = df[df['risk_group'] == group2]
                result = logrank_test(data1['time'], data2['time'], event_observed_A=data1['event'], event_observed_B=data2['event'])
                print(f"Logrank test between {group1} and {group2}: p-value = {result.p_value}")   
                p_values.append(result.p_value)
    # add p-values to the plot
    for i, group1 in enumerate(groups):
        for j, group2 in enumerate(groups):
            if i < j:
                plt.text(0.5, 0.5, f"p-value: {p_values.pop(0)}", fontsize=12, ha='center', va='center', transform=ax.transAxes)
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

def kfold_results_merge(result_id_path):
    prefisso = "last_epoch_test_df_Fold_"
    # cartella = f"/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/results/{result_id}"
    cartella = result_id_path
    paths = [
        os.path.join(result_id_path,f)
        for f in os.listdir(cartella)
        if os.path.isfile(os.path.join(cartella, f)) and f.startswith(prefisso)
    ]

    dfs = [pd.read_hdf(path).set_index('patient_ids') for path in paths]

    is_loo_case = False
    if len(dfs):
        is_loo_case = len(dfs[0]['all_labels']) == 1
    
    df = pd.concat(dfs)

    all_labels = df["all_labels"].values
    all_predictions = df["treatment_response_predictions"].values
    all_logits = torch.tensor(df["treatment_response_logits"].tolist())
    
    logits_for_auc = torch.softmax(all_logits, dim=1).numpy()[:, 1]
    auc = roc_auc_score(all_labels, logits_for_auc)    
    f1 = f1_score(all_labels, all_predictions, average='macro')
    accuracy = np.mean(all_labels == all_predictions) 
    metrics_df = pd.DataFrame({"AUC": [auc], "F1-Score": [f1], "Accuracy": [accuracy]})
    
    log_dict = {"all_labels": all_labels, "treatment_response_predictions": all_predictions}
    image =  accuracy_confusionMatrix_plot(log_dict, metrics_df)
    
    out = {
        "AUC" : auc,
        "Accuracy" : accuracy,
        "F1-Score" : f1
    }
    
    if is_loo_case:
        return out
    out["Confusion_Matrix"] = image
    return out

def predTime_vs_actualTime_confusionMatrix_plot(self, log_dict):
    patient_ids = log_dict["patient_ids"]
    all_risk_scores = np.array(log_dict["all_risk_scores"])
    all_censorships = np.array(log_dict["all_censorships"])
    all_event_times = np.array(log_dict["all_event_times"])
    
    # Calculate the predicted labels based on risk scores
    predicted_labels = np.where(all_risk_scores > 0.5, 1, 0)
    
    # Create the confusion matrix
    confusion_matrix = np.zeros((2, 2))
    for i in range(len(patient_ids)):
        true_label = all_censorships[i]
        predicted_label = predicted_labels[i]
        confusion_matrix[true_label][predicted_label] += 1
    
    # Plot the confusion matrix
    plt.imshow(confusion_matrix, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.xticks([0, 1], ['Negative', 'Positive'])
    plt.yticks([0, 1], ['Negative', 'Positive'])
    plt.colorbar()
    plt.show()   