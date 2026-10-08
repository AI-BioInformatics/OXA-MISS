import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io, torch
from PIL import Image
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from sklearn.metrics import confusion_matrix
from sksurv.metrics import concordance_index_censored
from sksurv.exceptions import NoComparablePairException
from collections import defaultdict
import logging


def km_risk_groups_plot(times_days, events, risks, title, folds=None):
    """Kaplan-Meier curves of the high / low risk patients (split at the median risk; with folds, at the
    median of each fold, since every fold has its own model and risk scale), log-rank test p-value and
    numbers at risk. Returns (figure, p-value)."""
    from lifelines.plotting import add_at_risk_counts
    df = pd.DataFrame({"years": np.asarray(times_days, dtype=float) / 365.25,
                       "event": np.asarray(events, dtype=int), "risk": np.asarray(risks, dtype=float),
                       "fold": 0 if folds is None else np.asarray(folds)})
    df["group"] = df.groupby("fold")["risk"].transform(lambda r: np.where(r > r.median(), "High risk", "Low risk"))
    fig, ax = plt.subplots(figsize=(7, 5.5), dpi=120)
    fitters, legend = [], []
    for group, color in [("Low risk", "#364C83"), ("High risk", "#C1423A")]:
        g = df[df.group == group]
        if len(g) == 0:
            continue
        legend.append(f"{group} (n={len(g)}, events={int(g.event.sum())})")
        kmf = KaplanMeierFitter(label=group)  # short label: it is also the row name of the at-risk table
        kmf.fit(g.years, event_observed=g.event)
        kmf.plot_survival_function(ax=ax, color=color, ci_alpha=0.12, linewidth=2, show_censors=True,
                                   censor_styles={"ms": 5, "marker": "|"})
        fitters.append(kmf)
    p_value = np.nan
    low, high = df[df.group == "Low risk"], df[df.group == "High risk"]
    if len(low) and len(high) and df.event.any():
        p_value = logrank_test(low.years, high.years, event_observed_A=low.event, event_observed_B=high.event).p_value
    ax.set_title(f"{title}\nlog-rank p = {p_value:.2e}" if not np.isnan(p_value) else title, fontsize=10)
    ax.set_xlabel("Time (years)")
    ax.set_ylabel("Overall survival probability")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    handles = [line for line in ax.get_lines() if not line.get_label().startswith("_")][:len(legend)]
    if handles:
        ax.legend(handles, legend, loc="lower left", fontsize=9)
    if fitters:
        add_at_risk_counts(*fitters, ax=ax, rows_to_show=["At risk"])
    fig.tight_layout()
    return fig, p_value


def survival_at_edges(logits):
    """Discrete-time hazard logits (N, K) -> survival at the K+1 bin edges (N, K+1): S(edge_0) = 1,
    S(edge_k+1) = prod_{j<=k} (1 - h_j) (the NLL model: label k = event in bin k)."""
    hazards = 1 / (1 + np.exp(-np.asarray(logits, dtype=float)))
    return np.concatenate([np.ones((len(hazards), 1)), np.cumprod(1 - hazards, axis=1)], axis=1)


def survival_at(S_edges, edges, times):
    """Survival of each patient at times (N,) or at a common grid (T,) -> (N,) or (N, T): linear
    interpolation of the curve between the bin edges (days), constant outside."""
    edges = np.asarray(edges, dtype=float)
    if np.ndim(times) == 1 and len(times) == len(S_edges):
        return np.array([np.interp(t, edges, s) for t, s in zip(times, S_edges)])
    return np.stack([np.interp(times, edges, s) for s in S_edges])


def _surv(events, times):
    from sksurv.util import Surv
    return Surv.from_arrays(event=np.asarray(events, dtype=bool), time=np.asarray(times, dtype=float))


def extended_survival_metrics(logits, censorships, times, edges, train_censorships=None, train_times=None):
    """Uno's c-index (IPCW, censoring from the training patients), integrated Brier score and D-calibration
    (Haider et al., JMLR 2020: chi-square p-value, > 0.05 = calibrated) of a test set. NaN when undefined.
    Without training patients the censoring distribution is estimated on the test set itself."""
    from sksurv.metrics import concordance_index_ipcw, integrated_brier_score
    from scipy.stats import chisquare
    events = 1 - np.asarray(censorships, dtype=int)
    times = np.asarray(times, dtype=float)
    if train_times is None:
        train_censorships, train_times = censorships, times
    train = _surv(1 - np.asarray(train_censorships, dtype=int), train_times)
    test = _surv(events, times)
    S_edges = survival_at_edges(logits)
    risk = -S_edges.sum(axis=1)
    out = {"c-index_uno": np.nan, "IBS": np.nan, "D-cal_p": np.nan, "D-cal_stat": np.nan}
    # time range where the censoring distribution of the training set is defined
    t_max = min(np.max(train_times), np.percentile(times, 95))
    try:
        out["c-index_uno"] = float(concordance_index_ipcw(train, test, risk, tau=t_max)[0])
    except Exception as e:
        logging.warning(f"Uno c-index undefined: {e}")
    try:
        grid = np.linspace(np.percentile(times, 5), t_max, 50, endpoint=False)
        out["IBS"] = float(integrated_brier_score(train, test, survival_at(S_edges, edges, grid), grid))
    except Exception as e:
        logging.warning(f"IBS undefined: {e}")
    # D-calibration: S_i(t_i) uniform on [0, 1]; a censored patient spreads its mass over [0, S_i(c_i)]
    s = np.clip(survival_at(S_edges, edges, times), 1e-12, 1.0)
    n_bins = 10
    lower = np.arange(n_bins) / n_bins
    counts = np.zeros(n_bins)
    for si, ei in zip(s, events):
        b = min(int(si * n_bins), n_bins - 1)
        if ei:
            counts[b] += 1
        else:
            counts[b] += (si - lower[b]) / si
            counts[:b] += (1 / n_bins) / si
    if len(s) >= n_bins:
        stat, p = chisquare(counts, f_exp=np.full(n_bins, counts.sum() / n_bins))
        out["D-cal_p"], out["D-cal_stat"] = float(p), float(stat)
    return out


def safe_c_index(censorships, event_times, risk_scores):
    """c-index, NaN when it is undefined (fewer than 2 patients or no events) instead of raising."""
    events = (1 - np.asarray(censorships)).astype(bool)
    if len(events) < 2 or not events.any():
        logging.warning(f"c-index undefined ({len(events)} patients, {int(events.sum())} events): set to NaN")
        return np.nan
    try:
        return concordance_index_censored(events, np.asarray(event_times), np.asarray(risk_scores), tied_tol=1e-08)[0]
    except NoComparablePairException:  # events, but none before another patient's time (e.g. a few patients)
        logging.warning(f"c-index undefined ({len(events)} patients, {int(events.sum())} events, no comparable pairs): set to NaN")
        return np.nan

import importlib.util
import sys
from pathlib import Path


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

def import_class_from_path(import_path, model_name, **kwargs):
        # model_name = self.config.model.name
        try:
            # import_path = f"MultimodalDecider/experiments/models/{model_name}.py"
            # ModelClass = import_class_from_path(import_path, model_name)

            import_path = Path(import_path).resolve()  # Risolve il percorso assoluto
            module_name = import_path.stem  # Ottiene il nome del modulo dal file

            spec = importlib.util.spec_from_file_location(module_name, str(import_path))
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                ModelClass = getattr(module, model_name)  # Restituisce la classe
            else:
                raise ImportError(f"Impossibile importare il modulo da {import_path}")
            return ModelClass#(**kwargs)
        except (ModuleNotFoundError, AttributeError) as e:
            raise ValueError(f"Error importing {model_name} class from {import_path}") from e


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
    #image.save("MultimodalDecider/cm.png")
    
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
    #image.save("MultimodalDecider/cm.png")
    
    # Return the PIL Image object to be logged later
    return image


class ResultsStore:
    def __init__(self):
        # Structure: {scenario: {model_version: [fold_results]}}
        self.results = defaultdict(lambda: defaultdict(list))
        # out-of-fold test predictions, to compute one c-index over the test patients of all folds
        self.predictions = defaultdict(lambda: defaultdict(list))
        self.current_fold = 0
        
    def add_result(self, scenario, model_version, fold_result, fold_num=None, predictions=None):
        """Store results for a single fold (predictions: test DataFrame with all_risk_scores,
        all_censorships, all_original_event_times (days), dataset_name)"""
        if fold_num is not None:
            self.current_fold = fold_num
        self.results[scenario][model_version].append({**fold_result, "fold": self.current_fold})
        if predictions is not None:
            self.predictions[scenario][model_version].append(predictions.assign(fold=self.current_fold))

    def pooled_c_index(self, scenario, model_version):
        """c-index of the out-of-fold predictions of all folds together (every patient is in one test fold):
        more reliable than the mean over folds when a subset has few events per fold (e.g. MRI patients),
        also per dataset. Risks of different folds come from different models."""
        folds = self.predictions[scenario][model_version]
        if not folds:
            return {}
        df = pd.concat(folds, ignore_index=True)
        pooled = {"c-index_pooled": safe_c_index(df["all_censorships"], df["all_original_event_times"], df["all_risk_scores"])}
        if df["dataset_name"].nunique() > 1:
            for dataset, d in df.groupby("dataset_name"):
                pooled[f"{dataset}_c-index_pooled"] = safe_c_index(d["all_censorships"], d["all_original_event_times"], d["all_risk_scores"])
            # pooled - mean within-cohort c-index: > 0 = part of the pooled ranking comes from differences
            # BETWEEN cohorts (e.g. the model recognizing the cohort), not from ranking patients within each
            within = [v for k, v in pooled.items() if k != "c-index_pooled" and not np.isnan(v)]
            pooled["cohort_gap"] = pooled["c-index_pooled"] - np.mean(within) if within else np.nan
        return pooled
        
    def fold_table(self, scenario):
        """One row per (model version, fold) with the overall scalar test metrics of that fold
        (the <dataset>_* metrics are left out: they are in per_dataset_table)."""
        datasets = {d for folds in self.predictions[scenario].values() for p in folds for d in p["dataset_name"].unique()}
        rows = []
        for model_version, fold_results in self.results[scenario].items():
            for r in fold_results:
                row = {"model_version": model_version, "fold": r.get("fold")}
                row.update({k: v for k, v in r.items() if k != "fold" and np.isscalar(v)
                            and not any(k.startswith(f"{d}_") for d in datasets)})
                rows.append(row)
        return pd.DataFrame(rows)

    def per_dataset_table(self, scenario, model_version):
        """Long table, one row per (dataset, fold) plus one 'pooled' row per dataset, from the out-of-fold
        test predictions: patients, events, c-index. Stays compact with many datasets (pan-cancer)."""
        folds = self.predictions[scenario][model_version]
        if not folds:
            return pd.DataFrame()
        df = pd.concat(folds, ignore_index=True)
        rows = []
        for dataset, d in df.groupby("dataset_name"):
            for fold, g in list(d.groupby("fold")) + [("pooled", d)]:
                rows.append({"dataset": dataset, "fold": str(fold), "patients": len(g),
                             "events": int((g["all_censorships"] == 0).sum()),
                             "c-index": safe_c_index(g["all_censorships"], g["all_original_event_times"], g["all_risk_scores"])})
        return pd.DataFrame(rows)

    def compute_aggregated_metrics(self, scenario, task_type="Survival"):
        """Calculate aggregated metrics for all model versions in a scenario"""
        aggregated = {}
        
        for model_version, fold_results in self.results[scenario].items():
            if task_type == "Survival":
                c_indices = [r["c-index"] for r in fold_results]
                metrics = {
                    "c-index_mean": np.nanmean(c_indices),  # nan: fold without events
                    "c-index_std": np.nanstd(c_indices),
                    "c-index_list": c_indices,
                    **self.pooled_c_index(scenario, model_version),
                }
                for key in ("c-index_uno", "IBS", "D-cal_p", "D-cal_stat"):   # D-cal_stat: chi-square, lower = better calibrated
                    if all(key in r for r in fold_results):
                        values = [r[key] for r in fold_results]
                        metrics[f"{key}_mean"], metrics[f"{key}_std"] = np.nanmean(values), np.nanstd(values)
                if all("n_patients" in r for r in fold_results):
                    metrics["n_patients"] = int(sum(r["n_patients"] for r in fold_results))
                    metrics["n_events"] = int(sum(r["n_events"] for r in fold_results))
            
            elif task_type == "Treatment_Response":
                aucs = [r["AUC"] for r in fold_results]
                f1s = [r["F1-Score"] for r in fold_results]
                accs = [r["Accuracy"] for r in fold_results]
                
                metrics = {                    
                    "AUC_mean": np.mean(aucs),
                    "F1-Score_mean": np.mean(f1s), 
                    "Accuracy_mean": np.mean(accs),
                    
                    "AUC_std": np.std(aucs),
                    "F1-Score_std": np.std(f1s),
                    "Accuracy_std": np.std(accs),
                    
                    "AUC_list": aucs,
                    "F1-Score_list": f1s,
                    "Accuracy_list": accs
                }
                
                # if "Confusion_Matrix" in fold_results[-1]:
                #     metrics["Confusion_Matrix"] = fold_results[-1]["Confusion_Matrix"]
                    
                metrics["Mean_F1-Score_AUC"] = (metrics["F1-Score_mean"] + metrics["AUC_mean"]) / 2
            
            # metrics["model_version"] = model_version
            aggregated[model_version] = metrics
            
        return aggregated
