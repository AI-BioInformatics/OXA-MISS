
---

# OXA-MISS: A Novel Multimodal Model for Chemotherapy Response Prediction Handling Missing Modalities

OXA-MISS is a novel multimodal model for chemotherapy response prediction, designed to handle missing modalities. In the task of chemotherapy response prediction in ovarian cancer, OXA-MISS achieves a 20% absolute improvement in AUC over state-of-the-art models when trained on scarce and incomplete WSI–transcriptomics datasets.

This repository contains the official code for OXA-MISS, as presented at the COMPAYL - Computational Pathology and multimodal data workshop at MICCAI 2025.

**Link to Paper:** [https://openreview.net/forum?id=Hc1d53Oo28]

## Abstract

In clinical oncology, tumor heterogeneity, data scarcity, and missing modalities are pervasive issues that significantly hinder the effectiveness of predictive models. Although multimodal integration of Whole Slide Imaging (WSI) and molecular data has shown promise in predicting overall survival (OS), current approaches often struggle when dealing with scarce and incomplete multimodal datasets, a scenario that reflects the norm rather than the exception in real-world clinical practice, especially in tasks like chemotherapy resistance prediction, where data collection is substantially more challenging than for OS. Accurately identifying patients who will not respond to chemotherapy is a critical clinical need, enabling the timely redirection to alternative therapeutic strategies and avoiding unnecessary toxicity. Hence, this paper introduces OXA-MISS, a novel multimodal model for chemotherapy response prediction designed to handle missing modalities. In the task of chemotherapy response prediction in ovarian cancer, OXA-MISS achieves a 20% absolute improvement in AUC over state-of-the-art models when trained on scarce and incomplete WSI–transcriptomics datasets. To evaluate its generalizability, we benchmarked OXA-MISS on OS prediction across three TCGA cancer types under both complete and missing-modality conditions. In these settings, the results demonstrate thatOXA-MISS achieves performance comparable to that of state-of-the-art models. In conclusion, the proposed OXA-MISS is shown to be effective in OS prediction tasks, while substantially improving predictive accuracy in realistic clinical settings, such as the proposed prediction of chemotherapy response.

## Installation

To set up the environment required to run OXA-MISS, please follow these steps:

1.  **Create a new Conda environment:**
    ```bash
    conda create -n oxamiss python=3.10.14
    ```

2.  **Activate the environment:**
    ```bash
    conda activate oxamiss
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Demo

We provide a demo code base to showcase how to execute a k-fold training (or test) of OXA-MISS on the Overall Survival (OS) prediction task using a small set of data from the TCGA-KIRC dataset.

### Demo Folder Structure

The `demo_training/` directory contains all the configuration files and data needed to run the demo.

*   `config_demo.yaml`: The main configuration file. It defines all parameters for the experiment, including:
    *   **Data paths:** Specifies where to find the dataset configuration files.
    *   **DataLoader settings:** Controls how data is loaded, including missing modality handling (`missing_mod_rate`), batch size, and the number of bins for survival analysis.
    *   **Model configuration:** Defines the OXA-MISS architecture, such as input modalities (`WSI`, `Genomics`), layer dimensions, and dropout rates.
    *   **Training settings:** Sets the loss function (`NLLSurvLoss`), optimizer (`RAdam`), learning rate scheduler, and the number of epochs.
*   `TCGA_KIRC_dataset_demo_training.yaml`: A dataset-specific configuration file for TCGA-KIRC. It specifies the paths for:
    *   The labels file (`TCGA_KIRC_labels.csv`).
    *   The gene expression data (`fpkm_unstranded.csv`).
    *   The pre-extracted WSI features (`pt_files`).
*   `KIRC_missing_modality_table.csv`: A table that defines which data modalities (WSI or genomics) have been removed for each patient in the various missing data test scenarios (e.g., `wsi_miss_25`, `genomics_miss_50`, etc.).
*   `splits/`: This folder contains the CSV files that define the data splits into training, validation, and test sets for K-fold cross-validation, ensuring the reproducibility of the experiments.
*   `data/`: This directory should contain:
    *   **Genomic Data:** A CSV file with gene expression data.
    *   **WSI Features:** Patch features extracted from the WSIs. For feature extraction, we recommend using pre-trained models like **UNI** from Mahmood Lab.

### Running the Training Demo

To start the training process for the OXA-MISS model using the demo configuration, run the following command from the project's root directory:

```bash
python main.py --config demo_training/config_demo.yaml --verbose --debug --demo_training
```

### Running the Inference/Test Demo

To run inference or testing using a pre-trained model, you can adapt the command as follows:

```bash
python main.py --config demo_test/config_demo.yaml --verbose --debug --demo_test
```

**Argument Explanations:**
*   `--config`: Specifies the path to the main configuration file.
*   `--verbose`: Enables more detailed logging during execution.
*   `--debug`: Runs the script in debug mode.
*   `--demo_training`: A flag to indicate that the training demo is being run.
*   `--demo_test`: A flag to indicate that the test demo is being run.


---

### Demo Results

Upon completion of either the training or test demo, the script will print a summary of the model's performance directly to the terminal. This provides immediate feedback on the k-fold cross-validation run.

The output is formatted as a table, similar to the example below:


```
Demo results:

      ID model_name dataset_name                   model_version modality_setting     test_scenario  c-index_mean  c-index_std   c-index_list
8qxvy6t3   OXA_MISS         KIRC    Lowest_Validation_Loss_Model         complete          complete         0.327        0.244 [0.571, 0.083]
8qxvy6t3   OXA_MISS         KIRC Highest_Validation_Metric_Model         complete          complete         0.214        0.214   [0.429, 0.0]
8qxvy6t3   OXA_MISS         KIRC                Last_Epoch_Model         complete          complete         0.327        0.244 [0.571, 0.083]
8qxvy6t3   OXA_MISS         KIRC    Lowest_Validation_Loss_Model         complete      wsi_miss_100         0.298        0.131 [0.429, 0.167]
8qxvy6t3   OXA_MISS         KIRC Highest_Validation_Metric_Model         complete      wsi_miss_100         0.298        0.131 [0.429, 0.167]
8qxvy6t3   OXA_MISS         KIRC                Last_Epoch_Model         complete      wsi_miss_100         0.298        0.131 [0.429, 0.167]
8qxvy6t3   OXA_MISS         KIRC    Lowest_Validation_Loss_Model         complete genomics_miss_100         0.536        0.036   [0.571, 0.5]
8qxvy6t3   OXA_MISS         KIRC Highest_Validation_Metric_Model         complete genomics_miss_100         0.536        0.036   [0.571, 0.5]
8qxvy6t3   OXA_MISS         KIRC                Last_Epoch_Model         complete genomics_miss_100         0.536        0.036   [0.571, 0.5]

```

#### Understanding the Output Table
-   **ID**: The unique identifier for the run, corresponding to the Weights & Biases (wandb) run ID if used.
-   **model\_version**: The results are reported for three different model checkpoints saved during training:
    -   `Lowest_Validation_Loss_Model`: The model that achieved the lowest validation loss (this is the checkpoint we used for the results reported in our paper).
    -   `Highest_Validation_Metric_Model`: The model that achieved the best validation score. The metric is the C-index for OS prediction tasks and the mean of AUC and F1-score for chemotherapy response prediction.
    -   `Last_Epoch_Model`: The model saved at the very last training epoch.
-   **modality\_setting (Training Scenario)**: This column indicates the data scenario used for *training* the model. In the demo, this is set to `complete`, meaning all patients have all data modalities available during training, as configured in `config_demo.yaml`:
    ```yaml
    data_loader:
      missing_modalities_tables:
        active: True
        missing_mod_rate: complete
    ```
-   **test\_scenario (Testing Scenario)**: This column shows the scenario applied during the *testing* phase. The total scenarios are composed of the training scenario (tested on itself) plus any additional scenarios specified in the config. In the demo, we test on the `complete` scenario, `wsi_miss_100` (unimodal test on genomics, simulating 100% missing WSI), and `genomics_miss_100` (unimodal test on WSI, simulating 100% missing genomics):
    ```yaml
    missing_modality_test:
      active: True
      scenarios: [
        'wsi_miss_100',
        'genomics_miss_100',
      ]
    ```
-   **c-index\_mean**, **c-index\_std**, **c-index\_list**: The mean, standard deviation, and list of the Concordance Index scores across the different folds of the cross-validation.

#### Additional Outputs
-   **CSV Results**: The results table is also saved as a CSV file in the project directory at `experiments/test_results_csv/Surv_OXA_MISS.csv`.
-   **Wandb Integration**: If you log in to your Weights & Biases account, all relevant training metrics (loss, C-index, etc.) will be automatically logged, allowing you to visualize the training evolution and compare runs. For the test demo, performance results are also logged.


## Citation

If you use OXA-MISS in your work, please cite our paper:

```
@inproceedings{
miccolis2025oxamiss,
title={{OXA}-{MISS}: A Robust Multimodal Architecture for Chemotherapy Response Prediction under Data Scarcity},
author={Francesca Miccolis and Fabio Marinelli and Vittorio Pipoli and Daria Afenteva and Anni Virtanen and Marta Lovino and ELISA FICARRA},
booktitle={MICCAI Workshop on Computational Pathology with Multimodal Data (COMPAYL)},
year={2025},
url={https://openreview.net/forum?id=Hc1d53Oo28}
}
```
