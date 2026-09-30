import torch
import pandas as pd
import os
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset, SequentialSampler, SubsetRandomSampler
from .dataloader_utils import extract_names
import yaml
from munch import munchify, unmunchify, Munch
import json
import glob
import bisect

DEFAULT_CT_MAPPING = '/work/H2020DeciderFicarra/ccRCC/CT_mapping.csv'
DEFAULT_MRI_MAPPING = '/work/H2020DeciderFicarra/ccRCC/MRI_mapping.csv'



class Multimodal_Bio_Dataset(Dataset):
    def __init__(self,  datasets_configs = ["MultimodalDecider/config/Decider_dataset.yaml"],
                        task_type="Survival", # Survival or treatment_response
                        max_patches=4096,
                        n_bins=4,
                        eps=1e-6,
                        sample=True,
                        load_slides_in_RAM=False,
                        file_genes_group='D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/daria_mapped.json',
                        
                        genomics_group_name = ["high_refractory", "high_sensitive",  "hypoxia_pathway"],
                        cnv_group_name = ["high_refractory", "high_sensitive",  "hypoxia_pathway"],
                        use_WSI_level_embs=False,
                        use_missing_modalities_tables=False,
                        missing_mod_rate=None,

                        missing_modality_test_scenarios=[],
                        input_modalities=['WSI', 'Genomics', 'CNV','CT'],
                        missing_modality_table=None,
                        model_name=None,
                        ):
        self.model_name = model_name
        self.input_modalities = input_modalities
        self.missing_modality_test_scenarios = missing_modality_test_scenarios
        self.missing_mod_rate = missing_mod_rate
        self.use_missing_modalities_tables = use_missing_modalities_tables
        if missing_modality_table is not None:
            self.missing_modalities_table = pd.read_csv(missing_modality_table)
        if use_missing_modalities_tables and not missing_mod_rate:
            raise ValueError("Missing modalities tables are enabled but missing_mod_rate is not set")
        self.genomics_group_name = genomics_group_name
        self.cnv_group_name = cnv_group_name
        self.task_type = task_type
        self.load_slides_in_RAM = load_slides_in_RAM
        self.robust_training = False
        if self.load_slides_in_RAM:
            self.slides_cache = {}
            
        self.datasets = {}
        for i, dataset_config in enumerate(datasets_configs):
            config = yaml.load(open(dataset_config, "r"), yaml.FullLoader)
            config = munchify(config)
            if config.name in self.datasets:
                raise ValueError("Dataset name {} already exists".format(config.name))
            self.datasets[config.name] = config.parameters # asser config.name in datasets
                       
            params = self.datasets[config.name]
            dataframe = pd.read_csv(config.parameters.dataframe_path, sep="\t",dtype={params.case_id_name: str})
            cols_to_keep = ['case_id','slide_id', 'FUT', 'Survival', 'True_Label',"Treatment_Response"] # keep only relevant columns for now, rename later
            dataframe= dataframe[dataframe.columns.intersection(cols_to_keep)]
            # The label file lists patients: one row per slide, or a single row with an empty slide_id
            # when the patient has no WSI. Only the case_id and the label columns are required.
            required = [params.case_id_name, params.label_name]
            if task_type == "Survival":
                required.append(params.event_name if 'event_name' in params else params.censorships_name)
            missing_cols = [c for c in required if c not in dataframe.columns]
            if missing_cols:
                raise ValueError(f"Dataset {config.name}: columns {missing_cols} not found in {config.parameters.dataframe_path}")
            n_rows = len(dataframe)
            dataframe = dataframe.dropna(subset=required)
            if len(dataframe) < n_rows:
                print(f"[❗] Dataset {config.name}: dropped {n_rows - len(dataframe)} label rows with missing {required}")
            if params.slide_id_name not in dataframe.columns:
                dataframe[params.slide_id_name] = ""
            dataframe[params.slide_id_name] = dataframe[params.slide_id_name].fillna("").astype(str).str.strip()
            dataframe["dataset_name"] = [config.name for _ in range(len(dataframe))]
            if task_type == "Survival":
                # The code works with "censorship" (1 = censored/alive, 0 = event/dead).
                # A label file can provide it directly (censorships_name) or as an event
                # indicator (event_name: 1 = dead), which is converted here.
                rename_dict = { params.label_name: "time",
                                params.case_id_name: "case_id",
                                params.slide_id_name: "slide_id"}
                if 'event_name' in params:
                    if 'censorships_name' in params:
                        raise ValueError(f"Dataset {config.name}: set only one of 'event_name' and 'censorships_name'")
                    dataframe["censorship"] = 1 - dataframe[params.event_name].astype(int)
                else:
                    rename_dict[params.censorships_name] = "censorship"
                dataframe.rename(columns=rename_dict, inplace=True)
                dataframe["time"] = dataframe["time"].astype(int)
                self._check_censorship_convention(config.name, dataframe)
                self.case_id_name = "case_id"
                self.slide_id_name = "slide_id"
            else:
                self.case_id_name = self.datasets[config.name].case_id_name
                self.slide_id_name = self.datasets[config.name].slide_id_name
            dataframe = self.filter_by_tissue_type(config.name, dataframe, config.parameters.tissue_type_filter)                
            
            with open(file_genes_group, 'r') as f:
                self.genes_groups = json.load(f)
                if self.model_name == 'ProSurv':
                    seen_genes = set()
                    for group in self.genes_groups.values():
                        unique_genes = []
                        for gene in group['ensg_gene_id']:
                            if gene not in seen_genes:
                                unique_genes.append(gene)
                                seen_genes.add(gene)
                        group['ensg_gene_id'] = unique_genes

            if use_missing_modalities_tables:
                if hasattr(config.parameters, 'missing_modalities_table_path'):
                    missing_modalities_table = pd.read_csv(config.parameters.missing_modalities_table_path)
                    if hasattr(self, 'missing_modalities_table'):
                        new_rows = missing_modalities_table[~missing_modalities_table['case_id'].isin(self.missing_modalities_table['case_id'])]
                        if not new_rows.empty:
                            self.missing_modalities_table = pd.concat([self.missing_modalities_table, new_rows], ignore_index=True)
                        self.missing_modalities_table["dataset_name"] = [config.name for _ in range(len(self.missing_modalities_table))]
                        self.missing_modalities_table.rename(columns=rename_dict, inplace=True)
                        if 'slide_id' in self.missing_modalities_table.columns:
                            self.missing_modalities_table.drop(columns=['slide_id'],inplace=True)
                        self.missing_modalities_table = self.missing_modalities_table.dropna()
                        self.missing_modalities_table['time'] = self.missing_modalities_table['time'].astype(int)
                        self.missing_modalities_table["dataset_name"] = [config.name for _ in range(len(self.missing_modalities_table))]
                        self.missing_modalities_table.rename(columns=rename_dict, inplace=True)
                                                
                    else:
                        self.missing_modalities_table = missing_modalities_table
                        self.missing_modalities_table["dataset_name"] = [config.name for _ in range(len(self.missing_modalities_table))]
                        self.missing_modalities_table.rename(columns=rename_dict, inplace=True)
                        
                        self.missing_modalities_table = self.missing_modalities_table.dropna()
                        self.missing_modalities_table['time'] = self.missing_modalities_table['time'].astype(int)
                        
                else:
                    raise ValueError("Missing modalities table path not found in dataset config file")
            
            # CT/MRI/clinical sources are stored per dataset: each dataset config has its own feature folders,
            # so a single attribute would be overwritten by the last config in datasets_configs.
            # CT/MRI: <modality>_path is the features folder, <modality>_mapping_path (optional) the csv
            # case_id -> chosen_exam (file name of the patient's exam inside the folder).
            if not hasattr(self, 'ct_paths'):
                self.ct_paths, self.mri_paths, self.clinical_data_per_dataset = {}, {}, {}
                self.ct_exams, self.mri_exams = {}, {}
            if hasattr(config.parameters,'ct_path'):
                self.ct_paths[config.name]=config.parameters.ct_path
                self.ct_exams[config.name]=self._read_exam_mapping(config.parameters.get('ct_mapping_path', DEFAULT_CT_MAPPING))

            if hasattr(config.parameters,'mri_path'):
                self.mri_paths[config.name]=config.parameters.mri_path
                self.mri_exams[config.name]=self._read_exam_mapping(config.parameters.get('mri_mapping_path', DEFAULT_MRI_MAPPING))
            if hasattr(config.parameters,'clinical_path'):
                clinical_data=pd.read_csv(config.parameters.clinical_path)
                CLINGEN_COLS = ['case_id','gender','age_diag','grade','cancer_history',
                    'ajcc_path_tumor_pt','ajcc_path_nodes_pn','ajcc_clin_metastasis_cm',
                    'ajcc_path_metastasis_pm','ajcc_path_tumor_stage','race_Asian','race_Black or African American',
                    'race_Hispanic or Latino','race_White','race_other']
                clinical_data = clinical_data[CLINGEN_COLS]
                clinical_data = clinical_data.set_index('case_id')
                self.clinical_data_per_dataset[config.name] = clinical_data
                # union over all datasets, used only to check which patients have clinical data
                if hasattr(self, 'clinical_data'):
                    clinical_data = pd.concat([self.clinical_data, clinical_data])
                    clinical_data = clinical_data[~clinical_data.index.duplicated(keep='first')]
                self.clinical_data = clinical_data
            if hasattr(config.parameters, 'genomics_path'):
                genomics_path = config.parameters.genomics_path
                if genomics_path.endswith(".tsv"):
                    genomics = pd.read_csv(genomics_path, sep="\t")
                elif genomics_path.endswith(".csv"):
                    genomics = pd.read_csv(genomics_path)
                else:
                    raise ValueError("Genomics file must be in .tsv or .csv format")
                patient_id_found = False
                for pid in ['patient', 'Unnamed: 0']:
                    if pid in genomics:
                        patient_id_found = True
                        genomics = genomics.set_index(pid)
                        break
                if not patient_id_found:
                    raise ValueError("Patient ID not found in genomics file. Please check the file format.")
                
                genomics.columns = genomics.columns.map(lambda x: x.split('.')[0])               
                self.GE_selected_gene_set = set()
                for key in self.genomics_group_name:
                    # Filter genes that are actually present in the genomics dataframe
                    self.GE_selected_gene_set.update([gene for gene in self.genes_groups[key]["ensg_gene_id"] if gene in genomics.columns])  #self.genes_groups[key]["ensg_gene_id"]
                    self.genes_groups[key]['count'] = len([gene for gene in self.genes_groups[key]["ensg_gene_id"] if gene in genomics.columns])
                    self.genes_groups[key]['ensg_gene_id'] = [gene for gene in self.genes_groups[key]["ensg_gene_id"] if gene in genomics.columns]
                                    
                self.GE_group_num = len(self.genomics_group_name)
             
                genomics = genomics[list(self.GE_selected_gene_set)]                
                genomics = genomics.loc[:, ~((genomics.columns.duplicated(keep='first')) & genomics.apply(lambda col: (col == 0).all()))]

                print("Genomics shape: ", genomics.shape)
                genomics = np.log(genomics+0.1)

            if hasattr(config.parameters, 'cnv_path'):
                cnv = pd.read_csv(config.parameters.cnv_path, sep="\t").set_index("patient")
                
                self.CNV_selected_gene_set = set()
                for key in self.cnv_group_name:
                    self.CNV_selected_gene_set.update([gene for gene in self.genes_groups[key]["ensg_gene_id"] if gene in cnv.columns])  #self.genes_groups[key]["ensg_gene_id"]

                self.CNV_group_num = len(self.cnv_group_name)
                cnv = cnv[list(self.CNV_selected_gene_set)]
                print("CNV shape: ", cnv.shape)
            
            if i==0:
                self.dataframe = dataframe
                if hasattr(config.parameters, 'genomics_path'):
                    self.genomics = genomics
                if hasattr(config.parameters, 'cnv_path'):
                    self.cnv = cnv
            else:
                self.dataframe = pd.concat([self.dataframe, dataframe], ignore_index=True)
                if hasattr(config.parameters, 'genomics_path'):
                    self.genomics = pd.concat([self.genomics, genomics], ignore_index=False)
                if hasattr(config.parameters, 'cnv_path'):
                    self.cnv = pd.concat([self.cnv, cnv], ignore_index=True)
                       
        #{'pAdnL', 'pOvaR', 'pMes1', 'pOth', 'pTubL', 'pPer', 'pAdnR', 'pTubL1', 'pOva', 'pTubR', 'p2Ome2', 'pPer2', 'pVag', 'pLNR', 'pUte1', 
        # 'pPerR1', 'pOvaL1', 'pOvaL', 'p2Oth', 'pPer ', 'pTub', 'pOme2', 'p0Ome', 'pUte2', 'pOva2', 'pMes', 'pOme ', 'pBow', 'pOme1', 'pOth2', 
        # 'pAdnR1', 'pOth1', 'p2Ome1', 'pOme', 'p2Per1', 'pPer3', 'pOvaR1', 'pPerL ', 'pUte', 'pOme3', 'pAndL', 'pTub2', 'pPer1'}
        
        self.max_patches = max_patches
        self.sample = sample
        self.n_bins = n_bins
        
        self.eps = eps
        
        self._compute_patient_dict()
        self._compute_patient_df()
        if use_missing_modalities_tables:
            patient_df_temp = self.patient_df.reset_index(drop=True)
            merged = pd.merge(
            patient_df_temp,
            self.missing_modalities_table,
            left_on=self.case_id_name,
            right_on='case_id',
            how='left', # keep every patient of the label files, also those not in the missing modalities table
            suffixes=('_x', '_y')
)

            cols_to_drop = []
            for col in merged.columns:
                if col.endswith('_x'):
                    base_col = col[:-2]
                    col_y = base_col + '_y'
                    if col_y in merged.columns:
                        if merged[col].equals(merged[col_y]):
                            merged = merged.drop(columns=[col_y])  # drop _y
                            merged = merged.rename(columns={col: base_col})  # rename _x → base
                        else:
                            temp = merged[col].combine_first(merged[col_y])
                            # sono diverse, lasciale entrambe
                            # continue
                            # Se dopo il combine_first è uguale a una delle due colonne, vuol dire che differivano solo per i NaN
                            if temp.equals(merged[col]) or temp.equals(merged[col_y]):
                                merged[base_col] = temp
                                merged = merged.drop(columns=[col, col_y])
                            else:
                                # Ci sono vere differenze → lascio entrambe
                                continue
            self.patient_df = merged       
            self.patient_df = self.patient_df.set_index(self.case_id_name, drop=False)
            
            missing_dataset_name = self.patient_df['dataset_name'].isna().sum()
            if missing_dataset_name > 0:
                print(f"[❗] Found {missing_dataset_name} rows with missing 'dataset_name'")
            most_common_name = self.patient_df['dataset_name'].dropna().mode()[0]
            self.patient_df['dataset_name'] = self.patient_df['dataset_name'].fillna(most_common_name)
            nan_counts = self.patient_df.isna().sum()
            nan_counts = nan_counts[nan_counts > 0]

            if not nan_counts.empty:
                print("[❗] Found NaN values after merge:")
                print(nan_counts)
        # --- patient-first: keep every patient with at least one available input modality ---
        self._compute_modality_availability()
        self._filter_patients_by_modalities()

        ############################
        # maybe wrap this into a function
        # self.patient_df = self.patient_df[self.patient_df.index.isin(self.genomics.index)]
        # self.patient_df = self.patient_df[self.patient_df.index.isin(self.cnv.join(self.genomics, rsuffix="zio_", how="inner").index)]
        self.patient_list = list(self.patient_df.index)
        ############################
        if self.task_type == "Survival":
            self._compute_labels()
        else:
            self.patient_df["label"] = self.patient_df["Treatment_Response"]
        
        n_slides = sum(len(self.slides_on_disk[p]) for p in self.patient_df.index)
        print("Dataset loaded with {} patients ({} with WSI, {} slides on disk)".format(
            len(self.patient_df), int(self.patient_df["has_WSI"].sum()), n_slides))


    @staticmethod
    def _read_exam_mapping(path):
        mapping = pd.read_csv(path, dtype={'case_id': str})
        return mapping.drop_duplicates('case_id').set_index('case_id')['chosen_exam']

    def _placeholder_shape(self, modality, dataset_name):
        """Shape of the zero features of a missing CT/MRI, matching the encoder of the configured features."""
        paths = self.ct_paths if modality == 'CT' else self.mri_paths
        if len(paths) == 0:
            return (1, 512)
        path = paths.get(dataset_name) or next(iter(paths.values()))
        if 'mednet' in path:
            return (1, 2048)
        return (1, 768) if modality == 'CT' else (1, 320)

    def _wsi_feature_dim(self, dataset_name):
        """Patch feature size of the WSI of a dataset (read once from one of its .pt files)."""
        if not hasattr(self, '_wsi_dims'):
            self._wsi_dims = {}
        if dataset_name not in self._wsi_dims:
            dim = None
            candidates = [dataset_name] + [d for d in self.datasets if d != dataset_name]
            for ds in candidates:
                pids = self.patient_df.index[(self.patient_df["dataset_name"] == ds) & self.patient_df["has_WSI"]]
                if len(pids) > 0:
                    dim = self._load_wsi_embs_from_path(ds, self.slides_on_disk[pids[0]][:1])[0].shape[1]
                    break
            self._wsi_dims[dataset_name] = dim if dim is not None else 1024
        return self._wsi_dims[dataset_name]

    def _check_censorship_convention(self, dataset_name, dataframe):
        """Cross-check censorship against True_Label (Alive/Dead), when the label file has it."""
        if "True_Label" not in dataframe.columns:
            return
        status = dataframe["True_Label"].astype(str).str.strip().str.lower()
        known = status.isin(["alive", "dead"])
        if not known.any():
            return
        expected = (status[known] == "alive").astype(int)
        mismatch = dataframe.loc[known, "censorship"].astype(int) != expected
        if mismatch.all():
            raise ValueError(
                f"Dataset {dataset_name}: censorship is inverted w.r.t. True_Label (Dead rows have censorship=1). "
                f"The label file stores an event indicator (1 = dead): use 'event_name' instead of "
                f"'censorships_name' in the dataset yaml.")
        if mismatch.any():
            print(f"[❗] Dataset {dataset_name}: {mismatch.sum()} rows have censorship inconsistent with True_Label")

    def filter_by_tissue_type(self, dataset_name, dataframe, tissue_type_filter):
        if dataset_name == "Decider":
            dataframe = dataframe[dataframe[self.slide_id_name].apply(lambda x: x == "" or self.get_tissue_type(x) in tissue_type_filter)]
            dataframe = dataframe.reset_index(drop=True)
        return dataframe

    def _compute_patient_dict(self):
        # patients come from the label files only; slides listed there may be empty (no WSI)
        self.dataframe[self.case_id_name] = self.dataframe[self.case_id_name].astype(str).str.strip()
        self.patient_list = list(self.dataframe[self.case_id_name].unique())
        slides = self.dataframe[self.dataframe[self.slide_id_name] != ""]
        slides_by_patient = slides.groupby(self.case_id_name)[self.slide_id_name].apply(list).to_dict()
        self.patient_dict = {patient: slides_by_patient.get(patient, []) for patient in self.patient_list}

    def _find_slides_on_disk(self):
        """patient -> .pt files (names without extension) of the patient's slides in their dataset folder.
        A slide of the label file is matched by exact name, otherwise by name prefix: a row with
        slide_id == case_id stands for all the slides of the patient, so every matching file is used."""
        pt_stems = {}
        for dataset_name, params in self.datasets.items():
            folder = params.get('pt_files_path')
            stems = []
            if folder is not None and os.path.isdir(folder):
                stems = sorted(f[:-3] for f in os.listdir(folder) if f.endswith('.pt'))
            pt_stems[dataset_name] = stems
        slides_on_disk = {}
        for pid, row in self.patient_df.iterrows():
            stems = pt_stems[row["dataset_name"]]
            found = []
            for slide in self.patient_dict.get(pid, []):
                i = bisect.bisect_left(stems, slide)
                if i < len(stems) and stems[i] == slide:
                    matches = [slide]
                else:
                    matches = []
                    while i < len(stems) and stems[i].startswith(slide):
                        matches.append(stems[i])
                        i += 1
                found.extend(m for m in matches if m not in found)
            if row["dataset_name"] == "Decider":
                tissue_type_filter = self.datasets["Decider"].tissue_type_filter
                found = [slide for slide in found if self.get_tissue_type(slide) in tissue_type_filter]
            slides_on_disk[pid] = found
        return slides_on_disk

    def _compute_modality_availability(self):
        """Adds has_<modality> columns to patient_df, checking the data of each patient's own dataset."""
        self.slides_on_disk = self._find_slides_on_disk()
        df = self.patient_df
        pids = df.index
        datasets = df["dataset_name"]
        df["has_WSI"] = [len(self.slides_on_disk[p]) > 0 for p in pids]
        df["has_Genomics"] = pids.isin(self.genomics.index) if hasattr(self, 'genomics') else False
        df["has_CNV"] = pids.isin(self.cnv.index) if hasattr(self, 'cnv') else False

        def imaging_available(paths, exams):
            out = []
            for p, ds in zip(pids, datasets):
                folder, ds_exams = paths.get(ds), exams.get(ds)
                out.append(folder is not None and p in ds_exams.index and os.path.exists(os.path.join(folder, ds_exams[p])))
            return out
        df["has_CT"] = imaging_available(self.ct_paths, self.ct_exams)
        df["has_MRI"] = imaging_available(self.mri_paths, self.mri_exams)
        df["has_Clinical"] = [ds in self.clinical_data_per_dataset and p in self.clinical_data_per_dataset[ds].index
                              for p, ds in zip(pids, datasets)]

    def _filter_patients_by_modalities(self):
        """Keeps the patients with at least one of the input modalities (SurvPath: WSI and genomics)."""
        known = [m for m in self.input_modalities if f"has_{m}" in self.patient_df.columns]
        unknown = [m for m in self.input_modalities if f"has_{m}" not in self.patient_df.columns]
        if unknown:
            print(f"[❗] Unknown input modalities, ignored for patient filtering: {unknown}")
        has = self.patient_df[[f"has_{m}" for m in known]]
        if self.model_name == 'SurvPath':
            keep = self.patient_df["has_WSI"] & self.patient_df["has_Genomics"]
        else:
            keep = has.any(axis=1)
        summary = has.groupby(self.patient_df["dataset_name"]).sum()
        summary.columns = known
        summary.insert(0, "patients", self.patient_df.groupby("dataset_name").size())
        summary["kept"] = keep.groupby(self.patient_df["dataset_name"]).sum()
        print(f"Patients with each input modality (label files), kept = at least one of {self.input_modalities}"
              + (" [SurvPath: WSI and Genomics]" if self.model_name == 'SurvPath' else "") + ":")
        print(summary.to_string())
        self.patient_df = self.patient_df[keep]

    def _compute_patient_df(self):
        # if len(list(self.dataframe[self.case_id_name].unique())) > len(list(self.missing_modalities_table[self.case_id_name].unique())):
        in_datasets = self.dataframe.groupby(self.case_id_name)["dataset_name"].nunique()
        if (in_datasets > 1).any():
            raise ValueError(f"Patients in more than one dataset: {list(in_datasets.index[in_datasets > 1][:10])}")
        self.patient_df = self.dataframe.drop_duplicates(subset=self.case_id_name)
        self.patient_df = self.patient_df.reset_index(drop=True)
        self.patient_df = self.patient_df.set_index(self.case_id_name, drop=False)
    # elif len(list(self.dataframe[self.case_id_name].unique())) <= len(list(self.missing_modalities_table[self.case_id_name].unique())):
        #     unique_missing = self.missing_modalities_table.drop_duplicates(subset=self.case_id_name)
    
        #     # 2. Creiamo una versione "leggera" del dataframe principale con le info che ci servono
        #     # Usiamo case_id (già rinominato) e dataset_name
        #     main_info = self.dataframe[[self.case_id_name, 'dataset_name']].drop_duplicates(subset=self.case_id_name)
            
        #     # 3. Facciamo il merge. Questo aggiungerà 'dataset_name' alla tabella delle modalità
        #     # Usiamo il rename preventivo per FUT/Survival se servono dopo
        #     rename_dict = {'FUT': 'time', 'Survival': 'censorship'}
        #     unique_missing = unique_missing.rename(columns=rename_dict)
    
        #     self.patient_df = pd.merge(
        #         unique_missing,
        #         main_info,
        #         on=self.case_id_name,
        #         how='left' # Mantiene tutti i pazienti di unique_missing e aggiunge dataset_name dove lo trova
        #     )
        #     # self.patient_df = self.missing_modalities_table.drop_duplicates(subset=self.case_id_name)
        #     self.patient_df = self.patient_df.reset_index(drop=True)    
        #     self.patient_df = self.patient_df.set_index(self.case_id_name, drop=False)

    def get_train_test_val_splits(self, train_size=0.7, val_size=0.15, test_size=0.15, random_state=42):
        np.random.seed(random_state)
        patients = np.array(self.patient_list)
        np.random.shuffle(patients)
        n = len(patients)
        train_end = int(n * train_size)
        val_end = int(n * (train_size + val_size))
        train_patients = patients[:train_end]
        val_patients = patients[train_end:val_end]
        test_patients = patients[val_end:]
        
        print("Train: {}, Val: {}, Test: {}".format(len(train_patients), len(val_patients), len(test_patients)))
        assert len(train_patients) + len(val_patients) + len(test_patients) == len(self.patient_list)
        return train_patients, val_patients, test_patients
    
    def normalize_genomics(self, train_patients, val_patients=None, test_patients=None):
        mask = np.isin(train_patients, self.patient_df.join(self.genomics, how="inner").index)
        filtered_train_patients = train_patients[mask]
        if len(filtered_train_patients) != len(train_patients):
            print("Some train patients are not in the dataset: ", set(train_patients) - set(filtered_train_patients))
        if val_patients is not None:
            mask = np.isin(val_patients, self.patient_df.join(self.genomics, how="inner").index)
            filtered_val_patients = val_patients[mask]
            if len(filtered_val_patients) != len(val_patients):
                print("Some val patients are not in the dataset: ", set(val_patients) - set(filtered_val_patients))
        if test_patients is not None:
            mask = np.isin(test_patients, self.patient_df.join(self.genomics, how="inner").index)
            filtered_test_patients = test_patients[mask]
            if len(filtered_test_patients) != len(test_patients):
                print("Some test patients are not in the dataset: ", set(test_patients) - set(filtered_test_patients))

        train_patients_idx = pd.Index(filtered_train_patients)
        if val_patients is not None:
            val_patients_idx = pd.Index(filtered_val_patients)
        if test_patients is not None:
            test_patients_idx = pd.Index(filtered_test_patients)

        self.normalized_genomics = deepcopy(self.genomics)
        X_train = self.normalized_genomics.loc[train_patients_idx, :]
        if val_patients is not None:
            X_val = self.normalized_genomics.loc[val_patients_idx, :]
        if test_patients is not None:
            X_test = self.normalized_genomics.loc[test_patients_idx, :]

        scaler = StandardScaler()
        scaler.fit(X_train)  # fit on train set

        # Transform entire subsets of the copied DataFrame
        self.normalized_genomics.loc[train_patients_idx, :] = scaler.transform(X_train)
        if val_patients is not None and len(val_patients) > 0:
            self.normalized_genomics.loc[val_patients_idx, :] = scaler.transform(X_val)
        if test_patients is not None:
            self.normalized_genomics.loc[test_patients_idx, :] = scaler.transform(X_test)

    def normalize_cnv(self, train_patients, val_patients=None, test_patients=None):
        mask = np.isin(train_patients, self.patient_df.join(self.cnv, how="inner").index)
        filtered_train_patients = train_patients[mask]
        if len(filtered_train_patients) != len(train_patients):
            print("Some train patients are not in the dataset: ", set(train_patients) - set(filtered_train_patients))
        if val_patients is not None and len(val_patients) > 0:
            mask = np.isin(val_patients, self.patient_df.join(self.cnv, how="inner").index)
            filtered_val_patients = val_patients[mask]
            if len(filtered_val_patients) != len(val_patients):
                print("Some val patients are not in the dataset: ", set(val_patients) - set(filtered_val_patients))
        if test_patients is not None:
            mask = np.isin(test_patients, self.patient_df.join(self.cnv, how="inner").index)
            filtered_test_patients = test_patients[mask]
            if len(filtered_test_patients) != len(test_patients):
                print("Some test patients are not in the dataset: ", set(test_patients) - set(filtered_test_patients))

        train_patients_idx = pd.Index(filtered_train_patients)
        if val_patients is not None:
            val_patients_idx = pd.Index(filtered_val_patients)
        if test_patients is not None:
            test_patients_idx = pd.Index(filtered_test_patients)

        self.normalized_cnv = deepcopy(self.cnv)
        X_train = self.normalized_cnv.loc[train_patients_idx, :]
        if val_patients is not None:
            X_val = self.normalized_cnv.loc[val_patients_idx, :]
        if test_patients is not None:
            X_test = self.normalized_cnv.loc[test_patients_idx, :]

        scaler = StandardScaler()
        scaler.fit(X_train)

        self.normalized_cnv.loc[train_patients_idx, :] = scaler.transform(X_train)
        if val_patients is not None:
            self.normalized_cnv.loc[val_patients_idx, :] = scaler.transform(X_val)
        if test_patients is not None:
            self.normalized_cnv.loc[test_patients_idx, :] = scaler.transform(X_test)



    def _compute_labels(self):
        uncensored_df = self.patient_df[(self.patient_df["censorship"] == 0)]# & (self.patient_df['complete'] == True)]
        disc_labels, q_bins = pd.qcut(uncensored_df["time"], q=self.n_bins, retbins=True, labels=False, duplicates='drop')
        q_bins[-1] = self.patient_df["time"].max() + self.eps
        q_bins[0] = self.patient_df["time"].min() - self.eps
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.patient_df["time"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        def safe_binning(series, bins):
            binned = pd.cut(series, bins=bins, labels=False, include_lowest=True)
            binned_filled = binned.copy()
            binned_filled=binned_filled.fillna(-1)  # fill NaN with -1
            
            binned_filled[series < bins[0]] = 0
            binned_filled[series > bins[-1]] = len(bins) - 2
            return binned_filled.astype(int)
        self.patient_df.insert(2, 'label', safe_binning(self.patient_df["time"], q_bins))
        self.bins = q_bins

    def _load_wsi_embs_from_path(self, dataset_name, slide_names):
            """
            Load all the patch embeddings from a list a slide IDs. 

            Args:
                - self 
                - slide_names : List
            
            Returns:
                - patch_features : torch.Tensor 
                - mask : torch.Tensor

            """
            patch_features = []
            pt_files_path = self.datasets[dataset_name].pt_files_path
            slides_str_descriptor = ""
            # load all slide_names corresponding for the patient
            for slide_id in slide_names:                
                if self.load_slides_in_RAM:
                    if slide_id in self.slides_cache:
                        wsi_bag = self.slides_cache[slide_id]
                        num_patches = wsi_bag.shape[0]
                    else:
                        wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
                        if not os.path.exists(wsi_path):
                            wsi_path=glob.glob(os.path.join(pt_files_path, f'{slide_id}*.pt'))[0]
                        wsi_bag = torch.load(wsi_path, weights_only=True, map_location="cpu")
                        self.slides_cache[slide_id] = wsi_bag
                        num_patches = wsi_bag.shape[0]
                else:
                    wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
                    if not os.path.exists(wsi_path):
                        wsi_path=glob.glob(os.path.join(pt_files_path, f'{slide_id}*.pt'))[0]
                    wsi_bag = torch.load(wsi_path, weights_only=True, map_location="cpu") # changed to True due to python warning
                    num_patches = wsi_bag.shape[0]
                patch_features.append(wsi_bag)
                slides_str_descriptor += slide_id + "#" + str(num_patches) + "|"
            slides_str_descriptor = slides_str_descriptor[:-1]
            patch_features = torch.cat(patch_features, dim=0)
            
            if self.sample:
                max_patches = self.max_patches

                n_samples = min(patch_features.shape[0], max_patches)
                idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
                patch_features = patch_features[idx, :]
                
            
                # make a mask 
                if n_samples == max_patches:
                    # sampled the max num patches, so keep all of them
                    mask = torch.zeros([max_patches])
                else:
                    # sampled fewer than max, so zero pad and add mask
                    original = patch_features.shape[0]
                    how_many_to_add = max_patches - original
                    zeros = torch.zeros([how_many_to_add, patch_features.shape[1]])
                    patch_features = torch.concat([patch_features, zeros], dim=0)
                    mask = torch.concat([torch.zeros([original]), torch.ones([how_many_to_add])])
            
            else:
                mask = torch.zeros([patch_features.shape[0]])

            return patch_features, mask, slides_str_descriptor

    def get_tissue_type(self, slide_name):
        _, tissue_type, _, _ = extract_names(slide_name)
        return tissue_type
    
    def set_sample(self, sample):
        self.sample = sample

    def set_robust_training_on(self):
        self.robust_training = True
    
    def set_robust_training_off(self):
        self.robust_training = False

    def __getitem__(self, index):
        # Retrieve data from the dataframe based on the index
        row  = self.patient_df.loc[index]
        # every modality starts as missing, with a zero placeholder of the same shape of the real features
        ct_feats = torch.zeros(self._placeholder_shape('CT', row["dataset_name"]), dtype=torch.float32)
        mri_feats = torch.zeros(self._placeholder_shape('MRI', row["dataset_name"]), dtype=torch.float32)
        clinical_feats = torch.zeros(self.clinical_data.shape[1] if hasattr(self, 'clinical_data') else 14, dtype=torch.float32)
        ct_status = mri_status = clinical_status = False
        if isinstance(row, pd.DataFrame):
            print("⚠️ Più righe trovate con index, uso solo la prima:")
            row = row.iloc[0]
        if self.use_missing_modalities_tables and \
            (('wsi' in self.missing_mod_rate and not row[self.missing_mod_rate]) or \
             (self.missing_mod_rate.startswith('missing_all_') and not row[f'missing_all_wsi_{self.missing_mod_rate.split("_")[-1]}'])):
            WSI_status = False
        else:
            WSI_status = True
        if (hasattr(self, 'normalized_genomics') and index not in self.normalized_genomics.index) or not hasattr(self, 'normalized_genomics'): # or (self.use_missing_modalities_tables and not row[self.missing_mod_rate]):          
           genomics = {key: torch.zeros(self.genes_groups[key]["count"]) for key in self.genomics_group_name}
           genomics_status = False
        else:
            
            genomics = {}
            if hasattr(self, 'GE_selected_gene_set'):
                for key in self.genomics_group_name:
                    ensg_gene_id_list = self.genes_groups[key]["ensg_gene_id"]
                    missing_genes = [gene for gene in ensg_gene_id_list if gene not in self.normalized_genomics.columns]
                    available_genes = [gene for gene in ensg_gene_id_list if gene in self.normalized_genomics.columns]
                    
                    genomics[key] = torch.tensor(self.normalized_genomics[available_genes].loc[index].values,dtype=torch.float32)
                # In questo modo genomics_status è False se in train stiamo utilizzando 
                # una condizione di missing modality simulata,
                # ma la genomica viene caricata lo stesso cosi se il paziente finisce in test puo essere 
                # utilizzata nei missing modalities scenarios
                if self.use_missing_modalities_tables and \
                    (('genomics' in self.missing_mod_rate and not row[self.missing_mod_rate]) or \
                     (self.missing_mod_rate.startswith('missing_all_') and not row[f'missing_all_genomics_{self.missing_mod_rate.split("_")[-1]}'])):
                    genomics_status = False
                else:
                    genomics_status = True
            else:
                genomics_status = False
        if (hasattr(self, 'normalized_cnv') and index not in self.normalized_cnv.index) or not hasattr(self, 'normalized_cnv'):
            cnv = {key: torch.zeros(self.genes_groups[key]["count"]) for key in self.cnv_group_name}
            cnv_status = False
        else:
            cnv = {}
            if hasattr(self, 'CNV_selected_gene_set'):
                for key in self.cnv_group_name:
                    ensg_gene_id_list = self.genes_groups[key]["ensg_gene_id"]
                    missing_genes = [gene for gene in ensg_gene_id_list if gene not in self.normalized_cnv.columns]
                    available_genes = [gene for gene in ensg_gene_id_list if gene in self.normalized_cnv.columns]

                    cnv[key] = torch.tensor(self.normalized_cnv[available_genes].loc[index].values, dtype=torch.float32)
                # cnv_status = True
                if self.use_missing_modalities_tables and \
                    (('cnv' in self.missing_mod_rate and not row[self.missing_mod_rate]) or\
                     (self.missing_mod_rate.startswith('missing_all_') and not row[f'missing_all_cnv_{self.missing_mod_rate.split("_")[-1]}'])):
                    cnv_status = False
                else:
                    cnv_status = True
            else:
                cnv_status = False
        dataset_name = row["dataset_name"]
        # availability (has_*) is computed once at init, on the data of the patient's own dataset
        if 'CT' in self.input_modalities and row["has_CT"]:
            ct_sample = os.path.join(self.ct_paths[dataset_name], self.ct_exams[dataset_name][index])
            ct_feats = torch.from_numpy(np.squeeze(np.load(ct_sample)['arr_0'], axis=(1,3)))
            ct_status = True
        if 'MRI' in self.input_modalities and row["has_MRI"]:
            mri_sample = os.path.join(self.mri_paths[dataset_name], self.mri_exams[dataset_name][index])
            mri_feats = torch.from_numpy(np.squeeze(np.load(mri_sample)['arr_0'], axis=(1,3)))
            mri_status = True
        if 'Clinical' in self.input_modalities and row["has_Clinical"]:
            clinical_feats = torch.tensor(self.clinical_data_per_dataset[dataset_name].loc[index].values.astype(np.float32))
            clinical_status = True
        if self.robust_training:
            if WSI_status and genomics_status:
                # 66% chance to remove WSI or genomics
                if np.random.rand() < 0.66:
                    if np.random.rand() < 0.5:
                        # remove WSI
                        WSI_status = False
                    else:
                        # remove genomics
                        genomics_status = False

        # if self.patient_df.loc[index].case_id== 'TCGA-BP-4341':
        #     print("DEBUG: Trovato paziente TCGA-BP-4341! Controlla")
        slide_list = self.slides_on_disk[row[self.case_id_name]]  # already filtered on disk (and by tissue type)
        if len(slide_list) == 0 or not 'WSI' in self.input_modalities:
            WSI_status = False
            patch_features = torch.zeros((self.max_patches, self._wsi_feature_dim(dataset_name)))
            mask = torch.zeros(self.max_patches)
            slides_str_descriptor = ""
        else:
            patch_features, mask, slides_str_descriptor = self._load_wsi_embs_from_path(dataset_name, slide_list)
        label = row['label']
        if self.task_type == "Survival":
            censorship = row["censorship"]
            time = row["time"]
            label_names = ["time"]
        else:
            censorship = torch.tensor(0)
            time = torch.tensor(0)
            label_names = ["treatment_response"]

        
        
        missing_modality_test_scenarios_dict = {}
        for scenario in self.missing_modality_test_scenarios:
            if '_miss_' in scenario:
                if row[scenario]:
                    missing_modality_test_scenarios_dict[scenario] = True
                else:
                    missing_modality_test_scenarios_dict[scenario] = False
            elif scenario.startswith('missing_all_'):
                rate = scenario.split('_')[-1]
                for modality in self.input_modalities:
                    modality = modality.lower()
                    scenario_modality = f'missing_all_{modality}_{rate}'
                    if row[scenario_modality]:
                        missing_modality_test_scenarios_dict[scenario_modality] = True
                    else:
                        missing_modality_test_scenarios_dict[scenario_modality] = False

        data = {
                'input':{   
                            'patch_features': patch_features, 
                            'mask': mask,
                            'genomics': genomics,
                            'cnv': cnv,
                            'WSI_status': WSI_status,
                            'genomics_status': genomics_status,
                            'cnv_status': cnv_status,
                            'ct_features': ct_feats,
                            'ct_status': ct_status,
                            'mri_features': mri_feats,
                            'mri_status': mri_status,
                            "clinical_features": clinical_feats,
                            "clinical_status": clinical_status,
                            'missing_modality_test_scenarios': missing_modality_test_scenarios_dict,
                            'label': label, 
                            'censorship': censorship,
                        }, 
                'label': label, 
                'censorship': censorship, 
                'original_event_time': time,
                'label_names': label_names,
                'patient_id': row[self.case_id_name],
                'dataset_name': dataset_name,
                'slides_str_descriptor': slides_str_descriptor,
            }
        return data

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.patient_df)

if __name__ == "__main__":
    dataset = Multimodal_Bio_Dataset()
    train_indices, val_indices, test_indices = dataset.get_train_test_val_splits()
    train_dataloader = DataLoader(Subset(dataset, train_indices), batch_size=4, shuffle=True, drop_last=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(Subset(dataset, val_indices), batch_size=4, shuffle=False, drop_last=False, pin_memory=True, num_workers=1, prefetch_factor=1)
    test_dataloader = DataLoader(Subset(dataset, test_indices), batch_size=4, shuffle=False, drop_last=False, pin_memory=True, num_workers=1, prefetch_factor=1)

    print("\nTRAIN")
    for data in train_dataloader:
        print(data["patient_id"])
        # break

    print("\nVAL")
    for data in val_dataloader:
        print(data["patient_id"])
        # break

    print("\nTEST")
    for data in test_dataloader:
        print(data["patient_id"])
        # break

    print("DONE")
