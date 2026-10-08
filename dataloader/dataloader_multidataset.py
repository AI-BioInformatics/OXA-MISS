import torch
import pandas as pd
import os
import numpy as np
from copy import deepcopy
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader, Subset
from .dataloader_utils import extract_names
from .clinical_tokens import ClinicalTokens
from .radiology_io import load_radiology_tokens
import yaml
from munch import munchify
import json
import glob
import bisect
import csv
import re

DEFAULT_CT_MAPPING = '/work/H2020DeciderFicarra/ccRCC/CT_mapping.csv'
DEFAULT_MRI_MAPPING = '/work/H2020DeciderFicarra/ccRCC/MRI_mapping.csv'
# patient id at the start of a radiology file name: TCGA (TCGA-AY-4070A.npz -> TCGA-AY-4070) or CPTAC
CASE_ID_PATTERN = re.compile(r'(TCGA-[A-Z0-9]{2}-[A-Z0-9]{4}|C3[LN]-\d{5})')
DEFAULT_RADIOLOGY_REGISTRY = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config', 'radiology.yaml')



class Multimodal_Bio_Dataset(Dataset):
    def __init__(self,  datasets_configs = ["MultimodalDecider/config/Decider_dataset.yaml"],
                        task_type="Survival", # Survival or treatment_response
                        max_patches=4096,
                        n_bins=4,
                        eps=1e-6,
                        sample=True,
                        load_slides_in_RAM=False,
                        slides_cache_dtype="float32",
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
                        radiology_encoders=None,
                        radiology_registry=DEFAULT_RADIOLOGY_REGISTRY,
                        clinical_tokens=None,
                        radiology_tokens=False,
                        ):
        self.model_name = model_name
        # radiology_tokens: CT / MRI as token matrices (N regions, dim) from the unpooled encoder outputs
        # (radiology registry unpooled_root), for a model with its own radiology encoder (OXA_MISS_final);
        # otherwise one mean-pooled vector per exam
        self.radiology_tokens = radiology_tokens
        # {CT: <encoder>, MRI: <encoder>} of config/radiology.yaml for all the datasets (None: the ct_path /
        # mri_path of each dataset yaml)
        self.radiology_encoders = dict(radiology_encoders or {})
        if self.radiology_encoders:
            with open(radiology_registry) as f:
                self.radiology_registry = yaml.safe_load(f)
            unknown = {m: e for m, e in self.radiology_encoders.items()
                       if m not in self.radiology_registry['encoders'] or e not in self.radiology_registry['encoders'][m]}
            if unknown:
                raise ValueError(f"radiology_encoders {unknown} not in {radiology_registry}: {self.radiology_registry['encoders']}")
        if radiology_tokens and not self.radiology_encoders and any(m in input_modalities for m in ('CT', 'MRI')):
            raise ValueError("radiology_tokens needs data_loader.radiology_encoders (the unpooled features are found "
                             "through config/radiology.yaml)")
        self.input_modalities = input_modalities
        self.missing_modality_test_scenarios = missing_modality_test_scenarios
        self.missing_mod_rate = missing_mod_rate
        self.use_missing_modalities_tables = use_missing_modalities_tables
        # missing modality tables: needed for the simulated missingness of training (missing_mod_rate) and for
        # the test scenarios. One per dataset (missing_modalities_table_path of the dataset yaml); a table of
        # the main config (missing_modality_table) is read first and has precedence
        need_tables = use_missing_modalities_tables or len(missing_modality_test_scenarios) > 0
        missing_table_paths = [missing_modality_table] if (need_tables and missing_modality_table) else []
        if use_missing_modalities_tables and not missing_mod_rate:
            raise ValueError("Missing modalities tables are enabled but missing_mod_rate is not set")
        # None (e.g. a model without CNV, whose config has no cnv_group_name) = no group
        self.genomics_group_name = list(genomics_group_name or [])
        self.cnv_group_name = list(cnv_group_name or [])
        self.task_type = task_type
        self.load_slides_in_RAM = load_slides_in_RAM
        self.robust_training = False
        # missing WSI: (1, dim) placeholder instead of (max_patches, dim) zeros; safe only if every batch has
        # one patient (tensors of different shapes cannot be collated) -> enabled by get_dataloaders
        self.compact_missing_wsi = False
        if self.load_slides_in_RAM:
            self.slides_cache = {}
        # dtype of the slides kept in RAM: float16 halves the cache; features go back to float32 per item
        if slides_cache_dtype not in ("float32", "float16", "bfloat16"):
            raise ValueError(f"slides_cache_dtype must be float32 / float16 / bfloat16, got {slides_cache_dtype}")
        self.slides_cache_dtype = getattr(torch, slides_cache_dtype)
            
        # gene groups loaded once for all the datasets: each dataset's genomics then keeps only its genes
        # (intersection over the datasets). Reloading them per dataset kept only the last dataset's genes,
        # so genes missing there would be NaN for the patients of the other datasets.
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
                non_positive = dataframe.loc[dataframe["time"] <= 0, "case_id"].unique()
                if len(non_positive):  # not a valid survival time
                    print(f"[❗] Dataset {config.name}: {len(non_positive)} patients with follow-up <= 0 days skipped, e.g. {list(non_positive[:5])}")
                    dataframe = dataframe[dataframe["time"] > 0]
                self._check_censorship_convention(config.name, dataframe)
                self.case_id_name = "case_id"
                self.slide_id_name = "slide_id"
            else:
                self.case_id_name = self.datasets[config.name].case_id_name
                self.slide_id_name = self.datasets[config.name].slide_id_name
            dataframe = self.filter_by_tissue_type(config.name, dataframe, config.parameters.tissue_type_filter)                
            
            if need_tables:
                if not config.parameters.get('missing_modalities_table_path'):
                    raise ValueError(f"Dataset {config.name}: no missing_modalities_table_path in its yaml "
                                     f"(needed by missing_modalities_tables / missing_modality_test)")
                missing_table_paths.append(config.parameters.missing_modalities_table_path)

            # CT/MRI/clinical sources are stored per dataset: each dataset config has its own feature folders,
            # so a single attribute would be overwritten by the last config in datasets_configs.
            # CT/MRI: <modality>_path is the features folder, <modality>_mapping_path (optional) the csv
            # case_id -> chosen_exam (file name of the patient's exam inside the folder).
            if not hasattr(self, 'ct_paths'):
                self.ct_paths, self.mri_paths, self.clinical_data_per_dataset = {}, {}, {}
                self.ct_exams, self.mri_exams = {}, {}
            for modality, paths, exams, default_mapping in [('CT', self.ct_paths, self.ct_exams, DEFAULT_CT_MAPPING),
                                                            ('MRI', self.mri_paths, self.mri_exams, DEFAULT_MRI_MAPPING)]:
                key = modality.lower()
                if modality in self.radiology_encoders:
                    # encoder of the run: the dataset's folder of that encoder, if it has one
                    folder = self._radiology_folder(modality, config.name)
                    if folder is not None:
                        paths[config.name] = folder
                        exams[config.name] = self._scan_exams(folder)
                elif hasattr(config.parameters, f'{key}_path'):
                    paths[config.name] = config.parameters[f'{key}_path']
                    exams[config.name] = self._read_exam_mapping(config.parameters.get(f'{key}_mapping_path', default_mapping))
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
            # genomics is read only if used (SurvPath needs it to select the patients)
            load_genomics = hasattr(config.parameters, 'genomics_path') and \
                ('Genomics' in self.input_modalities or self.model_name == 'SurvPath')
            if load_genomics:
                genomics_path = config.parameters.genomics_path
                if genomics_path.endswith(".tsv"):
                    sep = "\t"
                elif genomics_path.endswith(".csv"):
                    sep = ","
                else:
                    raise ValueError("Genomics file must be in .tsv or .csv format")
                # read only the patient id and the genes of the selected groups (the matrix has ~20k genes)
                with open(genomics_path, newline='') as fh:  # pd.read_csv(nrows=0) takes seconds on 20k columns
                    header = next(csv.reader(fh, delimiter=sep))
                wanted_genes = set(g for key in self.genomics_group_name for g in self.genes_groups[key]["ensg_gene_id"])
                id_cols = [c for c in header if c in ('patient', 'Unnamed: 0')]
                gene_cols = [c for c in header if c not in id_cols and c.split('.')[0] in wanted_genes]
                # explicit dtypes: pandas usecols by name is slower than a full read without them
                genomics = pd.read_csv(genomics_path, sep=sep, usecols=id_cols + gene_cols,
                                       dtype={**{c: str for c in id_cols}, **{c: 'float64' for c in gene_cols}})
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
                if load_genomics:
                    self.genomics = genomics
                if hasattr(config.parameters, 'cnv_path'):
                    self.cnv = cnv
            else:
                self.dataframe = pd.concat([self.dataframe, dataframe], ignore_index=True)
                if load_genomics:
                    self.genomics = pd.concat([self.genomics, genomics], ignore_index=False) if hasattr(self, 'genomics') else genomics
                if hasattr(config.parameters, 'cnv_path'):
                    self.cnv = pd.concat([self.cnv, cnv], ignore_index=True)
                       
        
        # clinical as tokens (data_loader.clinical: table, categorical, use_study_id, excluded_studies):
        # availability and values from the clean table of all the studies, instead of the 14 columns of the
        # dataset yamls' clinical_path
        self.clinical_tokens = ClinicalTokens(**clinical_tokens) if clinical_tokens is not None else None

        # study (dataset) and cancer type of each patient, as indices (module B adversary, probes)
        self.study_names = list(self.datasets)
        self.cancer_type_names = sorted({self.cancer_type(n) for n in self.study_names})

        self.max_patches = max_patches
        self.sample = sample
        self.n_bins = n_bins
        
        self.eps = eps
        
        self._compute_patient_dict()
        self._compute_patient_df()
        if need_tables:
            self._merge_missing_modality_tables(missing_table_paths)
        # --- patient-first: keep every patient with at least one available input modality ---
        self._compute_modality_availability()
        self._filter_patients_by_modalities()
        self._compute_sites()

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
        if self.load_slides_in_RAM and 'WSI' in self.input_modalities:
            self._preload_slides()
        wsi_info = " ({} with WSI, {} slides on disk)".format(int(self.patient_df["has_WSI"].sum()), n_slides) \
            if 'WSI' in self.input_modalities else ""
        print("Dataset loaded with {} patients{}".format(len(self.patient_df), wsi_info))


    # columns of a missing modality table: True = the modality is kept for the patient in that setting
    MISSING_TABLE_COLUMN = re.compile(r'^(complete|[a-z]+_miss_\d+|missing_all_[a-z]+_\d+)$')
    STATUS_MODALITIES = ["WSI", "Genomics", "CNV", "CT", "MRI", "Clinical"]

    def _merge_missing_modality_tables(self, paths):
        """Adds the mask columns of the missing modality tables to patient_df (by case_id). A patient missing
        from the tables keeps every modality (its columns are True); a case_id in two tables keeps the first."""
        tables = []
        for path in paths:
            table = pd.read_csv(path, dtype={'case_id': str})
            columns = [c for c in table.columns if self.MISSING_TABLE_COLUMN.match(c)]
            table = table[['case_id'] + columns].copy()
            table['case_id'] = table['case_id'].astype(str).str.strip()
            tables.append(table)
        table = pd.concat(tables, ignore_index=True).drop_duplicates('case_id', keep='first').set_index('case_id')
        columns = list(table.columns)
        merged = self.patient_df.drop(columns=[c for c in columns if c in self.patient_df.columns]).join(table, how='left')
        not_in_tables = merged[columns].isna().all(axis=1) if columns else pd.Series(False, index=merged.index)
        if not_in_tables.any():
            print(f"[❗] {int(not_in_tables.sum())} patients not in the missing modality tables: they keep every modality, "
                  f"e.g. {list(merged.index[not_in_tables][:5])}")
        merged[columns] = merged[columns].fillna(True).astype(bool)
        self.patient_df = merged
        self.missing_table_columns = set(columns)

    def _table_keeps(self, row, setting, modality):
        """False if the missing modality setting (training missing_mod_rate or test scenario) removes `modality`
        for this patient: <modality>_miss_<rate> removes only that modality, missing_all_<rate> any of them."""
        if setting in (None, '', 'complete'):
            return True
        m = modality.lower()
        if setting.startswith('missing_all_'):
            column = f"missing_all_{m}_{setting.rsplit('_', 1)[1]}"
        elif '_miss_' in setting:
            if setting.split('_miss_')[0] != m:
                return True
            column = setting
        else:
            raise ValueError(f"missing modality setting {setting}: expected complete, <modality>_miss_<rate> or missing_all_<rate>")
        if column not in self.missing_table_columns:
            raise ValueError(f"missing modality tables have no column {column} (setting {setting}, modality {modality})")
        return bool(row[column])

    def _train_keeps(self, row, modality):
        """Simulated missingness of training (missing_modalities_tables.active / missing_mod_rate)."""
        return not self.use_missing_modalities_tables or self._table_keeps(row, self.missing_mod_rate, modality)

    MIN_SITE_PATIENTS = 5
    TCGA_SITE = re.compile(r'^TCGA-([A-Z0-9]{2})-')

    def _compute_sites(self):
        """Acquisition site of each patient (module B adversary, target "site"): the TCGA tissue source site
        (TCGA-<site>-<patient> barcode); a non-TCGA dataset (e.g. CPTAC) is one site. Sites with fewer than
        MIN_SITE_PATIENTS patients are merged per cancer type into <cancer>_other_sites."""
        raw = {}
        for pid, name in zip(self.patient_df.index, self.patient_df["dataset_name"]):
            match = self.TCGA_SITE.match(str(pid))
            raw[pid] = f"TCGA-{match.group(1)}" if match else name
        counts = pd.Series(raw).value_counts()
        self.site_of = {pid: site if counts[site] >= self.MIN_SITE_PATIENTS
                        else f"{self.cancer_type(self.patient_df.loc[pid, 'dataset_name'])}_other_sites"
                        for pid, site in raw.items()}
        self.site_names = sorted(set(self.site_of.values()))
        self.site_index_of = {site: i for i, site in enumerate(self.site_names)}

    @staticmethod
    def cancer_type(dataset_name):
        """Cancer type of a dataset name: CPTAC (ccRCC) -> KIRC, TCGA_<X>[RED] -> X."""
        return "KIRC" if dataset_name == "CPTAC" else dataset_name.replace("TCGA_", "").replace("RED", "")

    @staticmethod
    def _read_exam_mapping(path):
        mapping = pd.read_csv(path, dtype={'case_id': str})
        return mapping.drop_duplicates('case_id').set_index('case_id')['chosen_exam']

    def _radiology_folder(self, modality, dataset_name):
        """<root>/<encoder folder>/<tumor folder> of config/radiology.yaml, None if the dataset has none."""
        registry = self.radiology_registry
        tumor = registry['tumor_folders'].get(dataset_name)
        if tumor is None:
            return None
        root = registry['unpooled_root'] if self.radiology_tokens else registry['root']
        folder = os.path.join(root, registry['encoders'][modality][self.radiology_encoders[modality]], tumor)
        return folder if os.path.isdir(folder) else None

    def _scan_exams(self, folder):
        """case_id -> exam file of a features folder (<case_id>.<ext>, or <case_id><suffix>.<ext>, e.g.
        TCGA-AY-4070A; .pt token matrices with radiology_tokens, else .npz mean-pooled vectors). One exam per
        patient: a patient with several files is an error (they would need a mapping csv)."""
        ext = '.pt' if self.radiology_tokens else '.npz'
        files = sorted(f for f in os.listdir(folder) if f.endswith(ext))
        exams = {}
        for f in files:
            match = CASE_ID_PATTERN.match(f)
            case = match.group(1) if match else f[:-len(ext)]
            if case in exams:
                raise ValueError(f"{folder}: several exams for {case} ({exams[case]}, {f}): use a mapping csv")
            exams[case] = f
        return pd.Series(exams, dtype=object)

    def _radiology_dim(self, modality):
        """Embedding size of the CT / MRI features of the run (read once from a file of each dataset; they
        must agree: one encoder per run)."""
        if not hasattr(self, '_radiology_dims'):
            self._radiology_dims = {}
        if modality not in self._radiology_dims:
            paths, exams = (self.ct_paths, self.ct_exams) if modality == 'CT' else (self.mri_paths, self.mri_exams)
            dims = {}
            for name, folder in paths.items():
                files = [e for e in exams[name].values if os.path.exists(os.path.join(folder, e))]
                if files and self.radiology_tokens:
                    dims[name] = int(load_radiology_tokens(os.path.join(folder, files[0])).shape[-1])
                elif files:
                    dims[name] = int(np.load(os.path.join(folder, files[0]))['arr_0'].size)
            if len(set(dims.values())) > 1:
                raise ValueError(f"{modality} features of different size across datasets (different encoders?): {dims}")
            self._radiology_dims[modality] = next(iter(dims.values())) if dims else 512
        return self._radiology_dims[modality]

    def _radiology_features(self, modality, dataset_name, case_id):
        """Features of a patient's CT / MRI exam, read from disk: with radiology_tokens the raw unpooled encoder
        output (N regions, dim), pooled by the model's trainable attention pooling; else the (1, dim) mean-pooled
        vector."""
        paths, exams = (self.ct_paths, self.ct_exams) if modality == 'CT' else (self.mri_paths, self.mri_exams)
        path = os.path.join(paths[dataset_name], exams[dataset_name][case_id])
        if self.radiology_tokens:
            return load_radiology_tokens(path)
        return torch.from_numpy(np.squeeze(np.load(path)['arr_0'], axis=(1, 3)))

    def _placeholder_shape(self, modality, dataset_name):
        """Shape of the zero features of a missing CT/MRI: the embedding size of the run's features."""
        return (1, self._radiology_dim(modality))

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
        if self.clinical_tokens is not None:
            df["has_Clinical"] = [self.clinical_tokens.has(p) for p in pids]
        else:
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
        # labelled patients left out because they have none of the input modalities (reported per split)
        self.patients_without_input_modalities = set(self.patient_df.index[~keep])
        self.patient_df = self.patient_df[keep]

    def patients_with_modalities(self, patients, modalities):
        """The patients (same order) of the dataset with at least one of `modalities` (SurvPath: WSI and
        Genomics). Used per partition: train / val / test can use different modalities of the run."""
        patients = np.asarray(patients)
        df = self.patient_df.reindex(patients)
        if self.model_name == 'SurvPath':
            keep = df["has_WSI"].fillna(False).astype(bool) & df["has_Genomics"].fillna(False).astype(bool)
        else:
            cols = [f"has_{m}" for m in modalities if f"has_{m}" in df.columns]
            keep = df[cols].fillna(False).astype(bool).any(axis=1) if cols else pd.Series(False, index=df.index)
        return patients[keep.to_numpy()]

    def _compute_patient_df(self):
        in_datasets = self.dataframe.groupby(self.case_id_name)["dataset_name"].nunique()
        if (in_datasets > 1).any():
            raise ValueError(f"Patients in more than one dataset: {list(in_datasets.index[in_datasets > 1][:10])}")
        self.patient_df = self.dataframe.drop_duplicates(subset=self.case_id_name)
        self.patient_df = self.patient_df.reset_index(drop=True)
        self.patient_df = self.patient_df.set_index(self.case_id_name, drop=False)

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
    
    def _standardize(self, data, train_patients, val_patients=None, test_patients=None):
        """Copy of data (patients x genes) z-scored with a StandardScaler fitted on the training patients,
        applied to the training, validation and test patients (the others keep the raw values)."""
        available = self.patient_df.join(data, how="inner").index
        def kept(_partition, patients):
            if patients is None:
                return None
            # patients without data for this modality are just skipped by the standardization (no message:
            # every patient in the dataset has at least one input modality, which the model uses)
            return pd.Index(patients[np.isin(patients, available)])
        train_idx, val_idx, test_idx = kept("train", train_patients), kept("val", val_patients), kept("test", test_patients)
        normalized = deepcopy(data)
        scaler = StandardScaler().fit(normalized.loc[train_idx, :])
        for idx in (train_idx, val_idx, test_idx):
            if idx is not None and len(idx) > 0:
                normalized.loc[idx, :] = scaler.transform(normalized.loc[idx, :])
        return normalized

    def normalize_genomics(self, train_patients, val_patients=None, test_patients=None):
        self.normalized_genomics = self._standardize(self.genomics, train_patients, val_patients, test_patients)
        self.genomics_arrays = self._build_group_arrays(self.normalized_genomics, self.genomics_group_name)

    def normalize_cnv(self, train_patients, val_patients=None, test_patients=None):
        self.normalized_cnv = self._standardize(self.cnv, train_patients, val_patients, test_patients)
        self.cnv_arrays = self._build_group_arrays(self.normalized_cnv, self.cnv_group_name)

    def _build_group_arrays(self, df, group_names):
        """Per gene group, a float32 matrix (patients x genes) and patient -> row, built once after the
        normalization so that __getitem__ only indexes a row (same selection as df[genes].loc[patient])."""
        rows = {pid: i for i, pid in enumerate(df.index)}
        arrays = {}
        for key in group_names:
            available_genes = [gene for gene in self.genes_groups[key]["ensg_gene_id"] if gene in df.columns]
            arrays[key] = df[available_genes].to_numpy(dtype=np.float32)
        return {'rows': rows, 'arrays': arrays}

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

    def _slide_path(self, pt_files_path, slide_id):
        wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
        if not os.path.exists(wsi_path):
            wsi_path = glob.glob(os.path.join(pt_files_path, f'{slide_id}*.pt'))[0]
        return wsi_path

    def _preload_slides(self):
        """Loads all the slides of the dataset patients in RAM, once, in the main process.
        DataLoader workers are forked from it and share these tensors (copy-on-write): filling the
        cache inside the workers instead gives one private copy per worker and per DataLoader."""
        total_bytes = 0
        for pid in self.patient_df.index:
            pt_files_path = self.datasets[self.patient_df.loc[pid, "dataset_name"]].pt_files_path
            for slide_id in self.slides_on_disk[pid]:
                if slide_id not in self.slides_cache:
                    wsi_bag = torch.load(self._slide_path(pt_files_path, slide_id), weights_only=True, map_location="cpu")
                    wsi_bag = wsi_bag.to(self.slides_cache_dtype)
                    self.slides_cache[slide_id] = wsi_bag
                    total_bytes += wsi_bag.element_size() * wsi_bag.nelement()
        print(f"Preloaded {len(self.slides_cache)} slides in RAM as {str(self.slides_cache_dtype).replace('torch.', '')} "
              f"({total_bytes / 1024**3:.1f} GB)")

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
                        wsi_bag = torch.load(self._slide_path(pt_files_path, slide_id), weights_only=True, map_location="cpu")
                        wsi_bag = wsi_bag.to(self.slides_cache_dtype)
                        self.slides_cache[slide_id] = wsi_bag
                        num_patches = wsi_bag.shape[0]
                else:
                    wsi_bag = torch.load(self._slide_path(pt_files_path, slide_id), weights_only=True, map_location="cpu") # changed to True due to python warning
                    num_patches = wsi_bag.shape[0]
                patch_features.append(wsi_bag)
                slides_str_descriptor += slide_id + "#" + str(num_patches) + "|"
            slides_str_descriptor = slides_str_descriptor[:-1]
            patch_features = torch.cat(patch_features, dim=0)
            
            if self.sample:
                max_patches = self.max_patches

                n_samples = min(patch_features.shape[0], max_patches)
                idx = np.sort(np.random.choice(patch_features.shape[0], n_samples, replace=False))
                patch_features = patch_features[idx, :].float()   # float16 cache -> float32 after sampling
                
            
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
                patch_features = patch_features.float()
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
        return self.get_item(index, self.input_modalities)

    def get_item(self, index, modalities):
        """Data of a patient with only `modalities` (a subset of the run modalities, self.input_modalities):
        the other modalities are returned as missing (zero placeholder, status False)."""
        row  = self.patient_df.loc[index]
        # every modality starts as missing, with a zero placeholder of the same shape of the real features
        ct_feats = torch.zeros(self._placeholder_shape('CT', row["dataset_name"]), dtype=torch.float32)
        mri_feats = torch.zeros(self._placeholder_shape('MRI', row["dataset_name"]), dtype=torch.float32)
        clinical_feats = torch.zeros(self.clinical_data.shape[1] if hasattr(self, 'clinical_data') else 14, dtype=torch.float32)
        ct_status = mri_status = clinical_status = False
        if isinstance(row, pd.DataFrame):
            print("⚠️ Più righe trovate con index, uso solo la prima:")
            row = row.iloc[0]
        dataset_name = row["dataset_name"]
        slide_list = self.slides_on_disk[row[self.case_id_name]]  # already filtered on disk (and by tissue type)
        # available = the patient has the modality and it is a modality of this loader; the data of an available
        # modality are always loaded, also when a simulated missingness removes it (the test scenarios use it)
        available = {
            "WSI": 'WSI' in modalities and len(slide_list) > 0,
            "Genomics": 'Genomics' in modalities and hasattr(self, 'normalized_genomics') and hasattr(self, 'GE_selected_gene_set')
                        and index in self.normalized_genomics.index,
            "CNV": 'CNV' in modalities and hasattr(self, 'normalized_cnv') and hasattr(self, 'CNV_selected_gene_set')
                   and index in self.normalized_cnv.index,
            "CT": 'CT' in modalities and bool(row["has_CT"]),
            "MRI": 'MRI' in modalities and bool(row["has_MRI"]),
            "Clinical": 'Clinical' in modalities and bool(row["has_Clinical"]),
        }
        if available["Genomics"]:
            row_i = self.genomics_arrays['rows'][index]
            genomics = {key: torch.from_numpy(self.genomics_arrays['arrays'][key][row_i].copy()) for key in self.genomics_group_name}
        else:
            genomics = {key: torch.zeros(self.genes_groups[key].get("count", 0)) for key in self.genomics_group_name}
        if available["CNV"]:
            row_i = self.cnv_arrays['rows'][index]
            cnv = {key: torch.from_numpy(self.cnv_arrays['arrays'][key][row_i].copy()) for key in self.cnv_group_name}
        else:
            cnv = {key: torch.zeros(self.genes_groups[key].get("count", 0)) for key in self.cnv_group_name}
        # availability (has_*) is computed once at init, on the data of the patient's own dataset
        if available["CT"]:
            ct_feats = self._radiology_features('CT', dataset_name, index)
        if available["MRI"]:
            mri_feats = self._radiology_features('MRI', dataset_name, index)
        clinical_inputs = {}
        if self.clinical_tokens is not None:
            # raw values (age standardized in the model); missing / not a modality of this loader -> zeros
            clinical_inputs = self.clinical_tokens.get(index) if available["Clinical"] else self.clinical_tokens.missing()
            clinical_inputs.pop("clinical_status")
        elif available["Clinical"]:
            clinical_feats = torch.tensor(self.clinical_data_per_dataset[dataset_name].loc[index].values.astype(np.float32))
        if available["WSI"]:
            patch_features, mask, slides_str_descriptor = self._load_wsi_embs_from_path(dataset_name, slide_list)
        else:
            n_placeholder = 1 if self.compact_missing_wsi else self.max_patches
            patch_features = torch.zeros((n_placeholder, self._wsi_feature_dim(dataset_name)))
            mask = torch.zeros(n_placeholder)
            slides_str_descriptor = ""

        # status = available and kept by the simulated missingness of training (missing_mod_rate; as before it
        # applies to every loader: the base evaluation is in the training setting)
        status = {m: available[m] and self._train_keeps(row, m) for m in self.STATUS_MODALITIES}
        if self.robust_training:
            if status["WSI"] and status["Genomics"]:
                # 66% chance to remove WSI or genomics
                if np.random.rand() < 0.66:
                    if np.random.rand() < 0.5:
                        status["WSI"] = False
                    else:
                        status["Genomics"] = False
        WSI_status, genomics_status, cnv_status = status["WSI"], status["Genomics"], status["CNV"]
        ct_status, mri_status, clinical_status = status["CT"], status["MRI"], status["Clinical"]

        label = row['label']
        if self.task_type == "Survival":
            censorship = row["censorship"]
            time = row["time"]
            label_names = ["time"]
        else:
            censorship = torch.tensor(0)
            time = torch.tensor(0)
            label_names = ["treatment_response"]

        # statuses of each test scenario: available and kept by the scenario (independent of the training
        # setting); read by ModelManager.adjust_status
        missing_modality_test_scenarios_dict = {
            f"{scenario}/{m}": available[m] and self._table_keeps(row, scenario, m)
            for scenario in self.missing_modality_test_scenarios for m in self.STATUS_MODALITIES}

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
                            **clinical_inputs,
                            'missing_modality_test_scenarios': missing_modality_test_scenarios_dict,
                            'study_index': self.study_names.index(dataset_name),
                            'cancer_type_index': self.cancer_type_names.index(self.cancer_type(dataset_name)),
                            'site_index': self.site_index_of[self.site_of[row[self.case_id_name]]],
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
