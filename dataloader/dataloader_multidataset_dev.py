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



class Multimodal_Bio_Dataset(Dataset):
    def __init__(self,  datasets_configs = ["/work/H2020DeciderFicarra/D2_4/Development/MultimodalDecider/config/Decider_dataset.yaml"],
                        task_type="Survival", # Survival or treatment_response
                        max_patches=4096,
                        n_bins=4,
                        eps=1e-6,
                        sample=True,
                        load_slides_in_RAM=False,
                        file_genes_group='/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/daria_mapped.json',

                        use_WSI_level_embs=False
                        ):
        self.use_WSI_level_embs = use_WSI_level_embs
        self.task_type = task_type
        self.load_slides_in_RAM = load_slides_in_RAM
        if self.load_slides_in_RAM:
            self.slides_cache = {}
            
        self.datasets = {}
        for i, dataset_config in enumerate(datasets_configs):
            config = yaml.load(open(dataset_config, "r"), yaml.FullLoader)
            config = munchify(config)
            if config.name in self.datasets:
                raise ValueError("Dataset name {} already exists".format(config.name))
            self.datasets[config.name] = config.parameters # asser config.name in datasets
                       
            dataframe = pd.read_csv(config.parameters.dataframe_path, sep="\t",dtype={'case_id': str})
            dataframe = dataframe.dropna()
            dataframe["dataset_name"] = [config.name for _ in range(len(dataframe))]
            if task_type == "Survival":
                rename_dict = { self.datasets[config.name].label_name: "time",
                                self.datasets[config.name].censorships_name: "censorship",
                                self.datasets[config.name].case_id_name: "case_id",
                                self.datasets[config.name].slide_id_name: "slide_id"} 
                dataframe.rename(columns=rename_dict, inplace=True)
                dataframe["time"] = dataframe["time"].astype(int)
                self.case_id_name = "case_id"
                self.slide_id_name = "slide_id"
            else:
                self.case_id_name = self.datasets[config.name].case_id_name
                self.slide_id_name = self.datasets[config.name].slide_id_name
            dataframe = self.filter_by_tissue_type(config.name, dataframe, config.parameters.tissue_type_filter)                
            
            # load genomics data
            # if 'genomics_path' in config.parameters:
            
            # daria_json_path = "/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/daria_mapped.json"
            # daria_json_path = "/work/H2020DeciderFicarra/D2_4/datasets/DECIDER_cohorts/Gene_expression/Expression/daria_mapped_2.json"

            if hasattr(config.parameters, 'genomics_path'):
                genomics = pd.read_csv(config.parameters.genomics_path, sep="\t").set_index("patient")
                
                with open(file_genes_group, 'r') as f:
                    self.GE_selected_genes_groups = json.load(f)
                self.GE_selected_gene_set = set()
                for key in self.GE_selected_genes_groups:
                    self.GE_selected_gene_set.update(self.GE_selected_genes_groups[key]["ensg_gene_id"])
                self.GE_group_num = len(self.GE_selected_genes_groups)

                genomics = genomics[list(self.GE_selected_gene_set)]
                print("Genomics shape: ", genomics.shape)
                genomics = np.log(genomics+0.1)

            if hasattr(config.parameters, 'cnv_path'):
                cnv = pd.read_csv(config.parameters.cnv_path, sep="\t").set_index("patient")
                with open(file_genes_group, 'r') as f:
                    self.CNV_selected_genes_groups = json.load(f)
                self.CNV_selected_gene_set = set()
                for key in self.CNV_selected_genes_groups:
                    self.CNV_selected_gene_set.update(self.CNV_selected_genes_groups[key]["ensg_gene_id"])
                self.CNV_group_num = len(self.CNV_selected_genes_groups)
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
                    self.genomics = pd.concat([self.genomics, genomics], ignore_index=True)
                if hasattr(config.parameters, 'cnv_path'):
                    self.cnv = pd.concat([self.cnv, cnv], ignore_index=True)
                       
        #{'pAdnL', 'pOvaR', 'pMes1', 'pOth', 'pTubL', 'pPer', 'pAdnR', 'pTubL1', 'pOva', 'pTubR', 'p2Ome2', 'pPer2', 'pVag', 'pLNR', 'pUte1', 
        # 'pPerR1', 'pOvaL1', 'pOvaL', 'p2Oth', 'pPer ', 'pTub', 'pOme2', 'p0Ome', 'pUte2', 'pOva2', 'pMes', 'pOme ', 'pBow', 'pOme1', 'pOth2', 
        # 'pAdnR1', 'pOth1', 'p2Ome1', 'pOme', 'p2Per1', 'pPer3', 'pOvaR1', 'pPerL ', 'pUte', 'pOme3', 'pAndL', 'pTub2', 'pPer1'}
        # self.pt_files_path = pt_files_path
        self.max_patches = max_patches
        self.sample = sample
        self.n_bins = n_bins
        # self.label_name = label_name
        # self.censorships_name = censorships_name
        self.eps = eps
        # self._filter_by_tissue_type()
        self._compute_patient_dict()
        self._compute_patient_df()
        ############################
        # maybe wrap this into a function
        # self.patient_df = self.patient_df[self.patient_df.index.isin(self.genomics.index)]
        # if hasattr(config.parameters, 'genomics_path'):
        # self.patient_df = self.patient_df[self.patient_df.index.isin(self.cnv.join(self.genomics, rsuffix="zio_", how="inner").index)]
        self.patient_list = list(self.patient_df.index)
        ############################
        if self.task_type == "Survival":
            self._compute_labels()
        else:
            self.patient_df["label"] = self.patient_df["Treatment_Response"]
        print("Dataset loaded with {} slides and {} patients".format(len(self.dataframe), len(self.patient_df)))

    def filter_by_tissue_type(self, dataset_name, dataframe, tissue_type_filter):
        if dataset_name == "Decider":
            dataframe = dataframe[dataframe[self.slide_id_name].apply(lambda x: self.get_tissue_type(x) in tissue_type_filter)]
            dataframe = dataframe.reset_index(drop=True)
        return dataframe

    def _compute_patient_dict(self):
        self.patient_list = list(self.dataframe[self.case_id_name].unique())
        self.patient_dict = {patient: list(self.dataframe[self.dataframe[self.case_id_name] == patient][self.slide_id_name]) for patient in self.patient_list}

    def _compute_patient_df(self):
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

        # train_patients_idx = pd.Index(train_patients)
        # val_patients_idx = pd.Index(val_patients)
        # test_patients_idx = pd.Index(test_patients)

        # Normalize Genomics
        # if hasattr(self, 'genomics'):
        #     self.normalized_genomics = deepcopy(self.genomics)
        #     X_train = self.normalized_genomics.loc[train_patients_idx, :]
        #     X_val = self.normalized_genomics.loc[val_patients_idx, :]
        #     X_test = self.normalized_genomics.loc[test_patients_idx, :]
        #     scaler = StandardScaler()
        #     scaler.fit(X_train)  # fit on train set
        #     # Transform entire subsets of the copied DataFrame
        #     self.normalized_genomics.loc[train_patients_idx, :] = scaler.transform(X_train)
        #     self.normalized_genomics.loc[val_patients_idx, :] = scaler.transform(X_val)
        #     self.normalized_genomics.loc[test_patients_idx, :] = scaler.transform(X_test)

        # # Normalize CNV
        # if hasattr(self, 'cnv'):
        #     self.normalized_cnv = deepcopy(self.cnv)
        #     X_train = self.normalized_cnv.loc[train_patients_idx, :]
        #     X_val = self.normalized_cnv.loc[val_patients_idx, :]
        #     X_test = self.normalized_cnv.loc[test_patients_idx, :]
        #     scaler.fit(X_train)  # fit on train set
        #     self.normalized_cnv.loc[train_patients_idx, :] = scaler.transform(X_train)
        #     self.normalized_cnv.loc[val_patients_idx, :] = scaler.transform(X_val)
        #     self.normalized_cnv.loc[test_patients_idx, :] = scaler.transform(X_test)


        # train_indices = [i for i, patient in enumerate(self.patient_list) if patient in train_patients]
        # val_indices = [i for i, patient in enumerate(self.patient_list) if patient in val_patients]
        # test_indices = [i for i, patient in enumerate(self.patient_list) if patient in test_patients]
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
        if val_patients is not None:
            self.normalized_genomics.loc[val_patients_idx, :] = scaler.transform(X_val)
        if test_patients is not None:
            self.normalized_genomics.loc[test_patients_idx, :] = scaler.transform(X_test)

    def normalize_cnv(self, train_patients, val_patients=None, test_patients=None):
        mask = np.isin(train_patients, self.patient_df.join(self.cnv, how="inner").index)
        filtered_train_patients = train_patients[mask]
        if len(filtered_train_patients) != len(train_patients):
            print("Some train patients are not in the dataset: ", set(train_patients) - set(filtered_train_patients))
        if val_patients is not None:
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
        uncensored_df = self.patient_df[self.patient_df["censorship"] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df["time"], q=self.n_bins, retbins=True, labels=False, duplicates='drop')
        q_bins[-1] = self.patient_df["time"].max() + self.eps
        q_bins[0] = self.patient_df["time"].min() - self.eps
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.patient_df["time"], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patient_df.insert(2, 'label', disc_labels.values.astype(int))
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
            def get_features(path, suffix=""):
                patch_features = []
                pt_files_path = path
                # load all slide_names corresponding for the patient
                for slide_id in slide_names:
                    if self.load_slides_in_RAM:
                        if slide_id+suffix in self.slides_cache:
                            wsi_bag = self.slides_cache[slide_id+suffix]
                        else:
                            wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
                            wsi_bag = torch.load(wsi_path, weights_only=True, map_location="cpu")
                            self.slides_cache[slide_id+suffix] = wsi_bag
                    else:
                        wsi_path = os.path.join(pt_files_path, '{}.pt'.format(slide_id))
                        wsi_bag = torch.load(wsi_path, weights_only=True, map_location="cpu") # changed to True due to python warning
                    patch_features.append(wsi_bag)
                patch_features = torch.cat(patch_features, dim=0)
                return patch_features

            patch_features = get_features(self.datasets[dataset_name].pt_files_path)
            wsi_features = []
            if self.use_WSI_level_embs:
                for path in self.datasets[dataset_name].pt_files_path_SLIDE_LEVEL:
                    wsi_features.append(get_features(path, suffix="_TITAN"))
                if wsi_features:
                    wsi_features = torch.cat(wsi_features, dim=0)

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

            return wsi_features, patch_features, mask

    def get_tissue_type(self, slide_name):
        _, tissue_type, _, _ = extract_names(slide_name)
        return tissue_type
    
    def set_sample(self, sample):
        self.sample = sample

    def __getitem__(self, index):
        # Retrieve data from the dataframe based on the index
        row  = self.patient_df.loc[index]
        WSI_status = True
        if (hasattr(self, 'normalized_genomics') and index not in self.normalized_genomics.index) or not hasattr(self, 'normalized_genomics'):           
           genomics = {key: torch.zeros(self.GE_selected_genes_groups[key]["count"]) for key in self.GE_selected_genes_groups.keys()}
           genomics_status = False
        else:
            genomics = {}
            if hasattr(self, 'GE_selected_genes_groups'):
                for key in self.GE_selected_genes_groups:
                    ensg_gene_id_list = self.GE_selected_genes_groups[key]["ensg_gene_id"]
                    genomics[key] = torch.tensor(self.normalized_genomics[ensg_gene_id_list].loc[index].values, dtype=torch.float32)
                genomics_status = True
            else:
                genomics_status = False
        if (hasattr(self, 'normalized_cnv') and index not in self.normalized_cnv.index) or not hasattr(self, 'normalized_cnv'):
            cnv = {key: torch.zeros(self.GE_selected_genes_groups[key]["count"]) for key in self.CNV_selected_genes_groups.keys()}
            cnv_status = False
        else:
            cnv = {}
            if hasattr(self, 'CNV_selected_genes_groups'):
                for key in self.CNV_selected_genes_groups:
                    ensg_gene_id_list = self.CNV_selected_genes_groups[key]["ensg_gene_id"]
                    cnv[key] = torch.tensor(self.normalized_cnv[ensg_gene_id_list].loc[index].values, dtype=torch.float32)
                cnv_status = True
            else:
                cnv_status = False
        dataset_name = row["dataset_name"]
        tissue_type_filter = self.datasets[dataset_name].tissue_type_filter
        slide_list = self.patient_dict[row[self.case_id_name]]
        
        # bad_slides = ['8005_Ome_1009677_269186_ImageActual']
        # for bad_slide in bad_slides:
        #     if bad_slide in slide_list: slide_list.remove(bad_slide) 
        
        if dataset_name == "Decider":
            slide_list = [slide for slide in slide_list if self.get_tissue_type(slide) in tissue_type_filter]
        wsi_features, patch_features, mask = self._load_wsi_embs_from_path(dataset_name, slide_list) 
        label = row['label']
        if self.task_type == "Survival":
            censorship = row["censorship"]
            time = row["time"]
            label_names = ["time"]
        else:
            censorship = torch.tensor(0)
            time = torch.tensor(0)
            label_names = ["treatment_response"]


        data = {
                'input':{   
                            'patch_features': patch_features, 
                            'mask': mask,
                            'wsi_features': wsi_features,
                            'genomics': genomics,
                            'cnv': cnv,
                            'WSI_status': WSI_status,
                            'genomics_status': genomics_status,
                            'cnv_status': cnv_status
                        }, 
                'label': label, 
                'censorship': censorship, 
                'original_event_time': time,
                'label_names': label_names,
                'patient_id': row[self.case_id_name],
                'dataset_name': dataset_name,
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
