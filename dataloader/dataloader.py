import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, SequentialSampler, SubsetRandomSampler
from .dataloader_utils import extract_names

class WSI_Dataset(Dataset):
    def __init__(self,  dataframe_path="/work/H2020DeciderFicarra/D2_4/chemorefractory/MultimodalDecider/dataloader/dataset/OverallSurvival_labels_2024_08_01.csv",
                        pt_files_path="/work/H2020DeciderFicarra/D2_4/chemorefractory/features_UNI/pt_files",
                        tissue_type_filter = ["Ome"], # ["Ome", "Ova", "Per", "Tub", "Oth", "Adn", "Mes", "Ute", "Vag", "LNR", "Bow", "And"],
                        max_patches=4096,
                        n_bins=4,
                        label_name="FUT",
                        censorships_name="Survival",
                        eps=1e-6,
                        ):
        self.dataframe = pd.read_csv(dataframe_path, sep="\t",dtype={'case_id': str})        
        self.tissue_type_filter = tissue_type_filter
        #{'pAdnL', 'pOvaR', 'pMes1', 'pOth', 'pTubL', 'pPer', 'pAdnR', 'pTubL1', 'pOva', 'pTubR', 'p2Ome2', 'pPer2', 'pVag', 'pLNR', 'pUte1', 
        # 'pPerR1', 'pOvaL1', 'pOvaL', 'p2Oth', 'pPer ', 'pTub', 'pOme2', 'p0Ome', 'pUte2', 'pOva2', 'pMes', 'pOme ', 'pBow', 'pOme1', 'pOth2', 
        # 'pAdnR1', 'pOth1', 'p2Ome1', 'pOme', 'p2Per1', 'pPer3', 'pOvaR1', 'pPerL ', 'pUte', 'pOme3', 'pAndL', 'pTub2', 'pPer1'}
        self.pt_files_path = pt_files_path
        self.max_patches = max_patches
        self.sample = True
        self.n_bins = n_bins
        self.label_name = label_name
        self.censorships_name = censorships_name
        self.eps = eps
        self._filter_by_tissue_type()
        self._compute_patient_dict()
        self._compute_patient_df()
        self._compute_labels()
        print("Dataset loaded with {} slides and {} patients".format(len(self.dataframe), len(self.patient_df)))

    def _filter_by_tissue_type(self):
        self.dataframe = self.dataframe[self.dataframe["slide_id"].apply(lambda x: self.get_tissue_type(x) in self.tissue_type_filter)]
        self.dataframe = self.dataframe.reset_index(drop=True)

    def _compute_patient_dict(self):
        self.patient_list = list(self.dataframe["case_id"].unique())
        self.patient_dict = {patient: list(self.dataframe[self.dataframe["case_id"] == patient]["slide_id"]) for patient in self.patient_list}

    def _compute_patient_df(self):
        self.patient_df = self.dataframe.drop_duplicates(subset='case_id')
        self.patient_df = self.patient_df.reset_index(drop=True)    

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
        train_indices = [i for i, patient in enumerate(self.patient_list) if patient in train_patients]
        val_indices = [i for i, patient in enumerate(self.patient_list) if patient in val_patients]
        test_indices = [i for i, patient in enumerate(self.patient_list) if patient in test_patients]
        print("Train: {}, Val: {}, Test: {}".format(len(train_indices), len(val_indices), len(test_indices)))
        assert len(train_indices) + len(val_indices) + len(test_indices) == len(self.patient_list)
        return train_indices, val_indices, test_indices

    def _compute_labels(self):
        uncensored_df = self.patient_df[self.patient_df["Survival"] == 0]
        disc_labels, q_bins = pd.qcut(uncensored_df[self.label_name], q=self.n_bins, retbins=True, labels=False, duplicates='drop')
        q_bins[-1] = self.patient_df[self.label_name].max() + self.eps
        q_bins[0] = self.patient_df[self.label_name].min() - self.eps
        
        # assign patients to different bins according to their months' quantiles (on all data)
        # cut will choose bins so that the values of bins are evenly spaced. Each bin may have different frequncies
        disc_labels, q_bins = pd.cut(self.patient_df[self.label_name], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        self.patient_df.insert(2, 'label', disc_labels.values.astype(int))
        self.bins = q_bins

    def _load_wsi_embs_from_path(self, slide_names):
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
            # load all slide_names corresponding for the patient
            for slide_id in slide_names:
                wsi_path = os.path.join(self.pt_files_path, '{}.pt'.format(slide_id))
                wsi_bag = torch.load(wsi_path, weights_only=True) # changed to True due to python warning
                patch_features.append(wsi_bag)
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
                mask = torch.ones([1])

            return patch_features, mask

    def get_tissue_type(self, slide_name):
        _, tissue_type, _, _ = extract_names(slide_name)
        return tissue_type

    def __getitem__(self, index):
        # Retrieve data from the dataframe based on the index
        row  = self.patient_df.iloc[index]
        slide_list = self.patient_dict[row['case_id']]
        slide_list = [slide for slide in slide_list if self.get_tissue_type(slide) in self.tissue_type_filter]
        patch_features, mask = self._load_wsi_embs_from_path(slide_list)
        label = row['label']
        censorship = row['Survival']
        data = {
                'input':{   
                            'patch_features': patch_features, 
                            'mask': mask
                        }, 
                'label': label, 
                'censorship': censorship, 
                'original_event_time': row[self.label_name],
                'label_names': [self.label_name],
                'patient_id': row['case_id'],
            }
        return data

    def __len__(self):
        # Return the total number of samples in the dataset
        return len(self.patient_df)

if __name__ == "__main__":
    dataset = WSI_Dataset()
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
