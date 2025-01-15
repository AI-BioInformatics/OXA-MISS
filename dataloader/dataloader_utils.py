from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
import pandas as pd

def extract_names(f):
    """Function that extracts from the sample name:
    - patient
    - tissue
    - treatment phase
    - side (if present)
    """
    sample = ""
    tissue = ""
    treatment_phase = ""
    side = ""
    f = f.split(".")[0]
    d = f.split("_")
    sample = d[0]
    tissue = d[1]
    if tissue[-1].isdigit():
        tissue = tissue[0:-1]
    if tissue[-1] == "R" or tissue[-1] == "L":
        side = tissue[-1]
        tissue = tissue[0:-1]
    if (
        tissue[0:2] == "p2"
        or tissue[0:2] == "r1"
        or tissue[0:2] == "r2"
        or tissue[0:2] == "r3"
        or tissue[0:2] == "r4"
    ):
        treatment_phase = tissue[0:1]
        tissue = tissue[2:]

    if tissue[0] == "i" or tissue[0] == "p" or tissue[0] == "r" or tissue[0] == "o":
        treatment_phase = tissue[0]
        tissue = tissue[1:]
    if sample[-1] == "i":
        treatment_phase = sample[-1]
        sample = sample[0:-1]
    return sample, tissue, treatment_phase, side


def get_dataloaders(dataset, train_patients, val_patients, test_patients, config):
    mask = np.isin(train_patients, dataset.patient_df.index)
    # Filter the array to keep only elements in df.index
    filtered_train_patients = train_patients[mask]
    if len(filtered_train_patients) != len(train_patients):
        print("Some train patients are not in the dataset: ", set(train_patients) - set(filtered_train_patients))
    prefetch_factor = 4
    if config.data_loader.num_workers == 0:
        prefetch_factor = None
    train_dataloader = DataLoader(
                                Subset(dataset, filtered_train_patients), 
                                batch_size=config.data_loader.batch_size, 
                                shuffle=True, 
                                drop_last=True, 
                                pin_memory=True, 
                                num_workers=config.data_loader.num_workers, 
                                prefetch_factor=prefetch_factor
                            )
    if val_patients is not None:
        mask = np.isin(val_patients, dataset.patient_df.index)
        filtered_val_patients = val_patients[mask]
        if len(filtered_val_patients) != len(val_patients):
            print("Some val patients are not in the dataset: ", set(val_patients) - set(filtered_val_patients))
        batch_size = config.data_loader.batch_size
        if config.data_loader.test_sample == False:
            batch_size = 1
        val_dataloader = DataLoader(
                                        Subset(dataset, filtered_val_patients), 
                                        batch_size=batch_size,
                                        shuffle=False, 
                                        drop_last=False, 
                                        pin_memory=True, 
                                        num_workers=config.data_loader.num_workers, 
                                        prefetch_factor=prefetch_factor,
                                )
    else:
        val_dataloader = None   
    if test_patients is not None:
        mask = np.isin(test_patients, dataset.patient_df.index)
        filtered_test_patients = test_patients[mask]
        if len(filtered_test_patients) != len(test_patients):
            print("Some test patients are not in the dataset: ", set(test_patients) - set(filtered_test_patients))
        batch_size = config.data_loader.batch_size
        if config.data_loader.test_sample == False:
            batch_size = 1
        test_dataloader = DataLoader(
                                        Subset(dataset, filtered_test_patients), 
                                        batch_size=batch_size, 
                                        shuffle=False, 
                                        drop_last=False, 
                                        pin_memory=True, 
                                        num_workers=config.data_loader.num_workers, 
                                        prefetch_factor=prefetch_factor,
                                    )
    else:
        test_dataloader = None
    return train_dataloader, val_dataloader, test_dataloader