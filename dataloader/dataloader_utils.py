from torch.utils.data import DataLoader, Subset
from torch import Generator
import numpy as np

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


def report_patients_not_in_dataset(partition, requested, kept, dataset, n_show=5):
    """Split patients left out of a loader. Printed only for patients with none of the input modalities of
    this run (a patient with at least one of them is used); patients without a valid OS label (not expected
    with the current label files and splits) are reported on their own line."""
    missing = set(requested) - set(kept)
    no_modality = sorted(missing & dataset.patients_without_input_modalities)
    no_label = sorted(missing - dataset.patients_without_input_modalities)
    if no_modality:
        print(f"{len(no_modality)}/{len(requested)} {partition} patients have none of the input modalities "
              f"{list(dataset.input_modalities)}: left out, e.g. {no_modality[:n_show]}")
    if no_label:
        print(f"[❗] {len(no_label)}/{len(requested)} {partition} patients of the split have no valid OS label: "
              f"left out, e.g. {no_label[:n_show]}")


def get_dataloaders(dataset, train_patients, val_patients, test_patients, config):
    """DataLoaders of the given patients (None -> no loader); patients not in the dataset are left out.
    Training: shuffled (fixed seed), full batches only. Validation / test: in order, batch of one patient
    when test_sample is False (all patches of a patient, variable size)."""
    # all the loaders below use batches of one patient only when batch_size == 1; OXA_MISS never reads
    # patch_features of a missing WSI (other models are not checked, they keep the full-size zeros)
    dataset.compact_missing_wsi = config.data_loader.batch_size == 1 and dataset.model_name == 'OXA_MISS'
    num_workers = config.data_loader.num_workers
    eval_batch_size = 1 if config.data_loader.test_sample == False else config.data_loader.batch_size

    def make_loader(partition, patients, train):
        if patients is None:
            return None
        patients = patients[np.isin(patients, dataset.patient_df.index)]
        report_patients_not_in_dataset(partition, requested_patients[partition], patients, dataset)
        generator = None
        if train:
            generator = Generator()
            generator.manual_seed(42)
        return DataLoader(Subset(dataset, patients),
                          batch_size=config.data_loader.batch_size if train else eval_batch_size,
                          shuffle=train,
                          generator=generator,
                          drop_last=train,
                          pin_memory=True,
                          num_workers=num_workers,
                          prefetch_factor=4 if num_workers > 0 else None,
                          persistent_workers=num_workers > 0)

    requested_patients = {"train": train_patients, "val": val_patients, "test": test_patients}
    return (make_loader("train", train_patients, train=True),
            make_loader("val", val_patients, train=False),
            make_loader("test", test_patients, train=False))
