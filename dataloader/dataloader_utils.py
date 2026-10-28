from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
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


class ModalitySubset(Subset):
    """Subset whose items carry only `modalities` (the other modalities of the run are returned as missing):
    train / val / test loaders over the same dataset can use different modalities. Each loader (and each of
    its workers) keeps its own modalities, so nothing is switched on the shared dataset."""
    def __init__(self, dataset, indices, modalities):
        super().__init__(dataset, indices)
        self.modalities = list(modalities)

    def __getitem__(self, idx):
        return self.dataset.get_item(self.indices[idx], self.modalities)

    def __getitems__(self, indices):
        # newer torch versions call __getitems__ (batched fetch) instead of __getitem__ on a Subset
        return [self.__getitem__(idx) for idx in indices]


def report_patients_not_in_dataset(partition, requested, kept, dataset, modalities, n_show=5):
    """Split patients left out of a loader. Printed only for patients with none of the modalities of this
    partition (a patient with at least one of them is used); patients without a valid OS label (not expected
    with the current label files and splits) are reported on their own line."""
    missing = set(requested) - set(kept)
    in_dataset = set(dataset.patient_df.index)
    no_modality = sorted(missing & (dataset.patients_without_input_modalities | in_dataset))
    no_label = sorted(missing - dataset.patients_without_input_modalities - in_dataset)
    if no_modality:
        print(f"{len(no_modality)}/{len(requested)} {partition} patients have none of the {partition} modalities "
              f"{list(modalities)}: left out, e.g. {no_modality[:n_show]}")
    if no_label:
        print(f"[❗] {len(no_label)}/{len(requested)} {partition} patients of the split have no valid OS label: "
              f"left out, e.g. {no_label[:n_show]}")


def partition_modalities(config, dataset):
    """{train, val, test} modalities of the run (resolved by main.py into config.data_loader.modalities;
    test is a list of modality sets). Without it every partition uses all the dataset modalities."""
    modalities = config.data_loader.get('modalities')
    if modalities is None:
        return {'train': dataset.input_modalities, 'val': dataset.input_modalities, 'test': [dataset.input_modalities]}
    return modalities


SAMPLERS = ("uniform", "cancer_balanced", "cancer_sqrt")


def cancer_sampler(dataset, patients, mode, generator):
    """Training sampler (data_loader.sampler): uniform -> None (shuffle, every patient once per epoch);
    cancer_balanced -> every cancer type drawn with the same probability; cancer_sqrt -> probability of a cancer
    type proportional to the square root of its size (in between). Both draw len(patients) patients per epoch
    with replacement (WeightedRandomSampler)."""
    if mode not in SAMPLERS:
        raise ValueError(f"data_loader.sampler {mode}: one of {list(SAMPLERS)}")
    if mode == "uniform":
        return None
    cancer = dataset.patient_df.loc[patients, "dataset_name"].map(dataset.cancer_type).to_numpy()
    _, inverse, counts = np.unique(cancer, return_inverse=True, return_counts=True)
    per_patient = counts[inverse].astype(np.float64)
    weights = 1.0 / per_patient if mode == "cancer_balanced" else 1.0 / np.sqrt(per_patient)
    return WeightedRandomSampler(weights.tolist(), num_samples=len(patients), replacement=True, generator=generator)


def make_dataloader(dataset, partition, patients, modalities, config, train=False):
    """DataLoader of the given patients (None -> None) with only `modalities`; patients not in the dataset,
    or with none of `modalities`, are left out. Training: shuffled (seeded by config.seed), full batches only.
    Validation / test: in order, batch of one patient when test_sample is False (all patches of a patient,
    variable size)."""
    if patients is None:
        return None
    # all the loaders below use batches of one patient only when batch_size == 1; OXA_MISS(_final) never reads
    # patch_features of a missing WSI (other models are not checked, they keep the full-size zeros)
    dataset.compact_missing_wsi = config.data_loader.batch_size == 1 and str(dataset.model_name).startswith('OXA_MISS')
    num_workers = config.data_loader.num_workers
    eval_batch_size = 1 if config.data_loader.test_sample == False else config.data_loader.batch_size
    requested = patients
    patients = patients[np.isin(patients, dataset.patient_df.index)]
    patients = dataset.patients_with_modalities(patients, modalities)
    report_patients_not_in_dataset(partition, requested, patients, dataset, modalities)
    if len(patients) == 0:
        raise ValueError(f"No {partition} patients with any of {list(modalities)} (of {len(requested)} in the split): "
                         f"none of the datasets {list(dataset.datasets)} has these modalities?")
    generator = None
    sampler = None
    if train:
        # training order (shuffle / sampler) from the run's seed: different seeds see different orders
        generator = Generator()
        generator.manual_seed(int(config.get('seed', 42)))
        sampler = cancer_sampler(dataset, patients, config.data_loader.get('sampler', 'uniform'), generator)
    return DataLoader(ModalitySubset(dataset, patients, modalities),
                      batch_size=config.data_loader.batch_size if train else eval_batch_size,
                      shuffle=train and sampler is None,
                      sampler=sampler,
                      generator=generator,
                      drop_last=train,
                      pin_memory=True,
                      num_workers=num_workers,
                      prefetch_factor=4 if num_workers > 0 else None,
                      # not for the training loader: its workers are forked at each iteration, so they see the current
                      # set_sample flag (all patches for the Cox risk bank / Breslow passes, max_patches in training);
                      # persistent workers kept the flag of the first pass for the whole run
                      persistent_workers=num_workers > 0 and not train)


def get_dataloaders(dataset, train_patients, val_patients, test_patients, config, test_modalities=None):
    """(train, val, test) DataLoaders of the given patients (None -> no loader), each with the modalities of
    its partition (see partition_modalities); the test loader uses test_modalities if given, otherwise the
    first test modality set."""
    modalities = partition_modalities(config, dataset)
    return (make_dataloader(dataset, "train", train_patients, modalities['train'], config, train=True),
            make_dataloader(dataset, "val", val_patients, modalities['val'], config),
            make_dataloader(dataset, "test", test_patients, test_modalities or modalities['test'][0], config))
