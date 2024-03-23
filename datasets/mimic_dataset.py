import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from lightning.pytorch.core.datamodule import LightningDataModule
import numpy as np


gender_map = {"M": 0, "F": 1}
age_map = {"40-60": 2, "80+": 4, "60-80": 3, "0-20": 0, "20-40": 1}
race_map = {
    "WHITE": 0,
    "BLACK/AFRICAN AMERICAN": 1,
    "ASIAN": 2,
    "HISPANIC/LATINO": 3,
    "AMERICAN INDIAN/ALASKA NATIVE": 4,
    "OTHER": 5,
}
num_groups_per_attrb = [2, 5, 6]

protected_group_to_index = {"gender": 0, "age_decile": 1, "race": 2}
index_to_protected_group = {0: "gender", 1: "age_decile", 2: "race"}

protected_group_name_2_map = {
    "gender": gender_map,
    "age_decile": age_map,
    "race": race_map,
}

if torch.cuda.is_available():
    device = "cuda"
# elif torch.backends.mps.is_available():
#     device = "mps"
else:
    device = "cpu"
device = torch.device(device)


gender_labels = ['M', 'F']
age_labels = ['0-20', '20-40', '40-60', '60-80', '80+']
race_labels = ['WHITE', 'BLACK/AFRICAN AMERICAN', 'ASIAN',
               'HISPANIC/LATINO', 'AMERICAN INDIAN/ALASKA NATIVE', 'OTHER']
group_labels = [gender_labels, age_labels, race_labels]
ATTRB_LABELS = ['Gender', 'Age', 'Race']
disease_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum', 'Fracture',
                  'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion', 'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']
NO_FINDING_INDEX = 8


class MIMICEmbeddingDataset(Dataset):
    def __init__(self, data_path, split, subset_ratio=1.0):
        # Load your dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data['split'] == split]
        if subset_ratio != 1.0:
            self.data = self.data.sample(
                frac=subset_ratio, random_state=1, replace=False)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        demographic_data = np.array(
            [
                gender_map[sample["gender"]],
                age_map[sample["age_decile"]],
                race_map[sample["race"]],
            ]
        )
        # If the processed data has been made by the tfrecode data, fix the path to use np instead:
        sample_path = sample["path"]
        if '.tf' in sample_path:
            emb = np.load(sample["path"].split(".tf")[0] + ".npy", allow_pickle=True)
        else:
            emb = np.load(sample["path"], allow_pickle=True)
        label = np.array([sample[d] for d in disease_labels])
        return (
            torch.from_numpy(emb).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(demographic_data).long(),
        )



class MIMICEmbeddingModule(LightningDataModule):
    def __init__(self, data_csv, batch_size, num_workers):
        super().__init__()
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.test_set = MIMICEmbeddingDataset(self.data_csv, split="test")
        self.train_set = MIMICEmbeddingDataset(self.data_csv, split="train")
        self.val_set = MIMICEmbeddingDataset(self.data_csv, split="validate")

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return [
            DataLoader(
                self.val_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
            DataLoader(
                self.test_set,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=True,
            ),
        ]


class SBSMIMICEmbeddingModule(LightningDataModule):
    def __init__(self, data_csv, batch_size, num_workers, protected_attrb):
        super().__init__()
        self.protected_attrb = protected_attrb
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.test_set = MIMICEmbeddingDataset(self.data_csv, split="test")
        self.train_set = MIMICEmbeddingDataset(self.data_csv, split="train")
        self.val_set = MIMICEmbeddingDataset(self.data_csv, split="validate")

    def train_dataloader(self):
        data_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def test_dataloader(self):
        data_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader

    def val_dataloader(self):
        data_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def make_weights_for_balanced_classes(self, data_csv, data_loader):
        data = pd.read_csv(data_csv)
        protected_group_weights = {}
        sum_weights = 0
        protected_group_map = protected_group_name_2_map[self.protected_attrb]
        for k in protected_group_map.keys():
            num_group = data[data[self.protected_attrb] == k].shape[0]
            weight_group = 1.0 / (num_group + 0.01)
            protected_group_weights[protected_group_map[k]] = weight_group
            sum_weights += weight_group

        # This is not strictly necessary, but makes for a more easy to read weight ratio
        for k in protected_group_map.keys():
            protected_group_weights[protected_group_map[k]] /= sum_weights

        weight = torch.zeros(data.shape[0]).to(device)
        protected_group_idx = protected_group_to_index[self.protected_attrb]
        for i, (emb, _, demographic_data) in enumerate(data_loader):
            idxs = torch.arange(0, emb.shape[0]) + (i * self.batch_size)
            idxs = idxs.to(dtype=torch.long, device=device)
            for j, idx in enumerate(idxs):
                weight[idx] = protected_group_weights[
                    demographic_data[j, protected_group_idx].item()
                ]
        return weight


class MIMICProtectedGroupDataModule(LightningDataModule):
    def __init__(
        self, data_csv, batch_size, num_workers, protected_attrb, protected_attrb_val
    ):
        super().__init__()
        self.protected_attrb = protected_attrb
        self.protected_atrb_val = protected_attrb_val
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.test_set = MIMICProtectedGroupDataset(
            self.data_csv,
            split="test",
            protected_attribute=self.protected_attrb,
            protected_attribute_value=self.protected_attribute_value,
        )
        self.train_set = MIMICProtectedGroupDataset(
            self.data_csv,
            split="train",
            protected_attribute=self.protected_attrb,
            protected_attribute_value=self.protected_attribute_value,
        )
        self.val_set = MIMICProtectedGroupDataset(
            self.data_csv,
            split="validate",
            protected_attribute=self.protected_attrb,
            protected_attribute_value=self.protected_attribute_value,
        )

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return test_loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader


class PG_SBS_MIMICDataModule(LightningDataModule):
    """
    Data module that simultaneously implements MIMICProtectedGroupDataModule
    for the validation loader and
    SBSMIMICEmbeddingModule for the training and test.
    """

    def __init__(self, data_csv, batch_size, num_workers, protected_attrb):
        super().__init__()
        self.protected_attrb = protected_attrb
        self.protected_group_map = protected_group_name_2_map[protected_attrb]
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = MIMICEmbeddingDataset(self.data_csv, split="train")
        self.val_set = MIMICEmbeddingDataset(self.data_csv, split="validate")
        self.test_sets = [MIMICEmbeddingDataset(self.data_csv, split="test")] + [
            MIMICProtectedGroupDataset(
                self.data_csv,
                split="validate",
                protected_attribute=self.protected_attrb,
                protected_attribute_value=pav,
            )
            for pav in self.protected_group_map.keys()
        ]

    def train_dataloader(self):
        data_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def test_dataloader(self):
        data_loader = DataLoader(
            self.test_sets[0],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        test_loader = DataLoader(
            self.test_sets[0],
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loaders = [
            DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for test_set in self.test_sets[1:]
        ]
        return [test_loader] + val_loaders

    def val_dataloader(self):
        data_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def make_weights_for_balanced_classes(self, data_csv, data_loader):
        data = pd.read_csv(data_csv)
        protected_group_weights = {}
        protected_group_name_2_map = {
            "gender": gender_map,
            "age_decile": age_map,
            "race": race_map,
        }
        sum_weights = 0
        protected_group_map = protected_group_name_2_map[self.protected_attrb]
        for k in protected_group_map.keys():
            num_group = data[data[self.protected_attrb] == k].shape[0]
            weight_group = 1.0 / (num_group + 0.01)
            protected_group_weights[protected_group_map[k]] = weight_group
            sum_weights += weight_group

        # This is not strictly necessary, but makes for a more easy to read weight ratio
        for k in protected_group_map.keys():
            protected_group_weights[protected_group_map[k]] /= sum_weights

        weight = torch.zeros(data.shape[0]).to(device)
        protected_group_idx = protected_group_to_index[self.protected_attrb]
        for i, (emb, _, demographic_data) in enumerate(data_loader):
            idxs = torch.arange(0, emb.shape[0]) + (i * self.batch_size)
            idxs = idxs.to(dtype=torch.long, device=device)
            for j, idx in enumerate(idxs):
                weight[idx] = protected_group_weights[
                    demographic_data[j, protected_group_idx].item()
                ]
        return weight


class SBS_MIMICDataModule(LightningDataModule):
    """
    Data module that simultaneously implements MIMICProtectedGroupDataModule
    for the validation loader and
    SBSMIMICEmbeddingModule for the training and test.
    """

    def __init__(self, data_csv, batch_size, num_workers, protected_attrb):
        super().__init__()
        self.protected_attrb = protected_attrb
        self.protected_group_map = protected_group_name_2_map[protected_attrb]
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = MIMICEmbeddingDataset(self.data_csv, split="train")
        self.val_set = MIMICEmbeddingDataset(self.data_csv, split="validate")
        self.test_sets = [
            MIMICEmbeddingDataset(self.data_csv, split="test"),
            MIMICEmbeddingDataset(self.data_csv, split="validate"),
        ]

        data_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def test_dataloader(self):
        data_loader = DataLoader(
            self.test_sets[0],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        test_loader = DataLoader(
            self.test_sets[0],
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loaders = DataLoader(
                self.test_sets[1],
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True,
            )
        return [test_loader, val_loaders]

    def val_dataloader(self):
        data_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def make_weights_for_balanced_classes(self, data_csv, data_loader):
        data = pd.read_csv(data_csv)
        protected_group_weights = {}
        protected_group_name_2_map = {
            "gender": gender_map,
            "age_decile": age_map,
            "race": race_map,
        }
        sum_weights = 0
        protected_group_map = protected_group_name_2_map[self.protected_attrb]
        for k in protected_group_map.keys():
            num_group = data[data[self.protected_attrb] == k].shape[0]
            weight_group = 1.0 / (num_group + 0.01)
            protected_group_weights[protected_group_map[k]] = weight_group
            sum_weights += weight_group

        # This is not strictly necessary, but makes for a more easy to read weight ratio
        for k in protected_group_map.keys():
            protected_group_weights[protected_group_map[k]] /= sum_weights

        weight = torch.zeros(data.shape[0]).to(device)
        protected_group_idx = protected_group_to_index[self.protected_attrb]
        for i, (emb, _, demographic_data) in enumerate(data_loader):
            idxs = torch.arange(0, emb.shape[0]) + (i * self.batch_size)
            idxs = idxs.to(dtype=torch.long, device=device)
            for j, idx in enumerate(idxs):
                weight[idx] = protected_group_weights[
                    demographic_data[j, protected_group_idx].item()
                ]
        return weight


class MIMICProtectedGroupDataset(Dataset):
    def __init__(
        self, data_path, split, protected_attribute, protected_attribute_value
    ):
        # Load your dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data["split"] == split]
        # Only choose rows with a specific vlue for a protected attribute
        self.protected_attribute = protected_attribute
        self.protected_attribute_value = protected_attribute_value
        self.data = self.data[
            self.data[protected_attribute] == protected_attribute_value
        ]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        demographic_data = np.array(
            [
                gender_map[sample["gender"]],
                age_map[sample["age_decile"]],
                race_map[sample["race"]],
            ]
        )
        sample_path = sample["path"]
        # If the processed data has been made by the tfrecode data, fix the path to use np instead:
        if '.tf' in sample_path:
            emb = np.load(sample["path"].split(".tf")[0] + ".npy", allow_pickle=True)
        else:
            emb = np.load(sample["path"], allow_pickle=True)

        label = np.array([sample[d] for d in disease_labels])
        return (
            torch.from_numpy(emb).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(demographic_data).long(),
        )

class MultipleMiMicEmbeddingDataModule(MIMICEmbeddingModule):
    def __init__(self, data_csv, batch_size, num_workers, trainset_ratios):
        super().__init__(data_csv, batch_size, num_workers)
        self.trainset_ratios = trainset_ratios

    def setup(self, stage: str):
        self.test_set = MIMICEmbeddingDataset(self.data_csv, split='test')
        self.train_sets = [MIMICEmbeddingDataset(
            self.data_csv, split='train', subset_ratio=r) for r in self.trainset_ratios]
        self.val_set = MIMICEmbeddingDataset(self.data_csv, split='validate')

    def train_dataloader(self):
        return [DataLoader(ts, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True) for ts in self.train_sets]




class TinyMIMICEmbeddingDataset(Dataset):
    def __init__(self, data_path, split, subset_ratio=1.0):
        # Load your dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data['split'] == split]
        self.data = self.data[:15]
        if subset_ratio != 1.0:
            self.data = self.data.sample(
                frac=subset_ratio, random_state=1, replace=False)

    def __len__(self):
        return 15

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        demographic_data = np.array(
            [
                gender_map[sample["gender"]],
                age_map[sample["age_decile"]],
                race_map[sample["race"]],
            ]
        )
        # If the processed data has been made by the tfrecode data, fix the path to use np instead:
        sample_path = sample["path"]
        if '.tf' in sample_path:
            emb = np.load(sample["path"].split(".tf")[0] + ".npy", allow_pickle=True)
        else:
            emb = np.load(sample["path"], allow_pickle=True)
        label = np.array([sample[d] for d in disease_labels])
        return (
            torch.from_numpy(emb).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(demographic_data).long(),
        )
    

class TinyPG_SBS_MIMICDataModule(LightningDataModule):
    """
    Data module that simultaneously implements MIMICProtectedGroupDataModule
    for the validation loader and
    SBSMIMICEmbeddingModule for the training and test.
    """

    def __init__(self, data_csv, batch_size, num_workers, protected_attrb):
        super().__init__()
        self.protected_attrb = protected_attrb
        self.protected_group_map = protected_group_name_2_map[protected_attrb]
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.train_set = TinyMIMICEmbeddingDataset(self.data_csv, split="train")
        self.val_set = TinyMIMICEmbeddingDataset(self.data_csv, split="validate")
        self.test_sets = [TinyMIMICEmbeddingDataset(self.data_csv, split="test")] + [
            TinyMIMICProtectedGroupDataset(
                self.data_csv,
                split="validate",
                protected_attribute=self.protected_attrb,
                protected_attribute_value=pav,
            )
            for pav in self.protected_group_map.keys()
        ]

    def train_dataloader(self):
        data_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        train_loader = DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def test_dataloader(self):
        data_loader = DataLoader(
            self.test_sets[0],
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        test_loader = DataLoader(
            self.test_sets[0],
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        val_loaders = [
            DataLoader(
                test_set,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True,
            )
            for test_set in self.test_sets[1:]
        ]
        return [test_loader] + val_loaders

    def val_dataloader(self):
        data_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        weights = self.make_weights_for_balanced_classes(self.data_csv, data_loader)
        sampler = WeightedRandomSampler(weights, len(weights))
        val_loader = DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return val_loader

    def make_weights_for_balanced_classes(self, data_csv, data_loader):
        data = pd.read_csv(data_csv)[:15]
        protected_group_weights = {}
        protected_group_name_2_map = {
            "gender": gender_map,
            "age_decile": age_map,
            "race": race_map,
        }
        sum_weights = 0
        protected_group_map = protected_group_name_2_map[self.protected_attrb]
        for k in protected_group_map.keys():
            num_group = data[data[self.protected_attrb] == k].shape[0]
            weight_group = 1.0 / (num_group + 0.01)
            protected_group_weights[protected_group_map[k]] = weight_group
            sum_weights += weight_group

        # This is not strictly necessary, but makes for a more easy to read weight ratio
        for k in protected_group_map.keys():
            protected_group_weights[protected_group_map[k]] /= sum_weights

        weight = torch.zeros(data.shape[0]).to(device)
        protected_group_idx = protected_group_to_index[self.protected_attrb]
        for i, (emb, _, demographic_data) in enumerate(data_loader):
            idxs = torch.arange(0, emb.shape[0]) + (i * self.batch_size)
            idxs = idxs.to(dtype=torch.long, device=device)
            for j, idx in enumerate(idxs):
                weight[idx] = protected_group_weights[
                    demographic_data[j, protected_group_idx].item()
                ]
        return weight
    


class TinyMIMICProtectedGroupDataset(Dataset):
    def __init__(
        self, data_path, split, protected_attribute, protected_attribute_value
    ):
        # Load your dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data["split"] == split]
        # Only choose rows with a specific vlue for a protected attribute
        self.protected_attribute = protected_attribute
        self.protected_attribute_value = protected_attribute_value
        self.data = self.data[
            self.data[protected_attribute] == protected_attribute_value
        ]
        self.data = self.data[:15]

    def __len__(self):
        return 15

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        demographic_data = np.array(
            [
                gender_map[sample["gender"]],
                age_map[sample["age_decile"]],
                race_map[sample["race"]],
            ]
        )
        sample_path = sample["path"]
        # If the processed data has been made by the tfrecode data, fix the path to use np instead:
        if '.tf' in sample_path:
            emb = np.load(sample["path"].split(".tf")[0] + ".npy", allow_pickle=True)
        else:
            emb = np.load(sample["path"], allow_pickle=True)

        label = np.array([sample[d] for d in disease_labels])
        return (
            torch.from_numpy(emb).float(),
            torch.from_numpy(label).float(),
            torch.from_numpy(demographic_data).long(),
        )
