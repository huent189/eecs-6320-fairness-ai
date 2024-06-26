import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lightning.pytorch.core.datamodule import LightningDataModule
import numpy as np


class MIMICEmbeddingDataset(Dataset):
    gender_map = {'M': 0, 'F': 1}
    age_map = {'40-60': 2, '80+': 4, '60-80': 3, '0-20': 0, '20-40': 1}
    race_map = {'WHITE': 0, 'BLACK/AFRICAN AMERICAN': 1, 'ASIAN': 2, 'HISPANIC/LATINO': 3, 'AMERICAN INDIAN/ALASKA NATIVE': 4,
                'OTHER': 5}
    disease_labels = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged_Cardiomediastinum', 'Fracture',
                    'Lung_Lesion', 'Lung_Opacity', 'No_Finding', 'Pleural_Effusion', 'Pleural_Other', 'Pneumonia', 'Pneumothorax', 'Support_Devices']
    NO_FINDING_INDEX = 8
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
            [self.gender_map[sample['gender']], self.age_map[sample['age_decile']], self.race_map[sample['race']]])
        emb = np.load(sample['path'], allow_pickle=True)
        label = np.array([sample[d] for d in self.disease_labels])
        return torch.from_numpy(emb).float(), torch.from_numpy(label).float(), torch.from_numpy(demographic_data).long()


class MIMICEmbeddingModule(LightningDataModule):
    NO_FINDING_INDEX = MIMICEmbeddingDataset.NO_FINDING_INDEX
    disease_labels = MIMICEmbeddingDataset.disease_labels
    def __init__(self, data_csv, batch_size, num_workers):
        super().__init__()
        self.data_csv = data_csv
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: str):
        self.test_set = MIMICEmbeddingDataset(self.data_csv, split='test')
        self.train_set = MIMICEmbeddingDataset(self.data_csv, split='train')
        self.val_set = MIMICEmbeddingDataset(self.data_csv, split='validate')

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return [DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
                DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)]


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
