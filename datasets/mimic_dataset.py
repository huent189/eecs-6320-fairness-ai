import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from lightning.pytorch.core.datamodule import LightningDataModule
import numpy as np

# gender_map = {'M': 0, 'F': 1, np.nan : 2}
# age_map = {'40-60': 2, '80-100': 4, '60-80': 3, '0-20': 0, '20-40': 1, np.nan : 5}
# race_map = {'WHITE': 0, 'BLACK/AFRICAN AMERICAN': 1, 'ASIAN': 2, 'HISPANIC/LATINO': 3, 'AMERICAN INDIAN/ALASKA NATIVE': 4,
#             'OTHER': 5, 'UNKNOWN': 6, 'UNABLE TO OBTAIN' : 6, np.nan: 6}

gender_map = {'M': 0, 'F': 1}
age_map = {'40-60': 2, '80+': 4, '60-80': 3, '0-20': 0, '20-40': 1}
race_map = {'WHITE': 0, 'BLACK/AFRICAN AMERICAN': 1, 'ASIAN': 2, 'HISPANIC/LATINO': 3, 'AMERICAN INDIAN/ALASKA NATIVE': 4,
            'OTHER': 5}
num_groups_per_attrb = [2, 5, 6]

class MIMICEmbeddingDataset(Dataset):
    def __init__(self, data_path, split):
        # Load your dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data['split'] == split]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        demographic_data = np.array([gender_map[sample['gender']], age_map[sample['age_decile']], race_map[sample['race']]])
        emb = np.load(sample['path'], allow_pickle=True)
        label = np.array([sample['No_Finding']])
        return torch.from_numpy(emb).float(), torch.from_numpy(label).float(), torch.from_numpy(demographic_data).long()

class MIMICEmbeddingModule(LightningDataModule):
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



if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
device = torch.device(device)
def make_weights_for_balanced_classes(self, pd_path, data_loader):
    data = pd.read_csv(pd_path)
    protected_group_weights = {}
    protected_group_name_2_map = {
        'gender': gender_map,
        'age_decile': age_map,
        'race': race_map,
    }
    sum_weights = 0
    protected_group_map = protected_group_name_2_map[self.protected_attrb]
    for k in protected_group_map.keys():
        num_group = data[data[self.protected_attrb] == k].shape[0]
        weight_group = 1. / (num_group + 0.01)
        protected_group_weights[k] = weight_group
        sum_weights += weight_group

    # This is not strictly necessary, but makes for a more easy to read weight ratio
    for k in protected_group_map.keys():
        protected_group_weights[k] /= sum_weights

    weight = torch.zeros(data.shape[0]).to(device)

    for i, (img, label) in enumerate(data_loader):
        idx = torch.arange(0, img.shape[0]) + (i * self.batch_size)
        idx = idx.to(dtype=torch.long, device=device)
        weight[idx] = protected_group_weights[label]

    return weight 
