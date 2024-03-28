from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import torch, os
from lightning.pytorch.core.datamodule import LightningDataModule

class CheXpertEmbeddingDataset(Dataset):
    gender_map = {'Male': 0, 'Female': 1, np.nan :2}
    age_map = {'40-60': 2, '80+': 4, '60-80': 3, '0-20': 0, '20-40': 1, np.nan: 5}
    race_map = {'Other' : 5, 'White' : 0, 'Black': 1, 'American Indian/Alaska native': 4, 'Asian' : 2, 'Hispanic' : 3, np.nan: 6}
    disease_labels = ['Enlarged Cardiomediastinum', 'Cardiomegaly', 'Lung Opacity', 'Lung Lesion', 'Edema', 'Consolidation', 'Pneumonia', 'Atelectasis', 'Pneumothorax', 'Pleural Effusion', 'Pleural Other', 'Fracture', 'Support Devices', 'No Finding']
    NO_FINDING_INDEX = 13
    def __init__(self, data_path, split, feature_file_root, subset_ratio=1.0):
        # Load your dataset
        self.data = pd.read_csv(data_path)
        self.data = self.data[self.data['split'] == split]
        self.feature_file_root = feature_file_root
        if subset_ratio != 1.0:
            self.data = self.data.sample(
                frac=subset_ratio, random_state=1, replace=False)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        demographic_data = np.array(
            [self.gender_map[sample['GENDER']], self.age_map[sample['age_decile']], self.race_map[sample['Race']]])
        emb = np.load(os.path.join(self.feature_file_root, sample['path'] +'.npy'))
        label = np.array([sample[d] for d in self.disease_labels])
        return torch.from_numpy(emb).float(), torch.from_numpy(label).float(), torch.from_numpy(demographic_data).long()
class CheXpertEmbeddingModule(LightningDataModule):
    NO_FINDING_INDEX = CheXpertEmbeddingDataset.NO_FINDING_INDEX
    disease_labels = CheXpertEmbeddingDataset.disease_labels
    def __init__(self, data_root, batch_size, num_workers):
        super().__init__()
        self.data_root = data_root
        self.batch_size = batch_size
        self.num_workers = num_workers
    def setup(self, stage: str):
        feature_file = self.data_root
        self.test_set = CheXpertEmbeddingDataset(os.path.join(self.data_root, 'test_df.csv'), 'test', feature_file)
        self.train_set = CheXpertEmbeddingDataset(os.path.join(self.data_root, 'cxp_train_df.csv'), 'train', feature_file)
        self.val_set = CheXpertEmbeddingDataset(os.path.join(self.data_root, 'cxp_validation_df.csv'), 'validate', feature_file)
    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return [DataLoader(self.val_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True),
                DataLoader(self.test_set, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)]

if __name__ == '__main__':
    ds = CheXpertEmbeddingDataset('/local/ssd/huent/chexpert-embedding/cxp_train_df.csv', 'train', '/local/ssd/huent/chexpert-embedding')
    print(ds[0])
