"""
Using this file as a scratchpad to test out stuff
faster since it takes too much time to test using lightning torch
"""

import torch
from tqdm import tqdm
from models.mlp_model import MLP
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from datasets.mimic_dataset import SBSMIMICEmbeddingModule


epochs = 1
num_workers = 2
batch_size = 16
data_csv = "/Users/bahman/Documents/courses/Fairness/project/code/misc/MIMIC_CXR_EMB/processed_mimic_df.csv"


model = MLP()
loss = CrossEntropyLoss()
optimizer = Adam(model.parameters)
data_loader = SBSMIMICEmbeddingModule(
    data_csv=data_csv,
    batch_size=batch_size,
    num_workers=num_workers,
    protected_attrb="gender",
)

for epocn in tqdm(range(epochs)):
    for embs, labels, demographic_data in enumerate(data_loader):
        optimizer.zero_grad()
        predictions = model(embs)
        loss(embs, labels)
        optimizer.step()
