import torch
from sklearn.model_selection import StratifiedKFold

class StratifiedBatchSampler(torch.utils.data.Sampler):
    """Stratified batch sampling
    Provides equal representation of target classes in each batch
    """
    def __init__(self, dataset, labels, group_column, batch_size):
        self.dataset = dataset
        self.labels = labels
        self.group_column = group_column
        self.batch_size = batch_size
        
        # Get unique groups
        self.unique_groups = torch.unique(dataset[:, group_column])
        self.num_groups = len(self.unique_groups)
        
        # Create indices for each group
        self.group_indices = {group: (labels == group).nonzero().view(-1)
                              for group in self.unique_groups}
        
        # Initialize iterators for each group
        self.iterators = {group: iter(torch.randperm(len(indices)).tolist())
                          for group, indices in self.group_indices.items()}
        
    def __iter__(self):
        batch = []
        for _ in range(len(self)):
            # Randomly select a group
            group = self.unique_groups[random.randint(0, self.num_groups - 1)]
            # Get next index from selected group
            index = next(self.iterators[group])
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.dataset) // self.batch_size