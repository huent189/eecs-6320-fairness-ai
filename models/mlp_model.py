import torch.nn as nn
import torch
_DEFAULT_EMBEDDINGS_SIZE = 1376


class MLP(nn.Module):
    def __init__(self, num_heads=1, embeddings_size=_DEFAULT_EMBEDDINGS_SIZE, hidden_layer_sizes=[512, 256], dropout=0.0, seed=None, norm='layer'):
        super(MLP, self).__init__()
        self.seed = seed
        if self.seed is not None:
            torch.manual_seed(self.seed)

        layers = []
        input_size = embeddings_size
        for size in hidden_layer_sizes:
            layers.append(nn.Linear(input_size, size))
            layers.append(nn.ReLU())
            if norm == 'layer':
                layers.append(nn.LayerNorm(size))
            else:
                layers.append(nn.BatchNorm1d(size))
            layers.append(nn.Dropout(dropout))
            input_size = size

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(
            hidden_layer_sizes[-1] if hidden_layer_sizes else embeddings_size, num_heads)

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        output = self.classifier(x)
        return output
