import torch.nn as nn


class MLP(nn.Module):
    """Perceptrón multicapa: Linear -> ReLU -> Dropout por cada capa oculta."""

    def __init__(self, in_features, hidden_sizes, n_classes, dropout=0.0):
        super().__init__()
        layers = []
        prev = in_features
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
