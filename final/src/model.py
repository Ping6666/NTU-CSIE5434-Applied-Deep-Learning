import torch
from torch import nn


class Classifier(nn.Module):

    def __init__(
        self,
        dropout: float,
        num_feature: int,
        hidden_size: int,
        num_class: int,
    ) -> None:

        super(Classifier, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(num_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_class)

        self.bn = nn.BatchNorm1d(hidden_size)

        return

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO: implement model forward

        x = self.dropout(self.relu(self.bn(self.fc1(x))))
        x = self.dropout(self.relu(self.bn(self.fc2(x))))
        x = self.dropout(self.relu(self.bn(self.fc3(x))))

        return self.fc4(x)
