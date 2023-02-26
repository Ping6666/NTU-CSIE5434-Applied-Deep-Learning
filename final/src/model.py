import torch
from torch import nn


class Classifier(nn.Module):

    def __init__(
        self,
        dropout: float,
        embedding_size: int,
        num_feature: int,
        hidden_size: int,
        num_class: int,
    ) -> None:

        super(Classifier, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.e = nn.Embedding(4, embedding_size, padding_idx=0)

        self.fc1 = nn.Linear(num_feature + embedding_size, hidden_size)
        # self.fc1 = nn.Linear(num_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_class)

        self.bn = nn.BatchNorm1d(hidden_size)

        return

    def forward(self, x_gender: torch.Tensor,
                x_vector: torch.Tensor) -> torch.Tensor:
        x_gender = self.e(x_gender)
        _x = torch.cat((x_gender, x_vector), 1)

        # _x = x_vector

        _x = self.dropout(self.relu(self.bn(self.fc1(_x))))
        _x = self.dropout(self.relu(self.bn(self.fc2(_x))))
        _x = self.dropout(self.relu(self.bn(self.fc3(_x))))

        return self.fc4(_x)
