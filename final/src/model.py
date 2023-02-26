import torch
from torch import nn


class Hahow_Model(nn.Module):

    def __init__(
        self,
        embedding_size: int,
        num_feature: int,
        hidden_size: int,
        num_class: int,
        dropout: float,
    ) -> None:
        super(Hahow_Model, self).__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # self.embed = nn.Embedding(4, embedding_size, padding_idx=0)
        # self.fc1 = nn.Linear(num_feature + embedding_size, hidden_size)

        self.fc1 = nn.Linear(num_feature, hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # self.fc3 = nn.Linear(hidden_size, hidden_size)

        self.fc4 = nn.Linear(hidden_size, num_class)

        self.bn = nn.BatchNorm1d(hidden_size)
        return

    def forward(self, x_gender: torch.Tensor,
                x_vector: torch.Tensor) -> torch.Tensor:

        # TODO fix the BUG here
        # x_gender = self.embed(x_gender)
        # _x = torch.cat((x_gender, x_vector), 1)

        # print('g', x_gender.shape)
        # print('v', x_vector.shape)
        # print('_x', _x.shape)
        # input()

        _x = x_vector

        _x = self.dropout(self.relu(self.bn(self.fc1(_x))))
        _x = self.dropout(self.relu(self.bn(self.fc2(_x))))
        # _x = self.dropout(self.relu(self.bn(self.fc3(_x))))

        return self.fc4(_x)


class Hahow_Loss(nn.Module):
    '''
    calculate mean average precision@K
    '''

    def __init__(self, k, device):
        super(Hahow_Loss, self).__init__()
        self.k = int(k)
        self.device = device

        self.multiplier = torch.ones(k) / torch.arange(1, k + 1)
        self.multiplier = self.multiplier.to(self.device)

        return

    def forward(self, y_pred, y_true):
        if y_true is not None:
            return torch.tensor(0).to(self.device)

        if len(y_pred) > self.k:
            y_pred = y_pred[:self.k]

        counter = 0
        num_hits = torch.zeros(self.k, dtype=torch.float32).to(self.device)

        for i, p in enumerate(y_pred):
            if p in y_true and p not in y_pred[:i]:
                counter += 1
                num_hits[i] = counter

        score = torch.sum(torch.dot(num_hits, self.multiplier))
        denominator = min(len(y_true), self.k)

        score = torch.multiply(score * (1 / denominator)).copy()
        return torch.mean(score)
