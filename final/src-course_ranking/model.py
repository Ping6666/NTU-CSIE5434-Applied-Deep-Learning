import torch
from torch import nn


class Hahow_Model(nn.Module):

    def __init__(
        self,
        dropout: float,
    ) -> None:
        super(Hahow_Model, self).__init__()

        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(91 + 768, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 8)
        self.fc5 = nn.Linear(8, 1)

        return

    def forward(self, user_vector: torch.Tensor,
                course_vector: torch.Tensor) -> torch.Tensor:
        # print('user_vector', user_vector.shape)  # torch.Size([64, 91])
        # print('course_vector', course_vector.shape)  # torch.Size([64, 768])

        _x = torch.cat((user_vector, course_vector), 1)

        _x = self.dropout(self.relu(self.fc1(_x)))
        _x = self.dropout(self.relu(self.fc2(_x)))
        _x = self.dropout(self.relu(self.fc3(_x)))
        _x = self.dropout(self.relu(self.fc4(_x)))

        return self.sig(self.fc5(_x))
