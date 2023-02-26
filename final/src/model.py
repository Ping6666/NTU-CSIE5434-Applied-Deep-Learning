import torch
from torch import nn


class Hahow_Model(nn.Module):

    def __init__(
        self,
        topic_course: torch.Tensor,
        num_feature: int,
        hidden_size: int,
        num_class: int,
        dropout: float,
        device: str,
    ) -> None:
        super(Hahow_Model, self).__init__()

        # self.device = device
        self.topic_course = torch.tensor(
            topic_course,
            dtype=torch.float32,
        ).to(device)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        self.fc1 = nn.Linear(num_feature, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        # self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, num_class)

        self.bn = nn.BatchNorm1d(hidden_size)
        return

    def predict_course_search(self, predict: torch.Tensor,
                              topic_course: torch.Tensor) -> torch.Tensor:
        '''
            Args:
                predict: current predict metrix
                topic_course: topic course related metrix
        '''

        # loop the batch_size
        rt_tensor = None
        for p in predict:
            with torch.enable_grad():
                _, idx = torch.topk(p, 45, largest=False)
                p = p.scatter(-1, idx, value=0.05)

                p = p.reshape(-1, 1)

                # (728, 91), (91,) => (728,)
                rt = torch.matmul(topic_course,
                                  p.to(torch.float32)).reshape(1, -1)

                if rt_tensor is None:
                    rt_tensor = rt
                else:
                    rt_tensor = torch.cat((rt_tensor, rt), 0)

        return rt_tensor

    def forward(self, _x: torch.Tensor) -> torch.Tensor:

        _x = self.dropout(self.relu(self.bn(self.fc1(_x))))
        _x = self.dropout(self.relu(self.bn(self.fc2(_x))))
        # _x = self.dropout(self.relu(self.bn(self.fc3(_x))))
        _x = self.fc4(_x)

        return _x, self.predict_course_search(_x, self.topic_course)
