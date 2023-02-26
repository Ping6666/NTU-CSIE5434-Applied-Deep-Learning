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
        self.fc3 = nn.Linear(hidden_size, hidden_size)
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
                p = p.scatter(-1, idx, value=0.05).reshape(-1, 1)

                # (728, 91), (91,) => (728,)
                rt = torch.matmul(topic_course,
                                  p.to(torch.float32)).reshape(1, -1)

                if rt_tensor is None:
                    rt_tensor = rt
                else:
                    rt_tensor = torch.cat((rt_tensor, rt), 0)

        return rt_tensor

    def forward(self, x_vector: torch.Tensor) -> torch.Tensor:

        _x = x_vector

        _x = self.dropout(self.relu(self.bn(self.fc1(_x))))
        _x = self.dropout(self.relu(self.bn(self.fc2(_x))))
        _x = self.dropout(self.relu(self.bn(self.fc3(_x))))
        _x = self.fc4(_x)

        return _x, self.predict_course_search(_x, self.topic_course)


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

        # # try to fix the error
        # '''
        # RuntimeError: element 0 of tensors does not require grad and does not have a grad_fn
        # '''
        # y_pred.requires_grad = True
        # y_true.requires_grad = True

        # print('y_pred', y_pred.shape)
        # print('y_true', y_true.shape)

        scores = []
        with torch.enable_grad():
            for c_y_pred, c_y_true in zip(y_pred, y_true):
                c_score = None
                if c_y_true is None:
                    c_score = torch.tensor(0).to(self.device)
                else:

                    if len(c_y_pred) > self.k:
                        c_y_pred = c_y_pred[:self.k]

                    counter = 0
                    num_hits = torch.zeros(self.k, dtype=torch.float32)
                    num_hits = num_hits.to(self.device)

                    for i, p in enumerate(c_y_pred):
                        if p in c_y_true and p not in c_y_pred[:i]:
                            counter += 1
                            num_hits[i] = counter

                    score = torch.sum(torch.dot(num_hits, self.multiplier))
                    denominator = min(len(c_y_true), self.k)

                    c_score = torch.multiply(score, (1.0 / denominator))

                # print('c_score', c_score, c_score.shape)
                # scores.append(Variable(c_score, requires_grad=True))
                scores.append(c_score)

            scores = torch.mean(torch.tensor(scores).to(self.device))

        # print('scores', scores.shape)
        return scores
