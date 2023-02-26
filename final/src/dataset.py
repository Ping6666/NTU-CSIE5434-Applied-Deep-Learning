from typing import Tuple

import numpy as np

import torch
from torch.utils.data import Dataset


class Hahow_Dataset(Dataset):

    def __init__(self, data: np.ndarray, mode='Train') -> None:
        self.mode = mode
        data = np.array(data.tolist())

        if self.mode == 'Train':
            self.x = torch.tensor(data[:, 0], dtype=torch.float32)
            self.y = torch.tensor(data[:, 1], dtype=torch.float32)
            assert len(self.x) == len(self.y), 'length unmatch'
        else:
            self.x = torch.tensor(data[:, 0], dtype=torch.float32)
        return

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_x = c_y = []

        if self.mode == 'Train':
            c_x = self.x[id]
            c_y = self.y[id]
        else:
            c_x = self.x[id]
        return c_x, c_y

    def __len__(self) -> int:
        return len(self.x)
