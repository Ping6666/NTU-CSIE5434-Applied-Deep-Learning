from typing import Tuple

import numpy as np

import torch
from torch.utils.data import Dataset


class Hahow_Dataset(Dataset):

    def __init__(self, data: np.ndarray, mode='Train') -> None:
        self.mode = mode

        print('all columns', data.columns)
        print('length', data.shape[0])
        data = data.to_numpy()

        data_gender = np.array(data[:, 0].tolist())
        data = np.array(data[:, 1:].tolist())

        self.x_gender = torch.tensor(data_gender, dtype=torch.long)
        self.x_vector = torch.tensor(data[:, 0], dtype=torch.float32)

        if self.mode == 'Train':
            self.y = torch.tensor(data[:, 1], dtype=torch.float32)
        return

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_x_gender = self.x_gender[id]
        c_x_vector = self.x_vector[id]
        c_y = []

        if self.mode == 'Train':
            c_y = self.y[id]
        return (c_x_gender, c_x_vector), c_y

    def __len__(self) -> int:
        return len(self.x_vector)
