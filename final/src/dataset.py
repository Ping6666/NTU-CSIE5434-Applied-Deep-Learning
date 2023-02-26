from typing import Tuple

import numpy as np

import torch
from torch.utils.data import Dataset


class Hahow_Dataset(Dataset):

    def __init__(self, data: np.ndarray, mode='Train') -> None:
        self.mode = mode

        data, gt = data

        print('all columns', data.columns)
        print('length', data.shape[0])
        data = data.to_numpy()

        data_gender = np.array(data[:, 0].tolist())
        data = np.array(data[:, 1:].tolist())

        self.x_gender = torch.tensor(data_gender, dtype=torch.long)
        self.x_vector = torch.tensor(data[:, 0], dtype=torch.float32)
        # print('x_gender', len(self.x_gender))
        # print('x_vector', len(self.x_vector))

        self.gt = None
        if self.mode == 'TTT':
            # self.gt = np.array(gt, dtype=object)
            # self.gt = torch.tensor(gt)
            # self.gt = torch.tensor(gt, dtype=torch.int)
            self.gt = torch.tensor(gt)
            # self.gt = gt
            # print('gt', len(self.gt))

        if self.mode == 'Train' or self.mode == 'TTT':
            self.y = torch.tensor(data[:, 1], dtype=torch.float32)
            # print('y', len(self.y))
        return

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_x_gender = self.x_gender[id]
        c_x_vector = self.x_vector[id]
        c_y = []
        c_gt = []

        # ulimit -a
        # sudo sh -c "ulimit -n 65535 && exec su $LOGNAME"

        if self.mode == 'TTT':
            c_gt = self.gt[id]
            # print('c_gt', c_gt)
            # print('c_gt', len(c_gt))

        if self.mode == 'Train' or self.mode == 'TTT':
            c_y = self.y[id]
        return (c_x_gender, c_x_vector), c_y, c_gt

    def __len__(self) -> int:
        return len(self.x_vector)
