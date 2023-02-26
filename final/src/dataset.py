from typing import Tuple

import numpy as np

import torch
from torch.utils.data import Dataset

# from preprocess import MODES


class Hahow_Dataset(Dataset):

    # def __init__(self, data, mode: str) -> None:
    def __init__(self, data) -> None:
        # assert mode in MODES, 'Hahow_Dataset | wrong mode!!!'
        # self.mode = mode

        # unpack dataset
        user_id, df, courses_label = data

        self.user_id = torch.tensor(user_id)
        self.courses_label = torch.tensor(courses_label)

        # train part
        print('Hahow_Dataset all columns', df.columns)
        print('length', df.shape[0])
        df = df.to_numpy()

        x_gender = np.array(df[:, 0].tolist())
        x_vector = np.array(df[:, 1].tolist())
        y = np.array(df[:, 2].tolist())

        self.x_gender = torch.tensor(x_gender, dtype=torch.long)
        self.x_vector = torch.tensor(x_vector, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        return

    def __getitem__(self, id: int) -> Tuple[torch.Tensor, torch.Tensor]:
        c_user_id = self.user_id[id]
        c_x_gender = self.x_gender[id]
        c_x_vector = self.x_vector[id]
        c_y = self.y[id]
        c_courses_label = self.courses_label[id]
        return c_user_id, (c_x_gender, c_x_vector, c_y), c_courses_label

    def __len__(self) -> int:
        return len(self.x_vector)
