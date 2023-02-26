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
        ((user_id, user_id_labelencoder), df, input_feature,
         (course_id_labelencoder)) = data

        # preprocess
        print('Hahow_Dataset all columns', df.columns)
        print('length', df.shape[0])
        df = df.to_numpy()

        self.user_id = torch.tensor(user_id)
        self.user_id_labelencoder = user_id_labelencoder

        self.x_interests = torch.tensor(np.array(df[:, 0].tolist()),
                                        dtype=torch.long)
        self.y = torch.tensor(np.array(df[:, 1].tolist()), dtype=torch.float32)

        self.input_feature = torch.tensor(input_feature)

        # self.course_id = torch.tensor(np.array(course_id.tolist()))
        self.course_id_labelencoder = course_id_labelencoder
        return

    def __getitem__(self, id: int):
        c_user_id = self.user_id[id]

        c_if = self.input_feature
        c_x_interests = self.x_interests[id]
        c_y = self.y[id]

        return (c_user_id, (c_x_interests, c_if, c_y))

    def __len__(self) -> int:
        return len(self.x_interests)

    def get_user_id_labelencoder(self):
        return self.user_id_labelencoder

    def get_course_id_labelencoder(self):
        return self.course_id_labelencoder
