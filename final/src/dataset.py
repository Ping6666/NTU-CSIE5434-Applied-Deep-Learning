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
        ((user_id, user_id_labelencoder), df, topic,
         (course_id, course_id_labelencoder)) = data

        # preprocess
        print('Hahow_Dataset all columns', df.columns)
        print('length', df.shape[0])
        df = df.to_numpy()

        self.user_id = torch.tensor(user_id)
        self.user_id_labelencoder = user_id_labelencoder

        self.x_vector = torch.tensor(np.array(df[:, 0].tolist()),
                                     dtype=torch.float32)
        self.y_topic_vector = torch.tensor(np.array(df[:, 1].tolist()),
                                           dtype=torch.float32)
        self.y_course_vector = torch.tensor(np.array(df[:, 2].tolist()),
                                            dtype=torch.float32)

        self.topic = torch.tensor(np.array(topic.tolist()))

        self.course_id = torch.tensor(np.array(course_id.tolist()))
        self.course_id_labelencoder = course_id_labelencoder
        return

    def __getitem__(self, id: int):
        c_user_id = self.user_id[id]

        c_x_vector = self.x_vector[id]
        c_y_topic_vector = self.y_topic_vector[id]
        c_y_course_vector = self.y_course_vector[id]

        c_topic = self.topic[id]
        c_course_id = self.course_id[id]

        return (c_user_id, c_x_vector, (c_y_topic_vector, c_y_course_vector),
                (c_topic, c_course_id))

    def __len__(self) -> int:
        return len(self.user_id)

    def get_user_id_labelencoder(self):
        return self.user_id_labelencoder

    def get_course_id_labelencoder(self):
        return self.course_id_labelencoder
