import random
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from preprocess import MODES

from preprocess import (
    global_init_workhouse,
    preprocess_workhouse,
    dataset_workhouse,
    inference_prediction,
)
from model import Hahow_Model
from dataset import Hahow_Dataset

## global ##

SEED = 5487
DEVICE = 'cuda:1'

BATCH_SIZE = 64
NUM_WORKER = 8

DROPOUT = 0.1

TOPK = 50


def set_seed(seed):
    ## insure the reproducibility ##
    # Ref. https://pytorch.org/docs/stable/notes/randomness.html

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return


## predict ##


def predict(test_loader, model):
    _user_ids, _y_topic_preds, _y_course_preds = [], [], []
    for _, data in tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test',
                        leave=False):
        # data collate_fn
        _user_id, (_x_gender, _x_vector, _), (_, _) = data
        _x_gender = _x_gender.to(DEVICE)
        _x_vector = _x_vector.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            _y_pred = model(_x_gender, _x_vector)

        # report
        _user_ids.extend(_user_id)
        _y_topic_pred, _y_course_pred = inference_prediction(_y_pred)
        _y_topic_preds.extend(_y_topic_pred)
        _y_course_preds.extend(_y_course_pred)
    return _user_ids, _y_topic_preds, _y_course_preds


def save_prediction(prediction, les, save_file_topic, save_file_course):
    _user_ids, _y_topic_preds, _y_course_preds = prediction
    user_le, course_le = les

    _user_ids = list(user_le.inverse_transform(_user_ids))

    with open(save_file_topic, 'w') as f:
        f.write('user_id,subgroup\n')
        for c_user_id, c_pred in zip(_user_ids, _y_topic_preds):
            c_subgroup = ' '.join([str(c) for c in c_pred])

            f.write(f'{c_user_id},{c_subgroup}\n')

    with open(save_file_course, 'w') as f:
        f.write('user_id,course_id\n')
        for c_user_id, c_pred in zip(_user_ids, _y_course_preds):
            _c_pred = user_le.inverse_transform(np.array(c_pred))
            c_course = ' '.join([str(c) for c in _c_pred])

            f.write(f'{c_user_id},{c_course}\n')
    return


## main ##


def main():
    set_seed(SEED)

    print('***Model***')
    model = Hahow_Model(DROPOUT)
    model.load_state_dict(torch.load('./save/topic_25.pt'))
    model.to(DEVICE)
    model.eval()

    print('***Global***')
    global_init_workhouse()

    print('***Data***')
    df_preprocess = preprocess_workhouse()

    print('***Hahow_Dataset***')
    test_seen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[3]))
    test_unseen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[4]))

    print('***DataLoader***')
    test_seen_loader = DataLoader(
        test_seen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )
    test_unseen_loader = DataLoader(
        test_unseen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )

    print('***Predict Seen***')
    test_seen_user_le = test_seen_datasets.get_user_id_labelencoder()
    test_seen_course_le = test_seen_datasets.get_course_id_labelencoder()
    save_prediction(
        predict(test_seen_loader, model),
        (test_seen_user_le, test_seen_course_le),
        './seen_user_topic.csv',
        './seen_user_course.csv',
    )

    print('***Predict UnSeen***')
    test_unseen_user_le = test_unseen_datasets.get_user_id_labelencoder()
    test_unseen_course_le = test_unseen_datasets.get_course_id_labelencoder()
    save_prediction(
        predict(test_unseen_loader, model),
        (test_unseen_user_le, test_unseen_course_le),
        './unseen_user_topic.csv',
        './unseen_user_course.csv',
    )

    print('All Epoch on Test were finished.\n')
    return


if __name__ == "__main__":
    main()
