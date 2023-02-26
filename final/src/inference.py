from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from preprocess import MODES

from preprocess import (
    global_init_workhouse,
    preprocess_workhouse,
    dataset_workhouse,
    convert_predict_to_int_list,
)
from model import Hahow_Model
from dataset import Hahow_Dataset

## global ##

DEVICE = 'cuda:1'

BATCH_SIZE = 64
NUM_WORKER = 8

EMBED_SIZE = 2
FEATURE_NUM = 91
HIDDEN_NUM = 128
DROPOUT = 0.01

TOPK = 50

## predict ##


def predict(test_loader, model):
    _user_ids, _y_preds = [], []
    for _, data in tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test',
                        leave=False):
        # data collate_fn
        _user_id, (_x_gender, _x_vector, _), _ = data
        _x_gender = _x_gender.to(DEVICE)
        _x_vector = _x_vector.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            _y_pred = model(_x_gender, _x_vector)

        # report
        _user_ids.extend(convert_predict_to_int_list(_user_id))
        _y_preds.extend(convert_predict_to_int_list(_y_pred))
    return _user_ids, _y_preds


def save_prediction(prediction, save_file):
    _user_ids, _y_preds = prediction

    with open(save_file, 'w') as f:
        f.write('user_id,subgroup\n')
        for c_user_id, c_pred in zip(_user_ids, _y_preds):
            c_subgroup = ' '.join([str(sgp) for sgp in c_pred])

            f.write(f'{c_user_id},{c_subgroup}\n')
    return


## main ##


def main():
    print('***Model***')
    model = Hahow_Model(EMBED_SIZE, FEATURE_NUM, HIDDEN_NUM, FEATURE_NUM,
                        DROPOUT)
    model.load_state_dict(torch.load('./save/topic_25.pt'))
    model.to(DEVICE)
    model.eval()

    print('***Global***')
    global_init_workhouse()

    print('***Data***')
    df_preprocess = preprocess_workhouse()

    print('***Hahow_Dataset***')
    test_seen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[3]), MODES[3])
    test_unseen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[4]), MODES[4])

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
    save_prediction(
        predict(test_seen_loader, model),
        'test_seen_group.csv',
        './seen_user_topic.csv',
    )

    print('***Predict UnSeen***')
    save_prediction(
        predict(test_unseen_loader, model),
        'test_unseen_group.csv',
        './unseen_user_topic.csv',
    )

    print('All Epoch on Test were finished.\n')
    return


if __name__ == "__main__":
    main()
