import random
import numpy as np

import torch
from tqdm import trange, tqdm

from preprocess import get_dataset, get_test
from dataset import Hahow_Dataset
from model import Classifier

BATCH_SIZE = 64
NUM_WORKER = 8

LR = 0.001
DROPOUT = 0.1
HIDDEN_NUM = 128

DEVICE = 'cuda:0'

AP_K = 50


def main():
    print('***Model***')
    model = Classifier(DROPOUT, 91, HIDDEN_NUM, 91)
    model.load_state_dict(torch.load('./tmp.pt'))
    model.to(DEVICE)
    model.eval()

    print('***Hahow_Dataset***')
    # TODO_: crecate DataLoader for train / dev datasets
    test_datasets = Hahow_Dataset(get_dataset(), 'dev')

    print('***DataLoader***')
    test_loader = torch.utils.data.DataLoader(
        test_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )

    y_preds = []
    for i, data in tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test',
                        leave=False):

        # data collate_fn
        _input, _ = data
        _input = _input.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            y_pred = model(_input)
            y_preds.extend(y_pred)
        #     print(len(y_pred))
        #     print(len(y_preds))
        # input()

    subgroup_preds = []
    for y in y_preds:
        # subgroup (aka. topic)
        subgroup_pred = torch.topk(y, AP_K).indices

        # convert list_id to subgroup_id
        subgroup_pred = torch.add(subgroup_pred, 1)

        subgroup_preds.append(subgroup_pred.tolist())

        # print(subgroup_pred)
        # input()

    df = get_test('test_seen_group.csv')
    # df = get_test('test_unseen_group.csv')

    with open('./seen_user_topic.csv', 'w') as f:
    # with open('./unseen_user_topic.csv', 'w') as f:
        f.write('user_id,subgroup\n')
        for (_, c_row), subgroup_pred in zip(df.iterrows(), subgroup_preds):
            c_user_id = c_row['user_id']

            c_subgroup = [str(sgp) for sgp in subgroup_pred]
            c_subgroup = ' '.join(c_subgroup)

            f.write(f'{c_user_id},{c_subgroup}\n')

    print("All Epoch on Test were finished.\n")
    return


if __name__ == "__main__":
    main()
