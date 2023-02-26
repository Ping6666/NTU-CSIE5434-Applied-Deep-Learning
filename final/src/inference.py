import torch
from tqdm import tqdm

from preprocess import predataset_workhouse, get_dataframe_test, dataset_workhouse
from dataset import Hahow_Dataset
from model import Classifier

BATCH_SIZE = 64
NUM_WORKER = 8

LR = 0.001
DROPOUT = 0.1
HIDDEN_NUM = 128

TOPK = 91

DEVICE = 'cuda:1'


def topk_convertion(group_lists):
    # print(group_lists[0])
    c_group_lists = [
        torch.add(torch.topk(group_list, TOPK).indices, 1).tolist()
        for group_list in group_lists
    ]
    # print(c_group_lists[0])
    return c_group_lists


def predict(test_loader, model, predict_file, save_file):
    '''
    predict_file: 'test_seen_group.csv', 'test_unseen_group.csv'
    save_file: './seen_user_topic.csv', './unseen_user_topic.csv'
    '''

    y_preds = []
    for i, data in tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test',
                        leave=False):

        # data collate_fn
        (_gender, _vector), _ = data
        _gender = _gender.to(DEVICE)
        _vector = _vector.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            y_pred = model(_gender, _vector)
            y_preds.extend(y_pred)

    subgroup_preds = topk_convertion(y_preds)

    df = get_dataframe_test(predict_file)

    with open(save_file, 'w') as f:
        f.write('user_id,subgroup\n')
        for (_, c_row), subgroup_pred in zip(df.iterrows(), subgroup_preds):
            c_user_id = c_row['user_id']

            c_subgroup = [str(sgp) for sgp in subgroup_pred]
            c_subgroup = ' '.join(c_subgroup)

            f.write(f'{c_user_id},{c_subgroup}\n')

    return


def main():
    print('***Model***')
    model = Classifier(DROPOUT, 3, 91, HIDDEN_NUM, 91)
    model.load_state_dict(torch.load('./tmp.pt'))
    model.to(DEVICE)
    model.eval()

    print('***Data***')
    df_users, df_courses = predataset_workhouse()

    print('***Hahow_Dataset***')
    # TODO_: crecate DataLoader for train / dev datasets
    test_seen_datasets = Hahow_Dataset(
        dataset_workhouse(df_users, df_courses, 'Test_Seen'), 'Dev')
    test_unseen_datasets = Hahow_Dataset(
        dataset_workhouse(df_users, df_courses, 'Test_UnSeen'), 'Dev')

    print('***DataLoader***')
    test_seen_loader = torch.utils.data.DataLoader(
        test_seen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )
    test_unseen_loader = torch.utils.data.DataLoader(
        test_unseen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
    )

    predict(test_seen_loader, model, 'test_seen_group.csv',
            './seen_user_topic.csv')
    predict(test_unseen_loader, model, 'test_unseen_group.csv',
            './unseen_user_topic.csv')

    print("All Epoch on Test were finished.\n")
    return


if __name__ == "__main__":
    main()
