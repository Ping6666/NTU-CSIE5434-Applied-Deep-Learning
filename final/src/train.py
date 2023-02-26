import random
import numpy as np

import torch
from tqdm import trange, tqdm

from preprocess import read_csv_subgroups, get_dataframe_users, get_dataframe_courses, dataset_workhouse
from dataset import Hahow_Dataset
from model import Classifier

from average_precision import mapk

SEED = 5487

BATCH_SIZE = 64
NUM_WORKER = 8

NUM_EPOCH = 5
LR = 0.001
DROPOUT = 0.1
HIDDEN_NUM = 128

AP_K = 50

DEVICE = 'cuda:0'


def topk_convertion(group_lists):
    print(group_lists[0])
    c_group_lists = [
        torch.add(torch.topk(group_list, AP_K).indices, 1).tolist()
        for group_list in group_lists
    ]
    print(c_group_lists[0])
    return c_group_lists


def train_per_epoch(train_loader, model, optimizer, loss_fn):
    train_loss, train_acc = 0, 0
    _preds, _labels = [], []
    for i, data in tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc='Train',
                        leave=False):
        # data collate_fn
        (_gender, _vector), _label = data
        _gender = _gender.to(DEVICE)
        _vector = _vector.to(DEVICE)
        _label = _label.to(DEVICE)

        # train: data -> model -> loss
        y_pred = model(_gender, _vector)
        loss = loss_fn(y_pred, _label)  # eval loss from loss_fn

        _preds.extend(y_pred)
        _labels.extend(_label)

        # update network
        optimizer.zero_grad()  # zero gradients
        loss.backward()
        optimizer.step()  # adjust learning weights

        # report: loss, acc.
        train_loss += loss.item()

    _labels = topk_convertion(_labels)
    _preds = topk_convertion(_preds)
    print(len(_labels))
    print(len(_labels[0]))
    print(len(_preds))
    print(len(_preds[0]))
    train_acc = mapk(_labels, _preds, AP_K)
    return train_loss, train_acc


def eval_per_epoch(eval_loader, model, loss_fn) -> None:
    eval_loss, eval_acc = 0, 0
    _preds, _labels = [], []
    for i, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Evaluation',
                        leave=False):

        # data collate_fn
        (_gender, _vector), _label = data
        _gender = _gender.to(DEVICE)
        _vector = _vector.to(DEVICE)
        _label = _label.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            y_pred = model(_gender, _vector)
            loss = loss_fn(y_pred, _label)  # eval loss from loss_fn

            _preds.extend(y_pred)
            _labels.extend(_label)

        # report: loss, acc.
        eval_loss += loss.item()

    _labels = topk_convertion(_labels)
    _preds = topk_convertion(_preds)
    print(len(_labels))
    print(len(_labels[0]))
    print(len(_preds))
    print(len(_preds[0]))
    train_acc = mapk(_labels, _preds, AP_K)
    return eval_loss, eval_acc


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


def get_datas():

    ## constant ##
    subgroups_dict = read_csv_subgroups('subgroups.csv')

    ## dataframe ##

    # get_users
    '''
    col: 'user_id', 'gender', 'occupation_titles', 'interests', 'recreation_names',
         'v_interests'
    '''
    df_users = get_dataframe_users('users.csv', subgroups_dict)

    # get_courses
    '''
    col: 'course_id', 'course_name', 'course_price', 'teacher_id',
         'teacher_intro', 'groups', 'sub_groups', 'topics', 'course_published_at_local',
         'description', 'will_learn', 'required_tools', 'recommended_background', 'target_group',
         'v_sub_groups'
    '''
    df_courses = get_dataframe_courses('courses.csv', subgroups_dict)

    return df_users, df_courses


def main():
    set_seed(SEED)

    print('***Model***')
    model = Classifier(DROPOUT, 3, 91, HIDDEN_NUM, 91)
    model.to(DEVICE)

    print('***Data***')
    df_users, df_courses = get_datas()

    print('***Hahow_Dataset***')
    # TODO_: crecate DataLoader for train / dev datasets
    train_datasets = Hahow_Dataset(dataset_workhouse(df_users, df_courses))
    eval_seen_datasets = Hahow_Dataset(
        dataset_workhouse(df_users, df_courses, 'Eval_Seen'))
    eval_unseen_datasets = Hahow_Dataset(
        dataset_workhouse(df_users, df_courses, 'Eval_UnSeen'))

    print('***DataLoader***')
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )
    eval_seen_loader = torch.utils.data.DataLoader(
        eval_seen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )
    eval_unseen_loader = torch.utils.data.DataLoader(
        eval_unseen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )

    print('***optimizer & loss_fn***')
    # TODO_: init optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), LR)  # or use SGD
    loss_fn = torch.nn.MSELoss()

    output_str = ''

    print('***Train & Evaluation***')
    epoch_pbar = trange(NUM_EPOCH, desc="Epoch")
    for epoch in epoch_pbar:
        # Training loop
        model.train()
        train_loss, train_acc = train_per_epoch(train_loader, model, optimizer,
                                                loss_fn)
        train_loss /= len(train_loader)
        train_acc /= len(train_datasets)
        print(f"\nTrain {(epoch + 1):03d} / {NUM_EPOCH:03d} :",
              f"loss = {train_loss:.5f}, acc = {train_acc:.5f}")
        output_str += f"\nTrain {(epoch + 1):03d} / {NUM_EPOCH:03d} : loss = {train_loss:.5f}, acc = {train_acc:.5f}"

        # Evaluation loop
        model.eval()
        eval_seen_loss, eval_seen_acc = eval_per_epoch(eval_seen_loader, model,
                                                       loss_fn)
        eval_seen_loss /= len(eval_seen_loader)
        eval_seen_acc /= len(eval_seen_datasets)
        print(f"Eval_Seen {(epoch + 1):03d} / {NUM_EPOCH:03d} :",
              f"loss = {eval_seen_loss:.5f}, acc = {eval_seen_acc:.5f}")
        output_str += f"\nEval {(epoch + 1):03d} / {NUM_EPOCH:03d} : loss = {eval_seen_loss:.5f}, acc = {eval_seen_acc:.5f}"

        eval_unseen_loss, eval_unseen_acc = eval_per_epoch(
            eval_unseen_loader, model, loss_fn)
        eval_unseen_loss /= len(eval_unseen_loader)
        eval_unseen_acc /= len(eval_unseen_datasets)
        print(f"Eval_UnSeen {(epoch + 1):03d} / {NUM_EPOCH:03d} :",
              f"loss = {eval_unseen_loss:.5f}, acc = {eval_unseen_acc:.5f}")
        output_str += f"\nEval {(epoch + 1):03d} / {NUM_EPOCH:03d} : loss = {eval_unseen_loss:.5f}, acc = {eval_unseen_acc:.5f}"

    print("All Epoch on Train and Eval were finished.\n")
    print(output_str)

    # Inference on test set
    torch.save(model.state_dict(), './tmp.pt')
    print("Save model was done.\n")


if __name__ == "__main__":
    main()
