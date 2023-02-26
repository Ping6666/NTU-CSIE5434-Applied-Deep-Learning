import random
import numpy as np

import torch
from tqdm import trange, tqdm

from preprocess import predataset_workhouse, dataset_workhouse
from dataset import Hahow_Dataset
from model import Classifier

from average_precision import mapk

SEED = 5487

BATCH_SIZE = 128
NUM_WORKER = 8

NUM_EPOCH = 5
LR = 0.001

DROPOUT = 0.01
EMBED_SIZE = 2
HIDDEN_NUM = 128
F_NUM = 91

TOPK = 91

DEVICE = 'cuda:1'


def list_convertion(group_lists):
    c_group_lists = []
    for group_list in group_lists:
        k = (group_list != 0).sum()
        c_group_list = (group_list).tolist()[:k]
        # print(len(c_group_list))
        c_group_lists.append(c_group_list)

    return c_group_lists


def topk_convertion(group_lists, truncation=False):
    # if truncation:
    #     print(group_lists[0])

    c_group_lists = []
    for group_list in group_lists:
        k = len(group_list)
        if truncation:
            k = (group_list == 1).sum()
        c_group_list = (torch.topk(group_list, TOPK).indices + 1).tolist()[:k]
        c_group_lists.append(c_group_list)

    # if truncation:
    #     print(c_group_lists[0])

    return c_group_lists


def train_per_epoch(train_loader, model, optimizer, loss_fn, acc=False):
    train_loss, train_acc = 0, 0
    # _preds = []
    _preds, _gts = [], []
    for i, data in tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc='Train',
                        leave=False):
        # data collate_fn
        (_gender, _vector), _label, gt = data
        _gender = _gender.to(DEVICE)
        _vector = _vector.to(DEVICE)
        _label = _label.to(DEVICE)

        # train: data -> model -> loss
        y_pred = model(_gender, _vector)
        loss = loss_fn(y_pred, _label)  # eval loss from loss_fn

        if acc:
            _preds.extend(topk_convertion(y_pred))
            _gts.extend(list_convertion(gt))

        # update network
        optimizer.zero_grad()  # zero gradients
        loss.backward()
        optimizer.step()  # adjust learning weights

        # report: loss, acc.
        train_loss += loss.item()

    if acc:
        train_acc = mapk(_gts, _preds, TOPK)
    return train_loss, train_acc


def eval_per_epoch(eval_loader, model, loss_fn, acc=False) -> None:
    eval_loss, eval_acc = 0, 0
    _preds, _labels = [], []
    for i, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Evaluation',
                        leave=False):

        # data collate_fn
        (_gender, _vector), _label, _ = data
        _gender = _gender.to(DEVICE)
        _vector = _vector.to(DEVICE)
        _label = _label.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            y_pred = model(_gender, _vector)
            loss = loss_fn(y_pred, _label)  # eval loss from loss_fn

            if acc:
                _preds.extend(topk_convertion(y_pred))
                _labels.extend(topk_convertion(_label, True))

        # report: loss, acc.
        eval_loss += loss.item()

    if acc:
        eval_acc = mapk(_labels, _preds, TOPK)
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


def main():
    set_seed(SEED)

    print('***Model***')
    model = Classifier(DROPOUT, EMBED_SIZE, F_NUM, HIDDEN_NUM, F_NUM)
    model.to(DEVICE)

    print('***Data***')
    df_users, df_courses = predataset_workhouse()

    print('***Hahow_Dataset***')
    # crecate DataLoader for train / dev datasets
    train_datasets = Hahow_Dataset(dataset_workhouse(df_users, df_courses),
                                   'TTT')
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
                                                loss_fn, True)
        train_loss /= len(train_loader)
        train_acc /= len(train_datasets)
        c_train_output_str = f"Train {(epoch + 1):03d}/{NUM_EPOCH:03d} | loss = {train_loss:.5f}, acc = {train_acc:.5f}"

        # Evaluation loop
        model.eval()
        eval_seen_loss, eval_seen_acc = eval_per_epoch(eval_seen_loader, model,
                                                       loss_fn, True)
        eval_seen_loss /= len(eval_seen_loader)
        eval_seen_acc /= len(eval_seen_datasets)
        c_eval_seen_output_str = f"Eval_Seen {(epoch + 1):03d}/{NUM_EPOCH:03d} | loss = {eval_seen_loss:.5f}, acc = {eval_seen_acc:.5f}"

        eval_unseen_loss, eval_unseen_acc = eval_per_epoch(
            eval_unseen_loader, model, loss_fn, True)
        eval_unseen_loss /= len(eval_unseen_loader)
        eval_unseen_acc /= len(eval_unseen_datasets)
        c_eval_unseen_output_str = f"Eval_UnSeen {(epoch + 1):03d}/{NUM_EPOCH:03d} | loss = {eval_unseen_loss:.5f}, acc = {eval_unseen_acc:.5f}"

        print(f"\n{c_train_output_str}" + f"\n{c_eval_seen_output_str}" +
              f"\n{c_eval_unseen_output_str}")
        output_str += f"\n{c_train_output_str}" + f"\n{c_eval_seen_output_str}" + f"\n{c_eval_unseen_output_str}"

    print("All Epoch on Train and Eval were finished.\n")
    print(output_str)

    # Inference on test set
    torch.save(model.state_dict(), './tmp.pt')
    print("Save model was done.\n")


if __name__ == "__main__":
    main()
