import random
import numpy as np
from tqdm import trange, tqdm

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
from average_precision import mapk

## global ##

SEED = 5487
DEVICE = 'cuda:1'

BATCH_SIZE = 64
NUM_WORKER = 8

EMBED_SIZE = 2
FEATURE_NUM = 91
HIDDEN_NUM = 128
DROPOUT = 0.1

NUM_EPOCH = 50
LR = 0.001

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


def convert_ground_truths(gts):
    c_gts = []
    for gt in gts:
        k = (gt != 0).sum()
        c_gts.append((gt).tolist()[:k])
    return c_gts


## per_epoch ##


def train_per_epoch(train_loader, model, optimizer, loss_fn):
    train_loss, train_acc = 0, 0
    _y_preds, _subgroups = [], []
    for _, data in tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc='Train',
                        leave=False):
        # data collate_fn
        _, (_x_gender, _x_vector, _y), _subgroup = data
        _x_gender = _x_gender.to(DEVICE)
        _x_vector = _x_vector.to(DEVICE)
        _y = _y.to(DEVICE)

        # train: data -> model -> loss
        _y_pred = model(_x_gender, _x_vector)
        loss = loss_fn(_y_pred, _y)

        # update network (zero gradients -> backward ->  adjust learning weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # report: loss, acc.
        train_loss += loss.item()

        # accuracy
        c_y_pred = convert_predict_to_int_list(_y_pred)
        c_subgroup = convert_ground_truths(_subgroup)

        _y_preds.extend(c_y_pred)
        _subgroups.extend(c_subgroup)

    # report: loss, acc.
    train_loss /= len(train_loader)
    train_acc = mapk(_subgroups, _y_preds, TOPK)
    return train_loss, train_acc


def eval_per_epoch(eval_loader, model, loss_fn):
    eval_loss, eval_acc = 0, 0
    _y_preds, _subgroups = [], []
    for _, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Evaluation',
                        leave=False):
        # data collate_fn
        _, (_x_gender, _x_vector, _y), _subgroup = data
        _x_gender = _x_gender.to(DEVICE)
        _x_vector = _x_vector.to(DEVICE)
        _y = _y.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            _y_pred = model(_x_gender, _x_vector)
            loss = loss_fn(_y_pred, _y)

            # report: loss, acc.
            eval_loss += loss.item()

        # accuracy
        c_y_pred = convert_predict_to_int_list(_y_pred)
        c_subgroup = convert_ground_truths(_subgroup)

        _y_preds.extend(c_y_pred)
        _subgroups.extend(c_subgroup)

    # report: loss, acc.
    eval_loss /= len(eval_loader)
    eval_acc = mapk(_subgroups, _y_preds, TOPK)
    return eval_loss, eval_acc


## main ##


def main():
    set_seed(SEED)

    print('***Model***')
    model = Hahow_Model(EMBED_SIZE, FEATURE_NUM, HIDDEN_NUM, FEATURE_NUM,
                        DROPOUT)
    model.to(DEVICE)

    print('***Global***')
    global_init_workhouse()

    print('***Data***')
    df_preprocess = preprocess_workhouse()

    print('***Hahow_Dataset***')
    train_datasets = Hahow_Dataset(dataset_workhouse(df_preprocess, MODES[0]))
    eval_seen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[1]))
    eval_unseen_datasets = Hahow_Dataset(
        dataset_workhouse(df_preprocess, MODES[2]))

    print('***DataLoader***')
    train_loader = DataLoader(
        train_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )
    eval_seen_loader = DataLoader(
        eval_seen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )
    eval_unseen_loader = DataLoader(
        eval_unseen_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )

    print('***optimizer & loss function***')
    optimizer = torch.optim.Adam(model.parameters(), LR)
    loss_fn = torch.nn.MSELoss()

    output_str = ''

    print('***Train & Evaluation***')
    epoch_pbar = trange(NUM_EPOCH, desc='Epoch')
    for epoch in epoch_pbar:
        # Training loop
        model.train()
        train_loss, train_acc = train_per_epoch(
            train_loader,
            model,
            optimizer,
            loss_fn,
        )

        # Evaluation loop
        model.eval()
        eval_seen_loss, eval_seen_acc = eval_per_epoch(
            eval_seen_loader,
            model,
            loss_fn,
        )
        eval_unseen_loss, eval_unseen_acc = eval_per_epoch(
            eval_unseen_loader,
            model,
            loss_fn,
        )

        # output string
        epoch_str = f'\n{(epoch + 1):03d}/{NUM_EPOCH:03d}'
        train_str = f'\nTrain       | loss = {train_loss:.5f}, acc = {train_acc:.5f}'
        eval_seen_str = f'\nEval_Seen   | loss = {eval_seen_loss:.5f}, acc = {eval_seen_acc:.5f}'
        eval_unseen_str = f'\nEval_UnSeen | loss = {eval_unseen_loss:.5f}, acc = {eval_unseen_acc:.5f}'
        c_str = epoch_str + train_str + eval_seen_str + eval_unseen_str
        output_str += c_str
        print(c_str)

        # Inference on test set
        torch.save(model.state_dict(), f'./save/topic_{(epoch + 1):02d}.pt')
        print(f'Save model_{(epoch + 1):02d} was done.\n')

    print(output_str)
    print('All Epoch on Train and Eval were finished.\n')
    return


if __name__ == "__main__":
    main()
