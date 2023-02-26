import random
import numpy as np

import torch
from tqdm import trange, tqdm

from preprocess import get_dataset
from dataset import Hahow_Dataset
from model import Classifier

SEED = 5487

BATCH_SIZE = 64
NUM_WORKER = 8

NUM_EPOCH = 25
LR = 0.001
DROPOUT = 0.1
HIDDEN_NUM = 128

DEVICE = 'cuda:0'


def train_per_epoch(train_loader, model, optimizer, loss_fn):
    train_loss, train_acc = 0, 0
    for i, data in tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc='Train',
                        leave=False):
        # data collate_fn
        _input, _label = data
        _input = _input.to(DEVICE)
        _label = _label.to(DEVICE)

        print(_label)

        for l in _label:
            for i in l:
                print(i.item(), end=' ')
            print()
            input()

        # train: data -> model -> loss
        y_pred = model(_input)
        loss = loss_fn(y_pred, _label)  # eval loss from loss_fn

        # update network
        optimizer.zero_grad()  # zero gradients
        loss.backward()
        optimizer.step()  # adjust learning weights

        # report: loss, acc.
        # _, y_pred = torch.max(y_pred, 1)
        train_loss += loss.item()
        # train_acc += (pred_ == intent_).sum().item()
    return train_loss, train_acc


def eval_per_epoch(eval_loader, model, loss_fn) -> None:
    eval_loss, eval_acc = 0, 0
    for i, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Evaluation',
                        leave=False):

        # data collate_fn
        _input, _label = data
        _input = _input.to(DEVICE)
        _label = _label.to(DEVICE)

        # eval: data -> model -> loss
        with torch.no_grad():
            y_pred = model(_input)
            loss = loss_fn(y_pred, _label)  # eval loss from loss_fn

        # report: loss, acc.
        # _, y_pred = torch.max(y_pred, 1)
        eval_loss += loss.item()
        # eval_acc += (pred_ == intent_).sum().item()
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
    model = Classifier(DROPOUT, 91, HIDDEN_NUM, 91)
    model.to(DEVICE)

    print('***Hahow_Dataset***')
    # TODO_: crecate DataLoader for train / dev datasets
    train_datasets = Hahow_Dataset(get_dataset())

    print('***DataLoader***')
    train_loader = torch.utils.data.DataLoader(
        train_datasets,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKER,
        shuffle=True,
    )

    print('***optimizer & loss_fn***')
    # TODO_: init optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), LR)  # or use SGD
    loss_fn = torch.nn.MSELoss()

    output_str = ''

    epoch_pbar = trange(NUM_EPOCH, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO_: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss, train_acc = train_per_epoch(train_loader, model, optimizer,
                                                loss_fn)
        train_loss /= len(train_loader)
        train_acc /= len(train_datasets)
        print(
            f"\nTrain {(epoch + 1):03d} / {NUM_EPOCH:03d} : loss = {train_loss:.5f}, acc = {train_acc:.5f}"
        )
        output_str += f"\nTrain {(epoch + 1):03d} / {NUM_EPOCH:03d} : loss = {train_loss:.5f}, acc = {train_acc:.5f}"

        # TODO_: Evaluation loop - calculate accuracy and save model weights
        # model.eval()
        # eval_loss, eval_acc = eval_per_epoch(eval_loader, model, loss_fn)
        # eval_loss /= len(train_loader)
        # eval_acc /= len(datasets)
        # output_str += f"\nEval {(epoch + 1):03d} / {NUM_EPOCH:03d} : loss = {eval_loss:.5f}, acc = {eval_acc:.5f}"

    print("All Epoch on Train and Eval were finished.\n")
    print(output_str)

    # TODO_: Inference on test set
    torch.save(model.state_dict(), './tmp.pt')
    print("Save model was done.\n")


if __name__ == "__main__":
    main()
