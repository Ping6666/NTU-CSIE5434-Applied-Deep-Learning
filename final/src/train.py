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
    inference_prediction,
)
from model import Hahow_Model
from dataset import Hahow_Dataset
from average_precision import mapk

## global ##

SEED = 5487
DEVICE = 'cuda:1'

BATCH_SIZE = 256
NUM_WORKER = 8

FEATURE_NUM = 91
HIDDEN_NUM = 256
DROPOUT = 0.1

NUM_EPOCH = 50
LR = 0.005

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
        k = (gt != -1).sum()
        c_gts.append((gt).tolist()[:k])
    return c_gts


## per_epoch ##


def per_epoch(mode: str,
              desc: str,
              loader: DataLoader,
              model: Hahow_Model,
              loss_fn: torch.nn.CosineEmbeddingLoss,
              optimizer: torch.optim.Optimizer = None):
    '''
    do things per epoch
    '''

    # variable
    total_topic_loss, total_course_loss = 0, 0
    _y_topic_preds, _y_course_preds, _gt_topics, _gt_courses = [], [], [], []

    # mode
    mode_fn = None
    if mode == 'train':
        mode_fn = torch.enable_grad
    elif mode == 'val':
        mode_fn = torch.no_grad

    # for all batch
    for _, data in tqdm(enumerate(loader), total=len(loader), desc=desc):
        # data collate_fn (user_id, x, y, ground_truth)
        _, _x_vector, (_y_topic, _y_course), (_gt_topic, _gt_course) = data
        _x_vector = _x_vector.to(DEVICE)
        _y_topic = _y_topic.to(DEVICE)
        _y_course = _y_course.to(DEVICE)

        with mode_fn():
            # train: data -> model -> loss
            _y_t_pred, _y_c_pred = model(_x_vector)

            # topic loss
            target = torch.ones(_y_t_pred.size(dim=0)).to(DEVICE)
            topic_loss: torch.Tensor = loss_fn(_y_t_pred, _y_topic, target)

            # course loss
            target = torch.ones(_y_c_pred.size(dim=0)).to(DEVICE)
            course_loss: torch.Tensor = loss_fn(_y_c_pred, _y_course, target)

            if mode == 'train':
                # update network (zero gradients -> backward ->  adjust learning weights)
                optimizer.zero_grad()

                # topic_loss.backward()

                topic_loss.backward(retain_graph=True)
                course_loss.backward()

                optimizer.step()

                # Tip: when call backward() multiple times, need to use `retain_graph=True` but not for the last time.
                # `retain_graph=True` will keep the buffer, and `retain_graph=False` will free the buffer.

            # report: loss, acc.
            total_topic_loss += topic_loss.item()
            total_course_loss += course_loss.item()

        # accuracy
        _y_topic_preds.extend(inference_prediction('topic', _y_t_pred))
        _y_course_preds.extend(inference_prediction('course', _y_c_pred))
        _gt_topics.extend(convert_ground_truths(_gt_topic))
        _gt_courses.extend(convert_ground_truths(_gt_course))

    # report: loss, acc.
    total_topic_loss /= len(loader)
    total_course_loss /= len(loader)
    train_subgroup_acc = mapk(_gt_topics, _y_topic_preds, TOPK)
    train_course_acc = mapk(_gt_courses, _y_course_preds, TOPK)
    return total_topic_loss, total_course_loss, train_subgroup_acc, train_course_acc


## main ##


def main():
    set_seed(SEED)
    output_str = ''

    print('***Global***')
    global_init_workhouse()

    print('***Data***')
    df_preprocess, topic_course_metrix = preprocess_workhouse()

    print('***Model***')
    model = Hahow_Model(topic_course_metrix, FEATURE_NUM, HIDDEN_NUM,
                        FEATURE_NUM, DROPOUT, DEVICE)
    model.to(DEVICE)

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

    print('***optimizer***')
    optimizer = torch.optim.Adam(model.parameters(), LR)

    print('***loss function***')
    # loss_fn = torch.nn.MSELoss()
    loss_fn = torch.nn.CosineEmbeddingLoss()

    print('***Train & Evaluation***')
    epoch_pbar = trange(NUM_EPOCH, desc='Epoch')
    for epoch in epoch_pbar:
        # Training loop
        model.train()
        train = per_epoch('train', 'Train', train_loader, model, loss_fn,
                          optimizer)

        # Evaluation loop
        model.eval()
        eval_seen = per_epoch('val', 'Eval_Seen', eval_seen_loader, model,
                              loss_fn)
        eval_unseen = per_epoch('val', 'Eval_UnSeen', eval_unseen_loader,
                                model, loss_fn)

        # output string
        c_str = (
            f'\n{(epoch + 1):03d}/{NUM_EPOCH:03d} | t is topic, c is course.' +
            f'\nTrain       | t_loss = {train[0]:.5f}, c_loss = {train[1]:.5f}, '
            + f't_acc = {train[2]:.5f}, c_acc = {train[3]:.5f}' +
            f'\nEval_Seen   | t_loss = {eval_seen[0]:.5f}, c_loss = {eval_seen[1]:.5f}, '
            + f't_acc = {eval_seen[2]:.5f}, c_acc = {eval_seen[3]:.5f}' +
            f'\nEval_UnSeen | t_loss = {eval_unseen[0]:.5f}, c_loss = {eval_unseen[1]:.5f}, '
            + f't_acc = {eval_unseen[2]:.5f}, c_acc = {eval_unseen[3]:.5f}')
        print(c_str)

        output_str += c_str

        # Inference on test set
        torch.save(model.state_dict(), f'./save/topic_{(epoch + 1):02d}.pt')
        print(f'Save model_{(epoch + 1):02d} was done.\n')

    print(output_str)
    print('All Epoch on Train and Eval were finished.\n')
    return


if __name__ == "__main__":
    main()
