import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import random
import numpy as np

import torch
from tqdm import tqdm, trange

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab

from seqeval.metrics import classification_report
from seqeval.scheme import IOB2

SEED = 5487
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def train_per_epoch(train_loader, model, optimizer, loss_fn):
    train_loss, train_token_acc, train_joint_acc = 0, 0, 0
    for i, data in tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc='Train',
                        leave=False):
        # data collate_fn
        text_, slot_, _, length_ = data  # id can be ignored (_: ignoring values)
        text_ = text_.to(args.device)
        slot_ = slot_.to(args.device)

        # train: data -> model -> loss
        # pred_ = model(text_)
        model_in = {'batch': text_, 'length': length_}
        pred_ = model(model_in, args.max_len)

        # loss = torch.tensor(0.0, device=args.device)
        # for c_pred, c_slot, c_length in zip(pred_, slot_, length_):
        #     # eval loss from loss_fn
        #     # loss += loss_fn(c_pred[:c_length], c_slot[:c_length])
        #     loss += loss_fn(c_pred.permute(1, 0), c_slot)
        # loss /= len(slot_)
        loss = loss_fn(pred_, slot_)  # eval loss from loss_fn

        # update network
        optimizer.zero_grad()  # zero gradients
        loss.backward()
        optimizer.step()  # adjust learning weights
        # scheduler.step(loss)

        # report: loss, acc.
        _, pred_ = torch.max(pred_, 1)
        train_loss += loss.item()
        # compute token & joint acc
        for c_pred, c_slot, c_length in zip(pred_, slot_, length_):
            token_hit = (c_pred[:c_length] == c_slot[:c_length]).sum().item()
            train_token_acc += token_hit
            train_joint_acc += int(token_hit == c_length)
    return train_loss, train_token_acc, train_joint_acc


def eval_per_epoch(eval_loader, model, loss_fn):
    eval_loss, eval_token_acc, eval_joint_acc = 0, 0, 0
    for i, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Evaluation',
                        leave=False):
        # data collate_fn
        text_, slot_, _, length_ = data
        text_ = text_.to(args.device)
        slot_ = slot_.to(args.device)

        # eval: data -> model -> loss
        with torch.no_grad():
            # pred_ = model(text_)
            model_in = {'batch': text_, 'length': length_}
            pred_ = model(model_in, args.max_len)

            # loss = torch.tensor(0.0, device=args.device)
            # for c_pred, c_slot, c_length in zip(pred_, slot_, length_):
            #     # eval loss from loss_fn
            #     # loss += loss_fn(c_pred[:c_length], c_slot[:c_length])
            #     loss += loss_fn(c_pred.permute(1, 0), c_slot)
            # loss /= len(slot_)
            loss = loss_fn(pred_, slot_)  # eval loss from loss_fn

        # report: loss, acc.
        _, pred_ = torch.max(pred_, 1)
        eval_loss += loss.item()
        # compute token & joint acc
        for c_pred, c_slot, c_length in zip(pred_, slot_, length_):
            token_hit = (c_pred[:c_length] == c_slot[:c_length]).sum().item()
            eval_token_acc += token_hit
            eval_joint_acc += int(token_hit == c_length)
    return eval_loss, eval_token_acc, eval_joint_acc


def final_eval(eval_loader, model, dataset):
    a_pred_, a_slot_, a_length_ = [], [], []
    for i, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Eval'):
        # data collate_fn
        text_, slot_, _, length_ = data
        a_slot_ += slot_
        a_length_ += length_
        text_ = text_.to(args.device)
        slot_ = slot_.to(args.device)

        # eval: data -> model -> loss
        with torch.no_grad():
            # pred_ = model(text_)
            model_in = {'batch': text_, 'length': length_}
            pred_ = model(model_in, args.max_len)

        # report: loss, acc.
        _, pred_ = torch.max(pred_, 1)

        a_pred_ += pred_

    a_true_slot_ = [[dataset.idx2label(int(ss_true)) for ss_true in s_true][:l]
                    for s_true, l in zip(a_slot_, a_length_)]
    a_pred_slot_ = [[dataset.idx2label(int(p_slot)) for p_slot in pred][:l]
                    for pred, l in zip(a_pred_, a_length_)]
    # print(a_true_slot_)
    # print(len(a_true_slot_))
    # print(a_pred_slot_)
    # print(len(a_pred_slot_))
    
    report = classification_report(a_true_slot_, a_pred_slot_, scheme=IOB2, mode='strict')
    print(report)
    return


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


def main(args):
    set_seed(SEED)

    output_str = ""
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {
        split: json.loads(path.read_text())
        for split, path in data_paths.items()
    }
    datasets: Dict[str, SeqTaggingClsDataset] = {
        split: SeqTaggingClsDataset(split_data, vocab, tag2idx, args.max_len)
        for split, split_data in data.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO_: init model and move model to target device(cpu / gpu)
    model = SeqTagger(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(tag2idx),
        nn_name=args.nn_name,
    )
    model = model.to(args.device)

    # TODO_: crecate DataLoader for train / dev datasets
    train_loader = torch.utils.data.DataLoader(
        datasets[TRAIN],
        batch_size=args.batch_size,
        collate_fn=datasets[TRAIN].collate_fn,
        num_workers=args.num_workers,
        shuffle=True,
    )
    eval_loader = torch.utils.data.DataLoader(
        datasets[DEV],
        batch_size=args.batch_size,
        collate_fn=datasets[DEV].collate_fn,
        num_workers=args.num_workers,
        shuffle=False,
    )

    # TODO_: init optimizer, loss function
    optimizer = torch.optim.Adam(model.parameters(), args.lr)  # or use SGD
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
    #                                                        mode='min',
    #                                                        factor=0.1,
    #                                                        patience=2)

    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=0.2,  # 0.05, 0.1, 0.15
    )

    train_token_len = sum(
        [len for c_data in train_loader for len in c_data[-1]])
    eval_token_len = sum([len for c_data in eval_loader for len in c_data[-1]])

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO_: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss, train_token_acc, train_joint_acc = train_per_epoch(
            train_loader, model, optimizer, loss_fn)
        train_loss /= len(train_loader)
        train_token_acc /= train_token_len
        train_joint_acc /= len(datasets[TRAIN])
        c_output_str = f"\nTrain {(epoch + 1):03d} / {args.num_epoch:03d} : loss = {train_loss:.5f}, token acc = {train_token_acc:.5f}, joint acc = {train_joint_acc:.5f}"
        print(c_output_str)
        output_str += c_output_str

        # TODO_: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        eval_loss, eval_token_acc, eval_joint_acc = eval_per_epoch(
            eval_loader, model, loss_fn)
        eval_loss /= len(train_loader)
        eval_token_acc /= eval_token_len
        eval_joint_acc /= len(datasets[DEV])
        c_output_str = f"\nEval {(epoch + 1):03d} / {args.num_epoch:03d} : loss = {eval_loss:.5f}, token acc = {eval_token_acc:.5f}, joint acc = {eval_joint_acc:.5f}"
        print(c_output_str)
        output_str += c_output_str

    print("All Epoch on Train and Eval were finished.\n")
    print(output_str)

    # TODO_: Inference on test set
    torch.save(model.state_dict(), args.ckpt_dir / args.ckpt_name)
    print("Save model was done.\n")

    print("\n\n\n\n")
    final_eval(eval_loader, model, datasets[DEV])


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/slot/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/slot/",
    )
    parser.add_argument("--ckpt_name", type=Path, default="best.pt")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)  # 512,
    parser.add_argument("--num_layers", type=int, default=2)  # 2,
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--nn_name", type=str, default='LSTM')

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=8)  # 12

    # training
    parser.add_argument("--device",
                        type=torch.device,
                        help="cpu, cuda, cuda:0, cuda:1",
                        default="cuda:0")
    parser.add_argument("--num_epoch", type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)