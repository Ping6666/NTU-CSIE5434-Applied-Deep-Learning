import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import random
import numpy as np

import torch
from tqdm import trange, tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab

SEED = 5487
TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def train_per_epoch(train_loader, model, optimizer, loss_fn):
    train_loss, train_acc = 0, 0
    for i, data in tqdm(enumerate(train_loader),
                        total=len(train_loader),
                        desc='Train',
                        leave=False):
        # data collate_fn
        text_, intent_, _ = data  # id can be ignored (_: ignoring values)
        text_ = text_.to(args.device)
        intent_ = intent_.to(args.device)

        # train: data -> model -> loss
        pred_ = model(text_)
        loss = loss_fn(pred_, intent_)  # eval loss from loss_fn

        # print(pred_)
        # print(pred_.shape)
        # print(intent_)
        # print(intent_.shape)
        # print(loss)
        # input()

        # update network
        optimizer.zero_grad()  # zero gradients
        loss.backward()
        optimizer.step()  # adjust learning weights

        # report: loss, acc.
        _, pred_ = torch.max(pred_, 1)
        train_loss += loss.item()
        train_acc += (pred_ == intent_).sum().item()
    return train_loss, train_acc


def eval_per_epoch(eval_loader, model, loss_fn) -> None:
    eval_loss, eval_acc = 0, 0
    for i, data in tqdm(enumerate(eval_loader),
                        total=len(eval_loader),
                        desc='Evaluation',
                        leave=False):

        # data collate_fn
        text_, intent_, _ = data
        text_ = text_.to(args.device)
        intent_ = intent_.to(args.device)

        # eval: data -> model -> loss
        with torch.no_grad():
            pred_ = model(text_)
            loss = loss_fn(pred_, intent_)  # eval loss from loss_fn

        # report: loss, acc.
        _, pred_ = torch.max(pred_, 1)
        eval_loss += loss.item()
        eval_acc += (pred_ == intent_).sum().item()
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


def main(args):
    set_seed(SEED)

    output_str = ""
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {
        split: json.loads(path.read_text())
        for split, path in data_paths.items()
    }
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    # TODO_: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings=embeddings,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        num_class=len(intent2idx),
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
    loss_fn = torch.nn.CrossEntropyLoss(
        label_smoothing=0.2,  # 0.05, 0.1, 0.15, 0.2, 0.25, 0.3
    )

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    for epoch in epoch_pbar:
        # TODO_: Training loop - iterate over train dataloader and update model weights
        model.train()
        train_loss, train_acc = train_per_epoch(train_loader, model, optimizer,
                                                loss_fn)
        train_loss /= len(train_loader)
        train_acc /= len(datasets[TRAIN])
        output_str += f"\nTrain {(epoch + 1):03d} / {args.num_epoch:03d} : loss = {train_loss:.5f}, acc = {train_acc:.5f}"

        # TODO_: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        eval_loss, eval_acc = eval_per_epoch(eval_loader, model, loss_fn)
        eval_loss /= len(train_loader)
        eval_acc /= len(datasets[DEV])
        output_str += f"\nEval {(epoch + 1):03d} / {args.num_epoch:03d} : loss = {eval_loss:.5f}, acc = {eval_acc:.5f}"

    print("All Epoch on Train and Eval were finished.\n")
    print(output_str)

    # TODO_: Inference on test set
    torch.save(model.state_dict(), args.ckpt_dir / args.ckpt_name)
    print("Save model was done.\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./ckpt/intent/",
    )
    parser.add_argument("--ckpt_name", type=Path, default="best.pt")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)  # 512, 1024
    parser.add_argument("--num_layers", type=int, default=2)
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
    parser.add_argument("--num_epoch", type=int, default=100)  # 200, 500
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
