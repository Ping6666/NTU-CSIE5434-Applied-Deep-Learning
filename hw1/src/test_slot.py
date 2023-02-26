import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SeqTaggingClsDataset
from model import SeqTagger
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    slot_idx_path = args.cache_dir / "tag2idx.json"
    slot2idx: Dict[str, int] = json.loads(slot_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTaggingClsDataset(data, vocab, slot2idx, args.max_len)

    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagger(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
        nn_name=args.nn_name,
    )
    model.eval()

    # TODO_: crecate DataLoader for test dataset

    test_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        shuffle=False,
    )

    # TODO_: load weights into model
    ckpt = torch.load(args.ckpt_path)
    model.load_state_dict(ckpt)
    model.to(args.device)

    # TODO_: predict dataset
    predictions = []
    for i, data in tqdm(enumerate(test_loader),
                        total=len(test_loader),
                        desc='Test'):
        # data collate_fn
        text_, _, id_, length_ = data
        text_ = text_.to(args.device)

        # test: data -> model -> label
        with torch.no_grad():
            # pred_ = model(text_)
            model_in = {'batch': text_, 'length': length_}
            pred_ = model(model_in, args.max_len)
            _, pred_ = torch.max(pred_, 1)

        pred_slot_ = [[dataset.idx2label(int(p_slot)) for p_slot in pred][:l]
                      for pred, l in zip(pred_, length_)]
        predictions += list(zip(id_, pred_slot_))
    print("Test was finished.\n")

    # TODO_: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as f:
        f.write("id,tags\n")
        for id, slot in predictions:
            f.write(f"{id},{' '.join(slot)}\n")
    print("Writing file was finished.\n")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True,
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True,
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)
    parser.add_argument("--nn_name", type=str, default='LSTM')

    # data loader
    parser.add_argument("--batch_size", type=int, default=128)

    parser.add_argument("--device",
                        type=torch.device,
                        help="cpu, cuda, cuda:0, cuda:1",
                        default="cuda:0")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)