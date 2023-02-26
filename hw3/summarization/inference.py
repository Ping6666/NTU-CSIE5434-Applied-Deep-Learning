from argparse import ArgumentParser, Namespace

import jsonlines
from tqdm import tqdm

import torch
from torch.utils.data.dataloader import DataLoader

from datasets import load_dataset
from transformers import (
    MT5ForConditionalGeneration,
    AutoTokenizer,
)

from dataset import MT5Dataset


def main(args):
    print("*** from_pretrained ***")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir,
                                              use_fast=False)
    model = MT5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir)
    model.to(args.device)

    print("*** Dataset ***")
    dataset = load_dataset('json', data_files={
        'test': args.test_file,
    })
    test_dataset = MT5Dataset(dataset=dataset['test'],
                              tokenizer=tokenizer,
                              mode='test',
                              max_input=args.max_input,
                              max_output=args.max_output)

    print("*** DataLoader ***")
    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=args.batch_size,
                             shuffle=True)

    print("*** model load ***")
    model.load_state_dict(torch.load(args.pt_path))
    model.eval()

    print("*** Testing ***")
    with torch.no_grad():
        ids, pred_seqs = [], []
        for d in tqdm(test_loader):
            text_seq, text_mask = d['text_seq'].to(
                args.device), d['text_mask'].to(args.device)
            id = d['id']

            pred_seq = model.generate(
                input_ids=text_seq,
                attention_mask=text_mask,
                max_length=args.max_output,
                num_beams=args.num_beams,
                do_sample=args.do_sample,
                top_k=args.top_k,
                top_p=args.top_p,
                temperature=args.temperature,
            )

            for i, p in zip(id, pred_seq):
                _p = tokenizer.decode(p, skip_special_tokens=True)
                ids.append(i)
                pred_seqs.append(_p)

    with jsonlines.open(args.inference_file, "w") as f:
        for i, p in zip(ids, pred_seqs):
            f.write({"title": p, "id": i})

    return


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str)
    parser.add_argument("--pt_path", type=str)

    parser.add_argument("--test_file", type=str)
    parser.add_argument("--inference_file", type=str)

    parser.add_argument("--max_input", type=int, default=256)
    parser.add_argument("--max_output", type=int, default=64)

    parser.add_argument("--batch_size", type=int, default=16)

    parser.add_argument("--num_beams", type=int, default=5)  # 1, 2, 3, 5, 7
    parser.add_argument("--do_sample", action="store_true",
                        default=False)  # True, False
    parser.add_argument("--top_k", type=int,
                        default=150)  # 25, 50, 75, 100, 125, 150
    parser.add_argument("--top_p", type=float,
                        default=1.0)  # 0.75, 0.8, 0.85, 0.9, 0.95, 1.0
    parser.add_argument("--temperature", type=float,
                        default=1.0)  # 0.8, 1.0, 4.0

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    print(args)

    main(args)
