# downlaod cache and config, special_tokens_map, tokenizer_config, tokenizer, vocab

from argparse import ArgumentParser, Namespace

from transformers import (
    AutoTokenizer,
    MT5ForConditionalGeneration,
)


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path,
                                              cache_dir=args.cache_dir)
    model = MT5ForConditionalGeneration.from_pretrained(
        args.model_name_or_path, cache_dir=args.cache_dir)
    return


def parse_args() -> Namespace:
    parser = ArgumentParser()

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--cache_dir", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
