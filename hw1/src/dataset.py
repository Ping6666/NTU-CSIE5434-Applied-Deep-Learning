from typing import List, Dict

from torch import LongTensor
from torch.utils.data import Dataset

from utils import Vocab, pad_to_len


class SeqClsDataset(Dataset):

    def __init__(
        self,
        data: List[Dict],
        vocab: Vocab,
        label_mapping: Dict[str, int],
        max_len: int,
    ):
        self.data = data
        self.vocab = vocab
        self.label_mapping = label_mapping
        self._idx2label = {
            idx: intent
            for intent, idx in self.label_mapping.items()
        }
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index) -> Dict:
        instance = self.data[index]
        return instance

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)

    def collate_fn(self, samples: List[Dict]) -> Dict:
        # TODO_: implement collate_fn
        texts, intents, ids = [], [], []

        ## Tensor ##
        # torch.LongTensor: CPU tensor 64-bit integer (signed)
        # torch.cuda.LongTensor: GPU tensor 64-bit integer (signed)

        texts = self.vocab.encode_batch(
            [sample["text"].split() for sample in samples])
        if "intent" in samples[0].keys():
            intents = [self.label2idx(sample["intent"]) for sample in samples]
        ids = [sample["id"] for sample in samples]

        # sequence is important
        return LongTensor(texts), LongTensor(intents), ids

    def label2idx(self, label: str):
        return self.label_mapping[label]

    def idx2label(self, idx: int):
        return self._idx2label[idx]


class SeqTaggingClsDataset(SeqClsDataset):
    ignore_idx = -100

    def collate_fn(self, samples):
        # TODO_: implement collate_fn
        tokens, tags, ids, lengths = [], [], [], []

        # for pack_padded_sequence and pad_packed_sequence in model
        samples.sort(key=lambda x: len(x['tokens']), reverse=True)

        tokens = self.vocab.encode_batch(
            [[token for token in sample["tokens"]] for sample in samples],
            to_len=self.max_len,
        )
        if "tags" in samples[0].keys():
            tags = [[self.label2idx(tag) for tag in sample["tags"]]
                    for sample in samples]
            # padding w/ 'O' to max_len
            tags = pad_to_len(
                tags,
                to_len=self.max_len,
                padding=self.label2idx('O'),  # ignore_idx
            )
        ids = [sample["id"] for sample in samples]
        lengths = [len(sample["tokens"]) for sample in samples]

        # sequence is important
        return LongTensor(tokens), LongTensor(tags), ids, lengths
