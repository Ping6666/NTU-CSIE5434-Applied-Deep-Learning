from torch.utils.data import Dataset


class MT5Dataset(Dataset):

    def __init__(self, dataset, tokenizer, mode, max_input=512, max_output=64):
        self.dataset = dataset
        self.tokenizer = tokenizer

        self.mode = mode

        self.max_input = max_input
        self.max_output = max_output
        return

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        c_dataset = self.dataset[index]

        text = self.tokenizer(
            [c_dataset["text"]],
            max_length=self.max_input,
            padding='max_length',
            truncation=True,
            return_tensors="pt",
        )
        text_seq = text["input_ids"].squeeze()
        text_mask = text["attention_mask"].squeeze()

        summary_seq = summary_mask = []
        if self.mode != 'test':
            with self.tokenizer.as_target_tokenizer():
                # label
                summary = self.tokenizer(
                    [c_dataset["summary"]],
                    max_length=self.max_output,
                    padding='max_length',
                    truncation=True,
                    return_tensors="pt",
                )
            summary_seq = summary["input_ids"].squeeze()
            summary_mask = summary["attention_mask"].squeeze()

        c_item = {
            "id": self.dataset[index]['id'],
            "text_seq": text_seq,
            "text_mask": text_mask,
            "summary_seq": summary_seq,
            "summary_mask": summary_mask,
        }
        return c_item
