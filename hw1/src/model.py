from typing import Dict

import torch


class SeqClassifier(torch.nn.Module):

    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        nn_name: str,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = torch.nn.Embedding.from_pretrained(
            embeddings,
            freeze=False,
        )

        self.rnn_input_size = embeddings.size(1)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.num_class = num_class

        # TODO_: model architecture

        self.nn_name = nn_name
        if self.nn_name == 'RNN':
            self.nn = torch.nn.RNN(
                input_size=self.rnn_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        elif self.nn_name == 'LSTM':
            self.nn = torch.nn.LSTM(
                input_size=self.rnn_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        elif self.nn_name == 'GRU':
            self.nn = torch.nn.GRU(
                input_size=self.rnn_input_size,
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                batch_first=True,
                dropout=self.dropout,
                bidirectional=self.bidirectional,
            )
        else:
            raise NameError

        self.seq = torch.nn.Sequential(
            torch.nn.Dropout(self.dropout),
            torch.nn.Linear(
                in_features=self.encoder_output_size,
                out_features=self.num_class,
            ),
        )

    @property
    def encoder_output_size(self) -> int:
        # TODO_: calculate the output dimension of nn
        size = self.hidden_size
        if self.bidirectional == True:
            size = 2 * self.hidden_size
        return size

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO_: implement model forward
        # print(batch.shape)
        batch_in = self.embed(batch)
        # print(batch_in.shape)
        nn_out, _ = self.nn(batch_in, None)
        # print(nn_out.shape)
        seq_out = self.seq(nn_out[:, -1, :])
        # print(seq_out.shape)
        return seq_out

    # test result is validated on kaggle

    ## RNN ##
    # bad performance on train stage

    ## LSTM ##
    # LSTM + Linear * 3 + Dropout * 2 + ReLU: 0.87422
    # LSTM + Linear * 3 + Dropout * 2: 0.88177
    # LSTM + Linear * 2 + Dropout * 2 + ReLU: 0.88888
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.30: 0.90222
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.25: 0.90622
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.20: 0.90400
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.15: 0.90088
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.10: 0.90000
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.05: 0.90488
    # LSTM + Linear * 2 + Dropout: 0.89511
    # LSTM + Linear * 2 + Dropout (larger) + epoch 500: 0.89511
    # LSTM + Linear * 1: 0.89244

    # LSTM + Linear * 1 + Dropout + label_smoothing 0.2: 0.90400 (vocab_size = 1000)
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.2: 0.91066 (vocab_size = 10000) # final

    # change on Linear hidden
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.25: 0.91333

    ## GRU ##
    # GRU + Linear * 2 + Dropout: 0.88711


class SeqTagger(SeqClassifier):

    def __init__(
        self,
        embeddings: torch.tensor,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        nn_name: str,
    ) -> None:
        super(SeqTagger,
              self).__init__(embeddings, hidden_size, num_layers, dropout,
                             bidirectional, num_class, nn_name)

    def forward(self, batch_forward, max_len) -> Dict[str, torch.Tensor]:
        # def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO_: implement model forward
        batch = batch_forward['batch']
        length = batch_forward['length']

        # print(batch.shape)
        batch_in = self.embed(batch)
        # print(batch_in.shape)

        # nn_out, _ = self.nn(batch_in, None)

        packed_batch_in = torch.nn.utils.rnn.pack_padded_sequence(
            batch_in, length, batch_first=True)
        # print(len(packed_batch_in))
        # print(packed_batch_in[0].shape)
        # print(packed_batch_in[1].shape)
        packed_batch_out, _ = self.nn(packed_batch_in, None)
        # print(len(packed_batch_out))
        # print(packed_batch_out[0].shape)
        # print(packed_batch_out[1].shape)
        nn_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
            packed_batch_out, batch_first=True, total_length=max_len)
        # print(nn_out.shape)

        seq_out = self.seq(nn_out)
        # print(seq_out.shape)
        seq_out = seq_out.permute(0, 2, 1)
        # print(seq_out.shape)
        return seq_out

    ## LSTM ##
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.10: 0.71420
    # LSTM + Linear * 2 + Dropout + label_smoothing 0.05: 0.71420

    # change on Linear hidden
    # LSTM + Linear + Dropout + PReLU + label_smoothing 0.2: 0.73083
    # GRU + Linear * 2 + Dropout * 2 + label_smoothing 0.2: 0.70402 (seed 5487)
    # LSTM + Linear * 2 + Dropout * 2 + label_smoothing 0.15: 0.71313 (seed 5487)
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.05: 0.72439
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.15: 0.73619 (seed 5487)

    # w/ pading in model
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.2: 0.71957

    # w/ pading in model change preprocess_slot vocab_size = 1000
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.2: 0.82841, 0.82412 # final

    # w/ pading in model change preprocess_slot vocab_size = 3000
    # LSTM + Linear * 1 + Dropout + label_smoothing 0.2: 0.81554
