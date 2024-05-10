import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

class Encoder(nn.Module):
    def __init__(
        self,
        rnn_type,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        dropout,
        device,
        pad_id = 0,
        bidirectional=False,
        one_hot=False,
        dumb=False
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.device = device
        self.one_hot = one_hot
        self.rnn_type = rnn_type
        self.dumb = dumb
        self.bididectional = bidirectional
        if not dumb:
            # print(input_dim, embedding_dim)
            if not one_hot:
                self.embedding = nn.Embedding(
                    input_dim, embedding_dim, padding_idx=pad_id
                )
            if rnn_type == "lstm":
                self.rnn = nn.LSTM(
                    embedding_dim,
                    hidden_dim,
                    n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            elif rnn_type == "rnn":
                self.rnn = nn.RNN(
                    embedding_dim,
                    hidden_dim,
                    n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            elif rnn_type == "gru":
                self.rnn = nn.GRU(
                    embedding_dim,
                    hidden_dim,
                    n_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                )
            self.dropout = nn.Dropout(dropout)

    def forward(self, src, src_len):
        if self.dumb:
            if self.rnn_type == "lstm":
                return (
                    torch.zeros((self.bididectional + 1) * self.n_layers, src.shape[1], self.hidden_dim),
                    torch.zeros((self.bididectional + 1) * self.n_layers, src.shape[1], self.hidden_dim)
                ).to(self.device)
            else:
                return (
                    torch.zeros((self.bididectional + 1) * self.n_layers, src.shape[1], self.hidden_dim)
                ).to(self.device)

        if not self.one_hot:
            embedded = self.dropout(self.embedding(src))
        else:
            embedded = torch.stack(
                [self.one_hot_encode(batch_sample, self.rnn.input_size) for batch_sample in src]
            ).to(self.device)

        packed_inputs = pack_padded_sequence(
            embedded, src_len.cpu(), enforce_sorted=False
        )

        if self.rnn_type == "lstm":
            _, (hidden, cell) = self.rnn(packed_inputs)
            return hidden, cell
        else:
            _, hidden = self.rnn(packed_inputs)
            return hidden
    
    @staticmethod
    def one_hot_encode(sequence, vocab_size):
        tensor = torch.zeros(len(sequence), int(vocab_size))
        tensor[torch.arange(len(sequence)), sequence] = 1
        return tensor