import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(
        self,
        rnn_type,
        input_dim, 
        output_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        dropout,
        pad_id=0,
        bidirectional=False,
        one_hot=False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.one_hot = one_hot
        if not one_hot:
            self.embedding = nn.Embedding(
                input_dim, embedding_dim, padding_idx=pad_id
            )
        self.embedding = nn.Embedding(output_dim, embedding_dim)
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
        if bidirectional:
            self.fc_out = nn.Linear(2 * hidden_dim, output_dim)
        else:
            self.fc_out = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell=None):
        input = input.unsqueeze(0)
        # input = [1, batch size]
        if not self.one_hot:
            embedded = self.dropout(self.embedding(input))
        else:
            embedded = torch.stack(
                [self.one_hot_encode(batch_sample, self.rnn.input_size) for batch_sample in input]
            ).to(device)
        # embedded = [1, batch size, embedding dim]
        if self.rnn_type == "lstm":
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            prediction = self.fc_out(output.squeeze(0))
            return prediction, hidden, cell
        else:
            output, hidden = self.rnn(embedded, hidden)
            prediction = self.fc_out(output.squeeze(0))
            return prediction, hidden
    
    @staticmethod
    def one_hot_encode(sequence, vocab_size):
        tensor = torch.zeros(len(sequence), int(vocab_size))
        tensor[torch.arange(len(sequence)), sequence] = 1
        return tensor
    