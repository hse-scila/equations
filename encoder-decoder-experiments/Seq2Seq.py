import torch
import torch.nn as nn
import random

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, src_len, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        
        if self.encoder.rnn_type == "lstm":
            hidden, cell = self.encoder(src, src_len)
        else:
            hidden = self.encoder(src, src_len)
        
        input = trg[0, :]
       
        for t in range(1, trg_length):
            
            if self.encoder.rnn_type == "lstm":
                output, hidden, cell = self.decoder(input, hidden, cell)
            else:
                output, hidden = self.decoder(input, hidden)
           
            outputs[t] = output
           
            teacher_force = random.random() < teacher_forcing_ratio
            
            top1 = output.argmax(1)
            
            input = trg[t] if teacher_force else top1
            
        return outputs