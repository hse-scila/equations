import torch
import tqdm as tqdm
import torch.nn as nn
from Tokenizers import detokenize
from Dataloaders import get_size


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.1, 0.1)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_fn(
    model, data_loader, optimizer, criterion, clip, teacher_forcing_ratio, device
):
    model.train()
    epoch_loss = 0
    for _, batch in enumerate(tqdm.tqdm(data_loader)):
        src = batch["eqs"].to(device)
        trg = batch["ans"].to(device)
        src_len = torch.tensor(batch["len_eqs"], dtype=torch.int64).to(device)
        # src = [src length, batch size]
        # trg = [trg length, batch size]
        optimizer.zero_grad()
        output = model(src, src_len, trg, teacher_forcing_ratio)
        # output = [trg length, batch size, trg vocab size]
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        # output = [(trg length - 1) * batch size, trg vocab size]
        trg = trg[1:].view(-1)
        # trg = [(trg length - 1) * batch size]
        loss = criterion(output, trg)
        # ignore_index=pad_toke_id
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def evaluate_fn(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for _, batch in enumerate(tqdm.tqdm(data_loader)):
            src = batch["eqs"].to(device)
            trg = batch["ans"].to(device)
            src_len = torch.tensor(batch["len_eqs"]).to(device)
            output = model(src, src_len, trg, 0) 
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(data_loader)

def predict(
    sentence,
    model,
    tokenizer,
    tokenizer_type,
    sentence_len=None,
    input_is_tokenized=False,
    device="cuda",
    max_output_length=200,
    only_decoder=False
):
    model.eval()
    with torch.no_grad():
        if input_is_tokenized:
            ids = sentence
        else:
            if tokenizer_type!= 'regexBPE':
                ids = tokenizer.encode(sentence).ids
                sentence_len = get_size(ids, tokenizer.encode("<pad>").ids[0])
            else:
                ids = tokenizer.encode(sentence)
                sentence_len = get_size(ids, tokenizer.encode("<pad>")[0])

        tensor = torch.LongTensor(ids).unsqueeze(-1).to(device)
        if only_decoder:
            if model.decoder.rnn_type == "lstm":
                hidden, cell = model.decoder(
                tensor, torch.tensor([sentence_len], dtype=torch.int64)
            )
            else:
                hidden = model.encoder(
                tensor, torch.tensor([sentence_len], dtype=torch.int64)
            )
        else:    
            if model.encoder.rnn_type == "lstm":
                hidden, cell = model.encoder(
                    tensor, torch.tensor([sentence_len], dtype=torch.int64)
                )
            else:
                hidden = model.encoder(
                    tensor, torch.tensor([sentence_len], dtype=torch.int64)
                )
        if tokenizer_type != 'regexBPE':
            if only_decoder:
                inputs = tokenizer.encode("<sep>").ids
            else:
                inputs = tokenizer.encode("<sos>").ids
        else:
            if only_decoder:
                inputs = tokenizer.encode("<sep>")
            else:
                inputs = tokenizer.encode("<sos>")
        # try:
        #     inputs = tokenizer.encode("<sos>").ids
        # except:
        #     inputs = tokenizer.encode("<sos>")
        for _ in range(max_output_length):
            inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
            if model.encoder.rnn_type == "lstm":
                output, hidden, cell = model.decoder(inputs_tensor, hidden, cell)
            else:
                output, hidden = model.decoder(inputs_tensor, hidden)
            predicted_token = output.argmax(-1).item()
            inputs.append(predicted_token)
            if tokenizer_type!= 'regexBPE':
                if predicted_token == tokenizer.encode("<eos>").ids[0]:
                    break
            else:
                if predicted_token == tokenizer.encode("<eos>")[0]:
                    break
        # tokens = detokenize(inputs, is_eq=False)[4:]
        tokens = detokenize(inputs, tokenizer)  # char
    return tokens