import pandas as pd
from datasets import Dataset, DatasetDict
import Tokenizers
import tqdm as tqdm
import torch
import torch.nn as nn

def get_size(sequence, pad):
    for i in range(len(sequence)):
        if sequence[i] == pad:
            return i + 1
    return len(sequence)


def get_loaders(filename, tokenizer_type, vocab_size, batch_size, only_decoder=False):
    
    df = pd.read_csv(filename, sep=',')

    if tokenizer_type=='BPE':
        tokenizer = Tokenizers.get_bpe_tokenizer(df, vocab_size)
    elif tokenizer_type=='byteBPE':
        tokenizer = Tokenizers.get_byte_bpe_tokenizer(df, vocab_size)
    elif tokenizer_type=='regexBPE':
        tokenizer = Tokenizers.get_regex_tokenizer(df, vocab_size)
    else:
        print('WTF BRO')
    
    eqs_lines, ans_lines = df["equation"], df["answer"]

    if tokenizer_type!='regexBPE':
        pad_id = tokenizer.encode("<pad>").ids[0]
        if only_decoder:
            eqs_tokenized = eqs_lines.apply(lambda line: tokenizer.encode(line + "<sep>").ids)
            ans_tokenized = ans_lines.apply(lambda line: tokenizer.encode("<sep>" + line).ids)
        else:
            eqs_tokenized = eqs_lines.apply(lambda line: tokenizer.encode("<sos>" + line + "<eos>").ids)
            ans_tokenized = ans_lines.apply(lambda line: tokenizer.encode("<sos>" + line + "<eos>").ids)
    else:
        pad_id = tokenizer.encode("<pad>")[0]
        if only_decoder:
            eqs_tokenized = eqs_lines.apply(lambda line: tokenizer.encode("<sos>" + line + "<sep>"))
            ans_tokenized = ans_lines.apply(lambda line: tokenizer.encode("<sep>" + line + "<eos>"))
        else:
            eqs_tokenized = eqs_lines.apply(lambda line: tokenizer.encode("<sos>" + line + "<eos>"))
            ans_tokenized = ans_lines.apply(lambda line: tokenizer.encode("<sos>" + line + "<eos>"))
    
    seq_length = max(
        eqs_tokenized.apply(lambda x: len(x)).max(), ans_tokenized.apply(lambda x: len(x)).max()
    ) 

    
    eqs_tokenized = eqs_tokenized.apply(lambda line: Tokenizers.pad(line, seq_length, pad_id))
    ans_tokenized = ans_tokenized.apply(lambda line: Tokenizers.pad(line, seq_length, pad_id))
    
    
    dataset_dict = {"eqs": eqs_tokenized, "ans": ans_tokenized}
    dataset = Dataset.from_dict(dataset_dict)
    
    train_testvalid = dataset.train_test_split(test_size=0.2)
    test_valid = train_testvalid["test"].train_test_split(test_size=0.5)
    train_test_valid_dataset = DatasetDict(
        {
            "train": train_testvalid["train"],
            "test": test_valid["test"],
            "valid": test_valid["train"],
        }
    )
    
    data_type = "torch"
    format_columns = ["eqs", "ans"]
    
    train_data = train_test_valid_dataset["train"].with_format(
        type=data_type, columns=format_columns, output_all_columns=True
    )
    
    valid_data = train_test_valid_dataset["valid"].with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True,
    )
    
    test_data = train_test_valid_dataset["test"].with_format(
        type=data_type,
        columns=format_columns,
        output_all_columns=True,
    )
    
    train_data = train_data.add_column(
        "len_eqs", [get_size(seq, pad_id) for seq in list(train_data["eqs"])]
    )
    train_data = train_data.add_column(
        "len_ans", [get_size(seq, pad_id) for seq in list(train_data["ans"])]
    )
    
    valid_data = valid_data.add_column(
        "len_eqs", [get_size(seq, pad_id) for seq in list(valid_data["eqs"])]
    )
    valid_data = valid_data.add_column(
        "len_ans", [get_size(seq, pad_id) for seq in list(valid_data["ans"])]
    )
    
    test_data = test_data.add_column(
        "len_eqs", [get_size(seq, pad_id) for seq in list(test_data["eqs"])]
    )
    test_data = test_data.add_column(
        "len_ans", [get_size(seq, pad_id) for seq in list(test_data["ans"])]
    )
    
    train_data_loader = get_data_loader(train_data, batch_size, pad_id, shuffle=True)
    valid_data_loader = get_data_loader(valid_data, batch_size, pad_id)
    test_data_loader = get_data_loader(test_data, batch_size, pad_id)

    val_references = [
        Tokenizers.detokenize(sentence.tolist(), tokenizer)
        for sentence in tqdm.tqdm(valid_data["ans"])
    ]
    test_references = [
        Tokenizers.detokenize(sentence.tolist(), tokenizer)
        for sentence in tqdm.tqdm(test_data["ans"])
    ]

    return train_data_loader, valid_data_loader, test_data_loader, tokenizer, val_references, test_references, pad_id


def get_collate_fn(pad_index):
    def collate_fn(batch):
        batch_eqs = [example["eqs"] for example in batch]
        batch_ans = [example["ans"] for example in batch]
        batch_eqs = nn.utils.rnn.pad_sequence(batch_eqs, padding_value=pad_index)
        batch_ans = nn.utils.rnn.pad_sequence(batch_ans, padding_value=pad_index)
        batch = {
            "eqs": batch_eqs,
            "ans": batch_ans,
            "len_eqs": [example["len_eqs"] for example in batch],
            "len_ans": [example["len_ans"] for example in batch],
        }
        return batch

    return collate_fn

def get_data_loader(dataset, batch_size, pad_index, shuffle=False):
    collate_fn = get_collate_fn(pad_index)
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
    )
    return data_loader