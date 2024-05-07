from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer
from tokenizers import ByteLevelBPETokenizer
import numpy as np
import sys
sys.path.insert(1, "minbpe-master")
from minbpe import RegexTokenizer


def detokenize(sentence, tokenizer):
    answer = tokenizer.decode(sentence)
    if answer[:5]=='<sos>':
        answer = answer[5:]
    end_token_idx = answer.find("<eos>")
    answer = answer[:end_token_idx]
    answer = answer.replace("<pad>", "")
    return answer

def pad(line, seq_length, pad):
    if len(line) < seq_length:
        line = np.concatenate((line, [pad]*(seq_length-len(line))))
    return line

def batch_iterator(dataset, batch_size):
    for i in range(0, len(dataset), batch_size):
        yield dataset[i : i + batch_size]["equation"] + dataset[i : i + batch_size]["answer"]


def get_bpe_tokenizer(df, vocab_size):
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(special_tokens=["<sos>", "<pad>", "<eos>", "<sep>"], vocab_size=vocab_size)
    tokenizer.train_from_iterator(batch_iterator(df, 32), trainer)
    tokenizer.enable_padding(pad_id=tokenizer.encode("<pad>").ids[0], pad_token="<pad>")
    return tokenizer

def get_byte_bpe_tokenizer(df, vocab_size):
    tokenizer = ByteLevelBPETokenizer()
    tokenizer.train_from_iterator(batch_iterator(df, 32), vocab_size=259+vocab_size, special_tokens=["<sos>", "<pad>", "<eos>", "<sep>"])
    tokenizer.enable_padding(pad_id=tokenizer.encode("<pad>").ids[0], pad_token="<pad>")
    return tokenizer


def get_regex_tokenizer(df, vocab_size):
    
    lines = df["equation"] + " " + df["answer"]
    str_for_vocab_training = lines.str.cat(sep=" ")
    tokenizer = RegexTokenizer()
    # doesn't allow adding special tokens; make it learn <pad>
    tokenizer.train(str_for_vocab_training, vocab_size=256+vocab_size)
    tokenizer.register_special_tokens({
        '<sos>':len(tokenizer.vocab), 
        '<pad>':len(tokenizer.vocab) + 1, 
        '<eos>':len(tokenizer.vocab) + 2,
        '<sep>':len(tokenizer.vocab) + 3
        })
    
    return tokenizer 