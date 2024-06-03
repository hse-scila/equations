from tokenizers import Tokenizer, models, trainers
from tokenizers.processors import TemplateProcessing
import copy


#посимвольный ткенизатор с таким же интерфейсом, как и токенизаторы из библиотеки tokenizers
class CharacterTokenizer:
    def __init__(self, special_tokens):
        self.special_tokens = special_tokens
        self.char_to_idx = dict()
        self.idx_to_char = dict()
        self.pad_token = special_tokens[0]

    def add_special_tokens(self, special_tokens):
        for token in special_tokens:
            if token not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[token] = idx
                self.idx_to_char[idx] = token

    def token_to_id(self, token):
        return self.char_to_idx[token]

    def train_from_iterator(self, corpus, *args):
        unique_chars = set("".join(corpus))
        for char in unique_chars:
            if char not in self.char_to_idx:
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char

    def set_special_tokens(self, model_type, is_source=True):
        if model_type == 'encoder-decoder':
            if is_source:
                self.sos_token = '<sos_1>'
                self.eos_token = '<eos_1>'
            else:
                self.sos_token = '<sos_2>'
                self.eos_token = '<eos_2>'
        elif model_type == 'decoder-only':
            if is_source:
                self.sos_token = '<sos_1>'
                self.eos_token = ''
            else:
                self.sos_token = '<sos_2>'
                self.eos_token = '<eos_2>'
        self.pad_token = '<pad>'

    def encode(self, text, max_length=None, add_special_tokens=True):
        tokens = [self.char_to_idx[char] for char in text]
        if add_special_tokens:
            tokens = [self.char_to_idx[self.sos_token]] + tokens + [self.char_to_idx[self.eos_token]] if self.eos_token else []
        if max_length:
            tokens = tokens[:max_length] + [self.char_to_idx[self.pad_token]] * (max_length - len(tokens))
        return tokens

    def decode(self, tokens, skip_special_tokens=True):
        decoded_chars = [self.idx_to_char[token] for token in tokens if token in self.idx_to_char]
        if skip_special_tokens:
            decoded_chars = [char for char in decoded_chars if char not in self.special_tokens]
        return " ".join(decoded_chars)
    
    
#кастомный токенизатор, который в зависимости от tokenizer_type становится тем или иным токенизатором
class CustomTokenizer:
    def __init__(self, 
                 tokenizer_type='character',
                 vocab_size=None):
        
        self.tokenizer_type = tokenizer_type
        self.special_tokens = ['<pad>', '<sos_1>', '<eos_1>', '<sos_2>', '<eos_2>']
        self.vocab_size = vocab_size
        self.build_tokenizer()

    #получаем тип токенизатора, который нужен
    def build_tokenizer(self):
        if self.tokenizer_type == 'character':
            assert self.vocab_size is None, "vocab_size must be None for character-wise tokenizer initially."
            self.src_tokenizer = CharacterTokenizer(self.special_tokens)
            self.trg_tokenizer = CharacterTokenizer(self.special_tokens)
        elif self.tokenizer_type == 'bpe':
            self.src_tokenizer = Tokenizer(models.BPE())
            self.trg_tokenizer = Tokenizer(models.BPE())
        elif self.tokenizer_type == 'wordpiece':
            self.src_tokenizer = Tokenizer(models.WordPiece())
            self.trg_tokenizer = Tokenizer(models.WordPiece())
        elif self.tokenizer_type == 'sentencepiece':
            self.tokenizer = Tokenizer(models.Unigram())
            self.trg_tokenizer = Tokenizer(models.Unigram())
        else:
            raise ValueError("Unsupported tokenizer type")

    #определяем специальные токены, которые будут вставлятся в начало/конец последовательности
    def set_special_tokens(self, tokenizer, model_type, is_source=True):
        if self.tokenizer_type != 'character':
            if model_type == 'encoder-decoder':
                if is_source:
                    tokenizer.post_processor = TemplateProcessing(
                        single="<sos_1> $A <eos_1>",
                        special_tokens=[(i,tokenizer.token_to_id(i)) for i in self.special_tokens]
                        )
                    tokenizer.sos_token = '<sos_1>'
                    tokenizer.eos_token = '<eos_1>'
                else:
                    tokenizer.post_processor = TemplateProcessing(
                        single="<sos_2> $A <eos_2>",
                        special_tokens=[(i,tokenizer.token_to_id(i)) for i in self.special_tokens]
                    )
                    tokenizer.sos_token = '<sos_2>'
                    tokenizer.eos_token = '<eos_2>'
            elif model_type == 'decoder-only':
                if is_source:
                    tokenizer.post_processor = TemplateProcessing(
                        single="<sos_1> $A",
                        special_tokens=[(i,tokenizer.token_to_id(i)) for i in self.special_tokens]
                    )
                    tokenizer.sos_token = '<sos_1>'
                    tokenizer.eos_token = ''
                else:
                    tokenizer.post_processor = TemplateProcessing(
                        single="<sos_2> $A <eos_2>",
                        special_tokens=[(i,tokenizer.token_to_id(i)) for i in self.special_tokens]
                    )
                    tokenizer.sos_token = '<sos_2>'
                    tokenizer.eos_token = '<eos_2>'
        else:
            self.src_tokenizer.set_special_tokens(model_type, is_source=True)
            self.trg_tokenizer.set_special_tokens(model_type, is_source=False)

    #обучение токенизатора
    def train(self, tokenizer, corpus):
        if self.tokenizer_type == 'character':
            tokenizer.train_from_iterator(corpus)
        elif self.tokenizer_type == 'bpe':
            trainer = trainers.BpeTrainer(vocab_size=self.vocab_size)
            tokenizer.train_from_iterator(corpus, trainer=trainer)
        elif self.tokenizer_type == 'sentencepiece':
            trainer = trainers.UnigramTrainer(vocab_size=self.vocab_size)
            tokenizer.train_from_iterator(corpus, trainer=trainer)
        elif self.tokenizer_type == 'wordpiece':
            trainer = trainers.WordPieceTrainer(vocab_size=self.vocab_size)
            tokenizer.train_from_iterator(corpus, trainer=trainer)
        return tokenizer

    #метод, который делает все сразу - инициализирует, обучает и добавляет спец токены
    def fit(self, corpus, corpus_2=None, model_type='encoder-decoder'):
        
        print('training tokenizer...')
        self.src_tokenizer = self.train(self.src_tokenizer, corpus)
        if corpus_2 is not None:
            self.trg_tokenizer = self.train(self.trg_tokenizer, corpus_2)
        else:
            print('copying tokenizer...')
            self.trg_tokenizer = copy.deepcopy(self.src_tokenizer)
        if self.tokenizer_type=='character':
            self.src_vocab_size = len(self.src_tokenizer.char_to_idx)
            self.trg_vocab_size = len(self.trg_tokenizer.char_to_idx)
        else:
            self.src_vocab_size = self.vocab_size
            self.trg_vocab_size = self.vocab_size

        self.src_tokenizer.add_special_tokens(self.special_tokens)
        self.trg_tokenizer.add_special_tokens(self.special_tokens)

        self.src_vocab_size += len(self.special_tokens)
        self.trg_vocab_size += len(self.special_tokens)

        self.set_special_tokens(self.src_tokenizer, model_type, is_source=True)
        self.set_special_tokens(self.trg_tokenizer, model_type, is_source=False)

    #перевод текста в токены
    def tokenize(self, text, max_len=None, add_special_tokens=True, is_source=True):
        tokenizer = self.src_tokenizer if is_source else self.trg_tokenizer
        if self.tokenizer_type!= 'character':
            return tokenizer.encode(text, add_special_tokens=add_special_tokens).ids
        return tokenizer.encode(text, add_special_tokens=add_special_tokens)

    #перевод токенов в текст
    def detokenize(self, tokens, is_source=True, skip_special_tokens=True):
        tokenizer = self.src_tokenizer if is_source else self.trg_tokenizer
        return tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)
    
