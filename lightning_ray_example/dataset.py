import torch
from torch.utils.data import DataLoader, Dataset, random_split
import lightning.pytorch as pl

#обычный torch Dataset
class TextDataset(Dataset):
    def __init__(self, 
                 dataframe, 
                 tokenizer, 
                 model_type, 
                 use_cache=False):
        
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.use_cache = use_cache
        self.cache = {}
        self.src_pad_id =  tokenizer.src_tokenizer.token_to_id('<pad>')
        self.trg_pad_id =  tokenizer.trg_tokenizer.token_to_id('<pad>')

    #определяет длинну датасета (нужно, чтобы сэплировать индексы для формирования батча)
    def __len__(self):
        return len(self.dataframe)

    #метод, который по индексу получает непосредственно sample
    def __getitem__(self, idx):
        #кэширование, чтобы токенизация проводилась только в первую эпоху
        if self.use_cache and idx in self.cache:
            return self.cache[idx]

        src = self.dataframe.iloc[idx, 0]
        trg = self.dataframe.iloc[idx, 1]
        src_tokens = self.tokenizer.tokenize(src, add_special_tokens=True, is_source=True)
        trg_tokens = self.tokenizer.tokenize(trg, add_special_tokens=True, is_source=False)

        if self.model_type == 'decoder-only':
            combined_tokens = src_tokens + trg_tokens
            result = combined_tokens
        else:
            result = (src_tokens, trg_tokens)

        if self.use_cache:
            self.cache[idx] = result

        return result

    #метод для формирования батчей (padding + для decoder-only моделей конкатенация equation и answer с разделяющим символом между ними)
    def collate_fn(self, batch):
        if self.model_type == 'decoder-only':
            lengths = [len(tokens) for tokens in batch]
            max_len = max(lengths)
            padded_batch = [tokens + [self.trg_pad_id] * (max_len - len(tokens)) for tokens in batch]
            return torch.tensor(padded_batch), torch.tensor(lengths)
        else:
            src_batch, trg_batch = zip(*batch)
            src_lengths = [len(tokens) for tokens in src_batch]
            trg_lengths = [len(tokens) for tokens in trg_batch]
            max_src_len = max(src_lengths)
            max_trg_len = max(trg_lengths)
            padded_src_batch = [tokens + [self.src_pad_id] * (max_src_len - len(tokens)) for tokens in src_batch]
            padded_trg_batch = [tokens + [self.trg_pad_id] * (max_trg_len - len(tokens)) for tokens in trg_batch]
            return torch.tensor(padded_src_batch), torch.tensor(src_lengths), torch.tensor(padded_trg_batch), torch.tensor(trg_lengths)


#lightning DataModule - разбивает выборку на обучающую, валидационную и тестовую и возвращает соответвующие лоудеры
class TextDataModule(pl.LightningDataModule):
    def __init__(self, 
                 dataframe, 
                 tokenizer, 
                 model_type, 
                 batch_size=32,
                 use_cache=False,
                 val_split=0.1,
                 test_split=0.1,
                 seed=228,
                 num_workers=0):
        super().__init__()
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.model_type = model_type
        self.batch_size = batch_size
        self.use_cache = use_cache
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed
        self.num_workers = num_workers

    #просто разбиваем данные
    def setup(self, stage=None):

        full_dataset = TextDataset(self.dataframe, self.tokenizer, self.model_type, self.use_cache)
        test_size = int(self.test_split * len(full_dataset))
        val_size = int(self.val_split * len(full_dataset))
        train_size = len(full_dataset) - val_size - test_size

        self.train_dataset, self.val_dataset, self.test_dataset = random_split(full_dataset,
                                                                                [train_size, val_size, test_size],
                                                                                generator = torch.Generator().manual_seed(self.seed))
    #определяем как получаются даталоудеры (очень просто)
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, 
                          shuffle=True, collate_fn=self.train_dataset.dataset.collate_fn,
                          pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, 
                          shuffle=False, collate_fn=self.val_dataset.dataset.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, 
                          shuffle=False, collate_fn=self.test_dataset.dataset.collate_fn)
