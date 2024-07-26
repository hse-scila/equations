import lightning.pytorch as pl
import torch
from lightning.pytorch.utilities import rank_zero_only
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.tensorboard.summary import hparams
from tokenization import CustomTokenizer
from dataset import TextDataModule
from model import Seq2Seq
import pandas as pd
import time
import ray
from ray import tune
from ray.tune import CLIReporter, TuneConfig
from ray.tune.integration.pytorch_lightning import TuneReportCallback
from ray.util import ActorPool
import itertools
from ray.train import RunConfig, ScalingConfig, CheckpointConfig
from ray.train.torch import TorchTrainer, TorchConfig
from ray.train.lightning import (
    RayDDPStrategy,
    RayLightningEnvironment,
    RayTrainReportCallback,
    prepare_trainer,
)
from ray.tune.search import Searcher

class TBLogger(TensorBoardLogger):
    @rank_zero_only
    def log_metrics(self, metrics, step):
        metrics.pop('epoch', None)
        return super().log_metrics(metrics, step)

#определяем функцию, котора фильтрует невалидные комбинации гиперпараметров (чтобы запускать сразу все эксперименты, которые нужны а не сабсетами)
def validate_combination(config):
    # Rule 1: There can't be decoder-only with shared embeddings
    if config["model_type"] == "decoder-only" and config["shared_embeddings"]:
        return False

    if config["model_type"] == "decoder-only" and config["init_decoder_from_encoder"]:
        return False
    
    if config["model_type"] == "decoder-only" and config["transform_encoder_hidden"]:
        return False
    
    if config["use_one_hot"] and config["shared_embeddings"]:
        return False
    
    if config["model_type"] == ['encoder-decoder'] and config["shared_embeddings"]:
        return False

    # Rule 2: There can't be decoder-only with attention
    if config["model_type"] == "decoder-only" and config["use_attention"]:
        return False

    # Rule 3: Emb size must be zero if use_one_hot is True
    if config["use_one_hot"] and config["emb_dim"] != 0:
        return False
    
    if not config["use_one_hot"] and config["emb_dim"] == 0:
        return False

    # Rule 4: For character tokenization vocab size must be None
    if config["tokenizer_type"] == "character" and config["vocab_size"] is not None:
        return False
    
    if config["tokenizer_type"] != "character" and config["vocab_size"] is None:
        return False

    # Rule 5: init_decoder_from_encoder can be True only for 'encoder-decoder'
    if config["init_decoder_from_encoder"] and config["model_type"] != "encoder-decoder":
        return False

    # Rule 6: transform_encoder_hidden can be True only for 'encoder-decoder'
    if config["transform_encoder_hidden"] and config["model_type"] != "encoder-decoder":
        return False

    # Rule 7: bidirectional can be True only for 'encoder-decoder'
    if config["bidirectional"] and config["model_type"] != "encoder-decoder":
        return False

    # Rule 8: decoder_input can include attention only when using attention
    if "attention" in config["decoder_input"] and not config["use_attention"]:
        return False

    # Rule 9: decoder_output can include attention only when using attention
    if "attention" in config["decoder_output"] and not config["use_attention"]:
        return False

    if config["scheduler_type"] is None and config['num_warmup_steps']>0:
        return False
    
    if not config['use_attention'] and 'attention' in config['decoder_input']:
        return False
    
    if not config['use_attention'] and 'attention' in config['decoder_output']:
        return False
    
    if config['use_attention'] and ('attention' not in config['decoder_input'] or 'attention' not in config['decoder_output']):
        return False
    
    if config['n_layers']==1 and config['dropout']>0:
        return False
    
    if not config['init_decoder_from_encoder'] and config['transform_encoder_hidden']:
        return False
    
    if not config['init_decoder_from_encoder'] and config["model_type"] == "encoder-decoder" and not ('attention' in config['decoder_output'] or 'attention' in config['decoder_input']):
        return False


    return True

#отбор валидный комбинаций с помощью предыдущей функции
def generate_valid_configs(grid=None):
    if grid is None:
        grid = {
        "tokenizer_type": ["character"],
        "vocab_size": [None],
        "model_type": ["encoder-decoder"],
        "batch_size": [5],
        "lr": [0.001, 0.01],
        "emb_dim": [0, 10],
        "shared_embeddings": [True],
        "hidden_dim": [8],
        "attention_dim":[4],
        "n_layers": [2],
        "dropout": [0.25],
        "scheduler_type": ["linear", None],
        "num_warmup_steps": [0, 100],
        "rnn_type": ["lstm"],
        "bidirectional": [False, True],
        "use_one_hot": [False, True],
        "use_attention": [False, True],
        "init_decoder_from_encoder": [True, False],
        "teacher_forcing_ratio": [1],
        "transform_encoder_hidden": [True, False],
        "decoder_input": ['emb_attention', 'emb'],
        "decoder_output": ['hidden_attention', 'hidden']
    }

    keys, values = zip(*grid.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    return [config for config in all_combinations if validate_combination(config)]

#специальный счетчик, который присваивает модели простой порядковый id (нужен именно такой, чтобы нормально работал с параллельными процессами)
@ray.remote
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1
        return self.count

#класс, который осуществляет выдачу наших гиперпараетров для каждой новой модели на очереди
@ray.remote
class CustomSearcher(Searcher):
    def __init__(self, search_space):
        super().__init__()
        self.search_space = search_space
        self.index = 0

    def suggest(self, trial_id):
        if self.index >= len(self.search_space):
            return None
        trial = self.search_space[self.index]
        self.index += 1
        return {'train_loop_config': trial}

    def on_trial_complete(self, trial_id, result=None, error=False, early_terminated=False):
        pass


class CustomSearcherWrapper(Searcher):
    def __init__(self, remote_searcher):
        super().__init__()
        self.remote_searcher = remote_searcher

    def suggest(self, trial_id):
        return ray.get(self.remote_searcher.suggest.remote(trial_id))

    def on_trial_complete(self, trial_id, result=None, error=False, early_terminated=False):
        ray.get(self.remote_searcher.on_trial_complete.remote(trial_id, result, error, early_terminated))



#функция обучения одной модели, которая получает на вход все необходимые гиперпараметры
def train_func(hparams):

    #читаем данные (вставить свой путь)
    dataframe = pd.read_csv('C:/users/toxas/eqs/generated_pairs.csv') #.iloc[:128,:] раскомменить для тестоого запуска
    #создаем и обучаем токенизатор в соответствии с гиперпараметрами
    tokenizer = CustomTokenizer(tokenizer_type=hparams['tokenizer_type'],
                                 vocab_size=hparams['vocab_size'])
    tokenizer.fit((dataframe.iloc[:,0]+dataframe.iloc[:,1]).tolist(),
                   model_type=hparams['model_type'])
    
    hparams['src_vocab_size'] = tokenizer.src_vocab_size
    hparams['trg_vocab_size'] = tokenizer.trg_vocab_size
    del hparams['vocab_size']
    
    #создаем модуль данных в соответсвии с шиперпараметрами
    data_module = TextDataModule(dataframe, 
                                 tokenizer, 
                                 model_type=hparams['model_type'], 
                                 batch_size=hparams['batch_size'], 
                                 use_cache=True, 
                                 val_split=0.1, 
                                 test_split=0.1)

    #создаем модель в соответствии с гиперпараметрами
    model = Seq2Seq(tokenizer,
                    hparams, device='cuda')


    #получаем id модели
    id = ray.get(counter.increment.remote())
    #определяем как будем сохранять (для каждой модели - лучшую эпоху)
    checkpoint_callback = ModelCheckpoint(
        monitor="loss/val_epoch",
        dirpath=f'C:/users/toxas/eqs/checkpoints/Seq2Seq_model/{id}', #путь куда сохранять модели
        filename='{epoch}.ckpt',
        save_top_k=1,
        mode='min',
        verbose=True,
    )

    #Определяем логгер, который будет записывать все наши метрики и строить графики 
    #(в нашем случае это тензорборд, но можно связывать и сдругими )в т.ч. онлайн платформами
    tensorboard_logger = TBLogger(f'C:/users/toxas/eqs/logs', #путь куда сохранять логи
                                  name='Seq2Seq_model',
                                   version=id, default_hp_metric=False, 
                                   log_graph=True
                                   )

    # Определяем трейнера, который будет обучать нашу модель - сам тренинг луп не нужен, все уже есть в классе модели 
    trainer = pl.Trainer(max_epochs=100, #для теста поставить 1 или 2 эпохи 
                         gradient_clip_val=1,
                         accelerator="gpu", devices=1,
                         callbacks=[checkpoint_callback], 
                         logger=tensorboard_logger, log_every_n_steps=1,
                         strategy=RayDDPStrategy(),
                         plugins=[RayLightningEnvironment()],
                         enable_progress_bar=False
                         )

    #обучаем модель (функция prepare trainer нужна именно для ray tune)
    trainer = prepare_trainer(trainer)
    trainer.fit(model, data_module)
 

    # Тестируем в конце на лучше эпохе
    best_model_path = checkpoint_callback.best_model_path
    trainer.test(ckpt_path=best_model_path, datamodule=data_module)


#непосредственно запуск экспериментов
if __name__ == "__main__":

    #определяем все параметры, которые хотим перебрать (невалидные комбинации отсеятся с помощью функций определенных выше)
    working_grid  = {
        "tokenizer_type": ["character"],
        "vocab_size": [None],
        "model_type": ["encoder-decoder", "decoder-only"],
        "batch_size": [64],
        "lr": [0.001, 0.0001],
        "emb_dim": [0, 64, 256],
        "shared_embeddings": [False, True],
        "hidden_dim": [64, 128, 256],
        'attention_dim': [128],
        "n_layers": [2, 4],
        "dropout": [0, 0.25],
        "scheduler_type": ["linear", None],
        "num_warmup_steps": [0, 500],
        "rnn_type": ["lstm", "gru"],
        "bidirectional": [False, True],
        "use_one_hot": [False, True],
        "use_attention": [False, True],
        "init_decoder_from_encoder": [True, False],
        "teacher_forcing_ratio": [1],
        "transform_encoder_hidden": [True, False],
        "decoder_input": ['emb_attention', 'emb'],
        "decoder_output": ['hidden_attention', 'hidden']
    }

    #прнтим всякую общую инфу вначале
    print("CUDA available: ", torch.cuda.is_available())
    print("CUDA device count: ", torch.cuda.device_count()) 
    valid_configs = generate_valid_configs(working_grid)
    print(f'Количество экспериментов: {len(valid_configs)}')
    ray.init(num_gpus = 1)
    print("Ray GPU resources: ", ray.available_resources().get("GPU", 0))
    #создаем счетчик моделей и переборщик гиперпараметров
    global counter
    counter = Counter.remote()
    print('Created Counter...')

    custom_searcher = CustomSearcher.remote(valid_configs)
    custom_searcher_wrapper = CustomSearcherWrapper(custom_searcher)
    print('Created searcher...')
    #пишем scaling config, который определяет сколько ресурсов будет выделятся одной задаче 
    #(в нашем случае 1 worker (т.е. одну модель будет обучать один процесс - для больших моделей можно задать несколько процессов, например, если батч не лезет в 1 GPU)
    #, 2 CPU и 0.1 GPU (в нашем случае модели маленькие относительно памяти карты A100 80GB на суперкомпе, поэтому это оптимально))
    scaling_config = ScalingConfig(
        num_workers=1,
        resources_per_worker={
            "GPU": 0.1,
            "CPU": 10}, #здесь имеются ввиду именно логические ядры, на обычном компе их не сильно больше, чем физических, а на суперкомпе - сильно 
                        #поэтому. запрсив 10 CPU, вы получите гораздо больше логических ядер, так что внимательнее с этим
            use_gpu=True)

    #задаем доп трейнер из ray (да знаю, много трейнеров, но зато можно параллелить)
    ray_trainer = TorchTrainer(
    torch_config = TorchConfig(backend="gloo"), #!!!!!!!!!!!! для линкуса или суперкомпа закоментировать эту строку - gloo для винды
    train_loop_per_worker = train_func,
    scaling_config = scaling_config
    )
    print('Created_Trianer...')
    #задаем тюнер
    tuner = tune.Tuner(
        ray_trainer,
        param_space = {
        "scaling_config": scaling_config, 
        'train_loop_config' : {}
        },
        tune_config=TuneConfig(search_alg=custom_searcher_wrapper,
                                num_samples=len(valid_configs), max_concurrent_trials=10, #максимально возможное количество одновременно обучаемых моделей
    ))
    #запускаем эксперименты и уходим на несколько дней - главное предусмотреть потребление ресурсов
    #модели будут сохранятся в папку указанную ModelCheckpoint, а метрики в папку указанную tensorboard_logger. 
    # Можно в любой момент (в т. ч. во время обучения) запустить tensorboard и посмотреть на графики
    print('Created_Tuner...')
    results = tuner.fit()
    ray.shutdown()

# Shutdown Ray
