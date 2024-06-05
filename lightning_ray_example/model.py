import torch
import torch.nn as nn
import lightning.pytorch as pl
from torch.optim import Adam
from torchmetrics.text import BLEUScore
from numba import jit
from argparse import Namespace
from lightning.pytorch.utilities import grad_norm
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, get_constant_schedule_with_warmup


#один способ считать attention
class AttentionLayerBilinear(nn.Module):
    def __init__(self,hid_size):
        """ Слой, подсчитывающий выходы аттеншена и веса """
        super().__init__()
        self.hid_size = hid_size # размер вектора скрытого состояния аттеншена 
        self.linear = nn.LazyLinear(hid_size)

    def forward(self, enc, dec, lengths):
        """
        Подсчитывает выход аттеншена и веса
        :param enc: входная последовательность кодировщика, float32[batch_size, ninp, enc_size]
        :param dec: выходная последовательность декодировщика (query), float32[batch_size, dec_size]
        :param inp_mask: маска для последовательностей кодировщика (0 после первого токена eos), float32 [batch_size, ninp]
        :returns: attn[batch_size, enc_size], probs[batch_size, ninp]
            - attn - вектор выхода аттеншена (взвешенная сумма enc).
            - probs - веса аттеншена после софтмакса (softmax)
        """
        
        # Подсчет логитов    
        # Наложение маски - если mask равна 0, логиты должны быть -inf или -1e9
        # Вам может понадобиться torch.where
        # Подсчет вероятностей аттеншена (softmax)
        # Подсчет выхода аттеншена, используя enc и probs
        x = self.linear(dec).unsqueeze(1)
        scores = torch.bmm(x, enc.permute(1, 0, 2)).squeeze(1)
        mask = torch.arange(scores.size(1)).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
        scores[mask] = -1e9
        scores = torch.nn.functional.softmax(scores, dim=1)
        
        return scores 

#второй способ считать attention
class Attention(nn.Module):
    def __init__(self, hidden_dim, attention_dim, bidirectional_encoder=False, init_decoder_from_encoder=False):
        super(Attention, self).__init__()
        input_dim = hidden_dim*2
        input_dim += hidden_dim*bidirectional_encoder
        input_dim += hidden_dim*bidirectional_encoder*init_decoder_from_encoder
        self.attn = nn.Linear(input_dim, attention_dim)
        self.v_fc = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs, lengths):
        hidden = hidden[-1,:,:]
        src_len = encoder_outputs.shape[0]
        hidden = hidden.unsqueeze(1).repeat(1, src_len, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2))) 
        attention = self.v_fc(energy).squeeze(2)
        mask = torch.arange(attention.size(1)).expand(lengths.size(0), -1) >= lengths.unsqueeze(1)
        attention[mask] = -1e9
        return nn.functional.softmax(attention, dim=1)
    
#encoder - кодирует последовательность
class Encoder(nn.Module):
    def __init__(self, 
                 pad_id, 
                 hparams,
                 embedding=None,):
        super().__init__()

        self.use_one_hot = hparams['use_one_hot']
        self.num_directions=1
        if hparams['bidirectional']:
            self.num_directions=2

        self.use_one_hot = hparams['use_one_hot']
        if not self.use_one_hot:
            self.embedding = embedding if embedding is not None else nn.Embedding(hparams['src_vocab_size'],
                                                                                   hparams['emb_dim'])
            input_dim = hparams['emb_dim']  # Update input_dim to emb_dim if not using one-hot encoding
        else:
            input_dim = hparams['src_vocab_size']
    
        self.rnn_type = hparams['rnn_type']
        rnn_module = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU if self.rnn_type == 'gru' else nn.RNN
        self.rnn = rnn_module(input_dim, hparams['hidden_dim'], hparams['n_layers'],
                               dropout=hparams['dropout'], bidirectional=hparams['bidirectional'])
        self.dropout = nn.Dropout(hparams['dropout']) if not self.use_one_hot else None
        self.src_pad_token_id = pad_id

    def forward(self, src, src_len):
        if not self.use_one_hot:
            src = self.dropout(self.embedding(src))
        else:
            #src = torch.stack(
            #    [self.one_hot_encode(batch_sample, self.rnn.input_size) for batch_sample in src]
            #).to(self.device)
            src = self.one_hot_encode(src, self.rnn.input_size)

        packed_embedded = nn.utils.rnn.pack_padded_sequence(src, src_len, enforce_sorted=False)
        packed_outputs, hidden = self.rnn(packed_embedded)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs)

        if self.rnn_type == 'lstm':
            hidden, cell = hidden
            if self.num_directions==2:
                hidden = torch.cat((hidden[::2], hidden[1::2]), dim=2)
                cell = torch.cat((cell[::2], cell[1::2]), dim=2)
            return outputs, hidden, cell
        else:
            if self.num_directions==2:
                hidden = torch.cat((hidden[::2], hidden[1::2]), dim=2)
            return outputs, hidden, None
        
    #если не используем слой эмбеддингов, то переводим токенизированные тексты в one-hot вектора
    def one_hot_encode(self, sequence, vocab_size):
        #tensor = torch.zeros(sequence.size(0), int(vocab_size))
        #tensor[torch.arange(sequence.size(0)), sequence] = torch.ones(sequence.size(0))
        one_hot = torch.zeros(sequence.size(0),
                            sequence.size(1), vocab_size,
                            device=self.device)
        one_hot.scatter_(2, sequence.unsqueeze(-1), 1)
        
        return one_hot

#декодер - декодирует последовательность для encoder-decoder варианта и просто работает на подобие gpt для decoder-only варианта
class Decoder(nn.Module):
    def __init__(self, 
                 hparams,
                 embedding=None,
                 attention=None):
        super().__init__()
        
        self.use_one_hot = hparams['use_one_hot']
        if not self.use_one_hot:
            self.embedding = embedding if embedding is not None else nn.Embedding(hparams['trg_vocab_size'],
                                                                                   hparams['emb_dim'])
            input_dim = hparams['emb_dim']
        else:
            input_dim = hparams['trg_vocab_size']  # When using one-hot, input_dim should match the output_dim (vocab size)
            self.one_hot_dim = input_dim

        self.decoder_input = hparams['decoder_input']
        self.decoder_output = hparams['decoder_output']
        if attention is not None and 'attention' in self.decoder_input:
            input_dim += hparams['hidden_dim']  # If attention is used, adjust the input dimension
              # If attention is used, adjust the input dimension
     
        hidden_dim = hparams['hidden_dim']
        if hparams['bidirectional']:
            if 'attention' in hparams['decoder_input']:
                input_dim += hparams['hidden_dim']
            if hparams['init_decoder_from_encoder']:
                hidden_dim += hidden_dim

        self.attention = attention
        self.rnn_type = hparams['rnn_type']
        rnn_module = nn.LSTM if self.rnn_type == 'lstm' else nn.GRU if self.rnn_type == 'gru' else nn.RNN
        
        self.rnn = rnn_module(input_dim, hidden_dim, 
                              hparams['n_layers'], dropout=hparams['dropout'])
        
        decoder_output = hidden_dim
        decoder_output += hparams['hidden_dim']*('attention' in self.decoder_output)*(1+hparams['bidirectional']) #TO-DO
        decoder_output += hparams['emb_dim']*('emb' in self.decoder_output) 
        self.fc_out = nn.Linear(decoder_output, hparams['trg_vocab_size'])
        self.dropout = nn.Dropout(hparams['dropout']) if not self.use_one_hot else None

    def forward(self, input, hidden, cell=None, encoder_outputs=None, lengths=None):
        
        input = input.unsqueeze(0)
        if not self.use_one_hot:
            embedded = self.dropout(self.embedding(input))
        else:
            #src = torch.stack(
            #    [self.one_hot_encode(batch_sample, self.one_hot_dim) for batch_sample in src]
            #).to(self.device)
            embedded = self.one_hot_encode(input, self.one_hot_dim)

        if encoder_outputs is not None:
            a = self.attention(hidden, encoder_outputs, lengths)
            a = a.unsqueeze(1)
            encoder_outputs = encoder_outputs.permute(1, 0, 2)
            weighted = torch.bmm(a, encoder_outputs)
            weighted= weighted.permute(1, 0, 2)
            if 'attention' in self.decoder_input:
                rnn_input = torch.cat((embedded, weighted), dim=2)
            else:
                rnn_input = embedded
        else:
            rnn_input = embedded

        if self.rnn_type == 'lstm':
            output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))
        else:
            output, hidden = self.rnn(rnn_input, hidden)
        
        if 'attention' in self.decoder_output:
            output = torch.cat((output, weighted), dim=2)
        if 'emb' in self.decoder_output:
            output = torch.cat((embedded, output), dim=2)
            
        output = self.fc_out(output.squeeze(0))
        if self.rnn_type == 'lstm':
            return output, hidden, cell
        else:
            return output, hidden, None
        
    #если не используем слой эмбеддингов, то переводим токенизированные тексты в one-hot вектора
    def one_hot_encode(self, sequence, vocab_size):
        #tensor = torch.zeros(sequence.size(0), int(vocab_size))
        #tensor[torch.arange(sequence.size(0)), sequence] = torch.ones(sequence.size(0))
        one_hot = torch.zeros(sequence.size(0),
                            sequence.size(1), vocab_size,
                              device=self.device)
        one_hot.scatter_(2, sequence.unsqueeze(-1), 1)
        
        return one_hot

    #инициализируем первое скрытое состояние нулями в том случае, когда не получаем его из энкодера
    def init_hidden(self, batch_size):
        hidden = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
        if self.rnn_type == 'lstm':
            cell = torch.zeros(self.rnn.num_layers, batch_size, self.rnn.hidden_size).to(self.device)
            return hidden, cell
        return hidden, None
    
#класс, который собирает все вместе
class Seq2Seq(pl.LightningModule):
    def __init__(self, 
                 tokenizer, 
                 hparams,
                 device='cuda'
                ):
        super().__init__()


        self.tokenizer = tokenizer

        self.model_type = hparams['model_type']
        self.scheduler_type = hparams['scheduler_type']
        self.num_warmup_steps = hparams['num_warmup_steps']
        self.lr = hparams['lr']
        self.use_attention = hparams['use_attention']
        self.rnn_type = hparams['rnn_type']
        self.teacher_forcing_ratio = torch.tensor(hparams['teacher_forcing_ratio'])
        self.init_decoder_from_encoder = hparams['init_decoder_from_encoder']
        self.transform_encoder_hidden = hparams['transform_encoder_hidden']
        self.decoder_input = hparams['decoder_input']
        self.decoder_output = hparams['decoder_output']

        self.save_hyperparameters(Namespace(**hparams),logger=False)

        if self.use_attention:
            self.attention = Attention(hparams['hidden_dim'], hparams['attention_dim'],
                                        hparams['bidirectional'],
                                        hparams['init_decoder_from_encoder'])
        else:
            self.attention = None

        if self.model_type == 'encoder-decoder':
            self.embedding = None
            if hparams['shared_embeddings']:
                self.embedding = nn.Embedding(hparams['src_vocab_size'], hparams['emb_dim'])

            self.encoder = Encoder(tokenizer.src_tokenizer.token_to_id('<pad>'), 
                                   hparams, self.embedding).to(device)
            self.encoder.device = device

            self.decoder = Decoder(hparams, self.embedding, self.attention).to(device)
            self.decoder.device = device

            if self.transform_encoder_hidden:
                self.transform = nn.Linear(self.encoder.rnn.hidden_size*(hparams['bidirectional']+1),
                                            self.decoder.rnn.hidden_size)

        elif self.model_type == 'decoder-only':

            self.decoder = Decoder(hparams)
            self.decoder.device = device
        else:
            raise ValueError("Unsupported model type")

        self.src_pad_token_id = tokenizer.src_tokenizer.token_to_id('<pad>')
        self.trg_pad_token_id = tokenizer.trg_tokenizer.token_to_id('<pad>')
        if hparams['model_type']=='encoder-decoder':
            self.src_eos_token_id = tokenizer.src_tokenizer.token_to_id(tokenizer.src_tokenizer.eos_token)
        self.trg_eos_token_id = tokenizer.trg_tokenizer.token_to_id(tokenizer.trg_tokenizer.eos_token)
        self.break_id = tokenizer.trg_tokenizer.token_to_id(tokenizer.trg_tokenizer.sos_token)


        self.criterion = nn.CrossEntropyLoss(ignore_index=self.trg_pad_token_id)
        self.bleu = BLEUScore()
        self.validation_preds = []
        self.validation_refs = []
        self.test_preds = []
        self.test_refs = []
        self.best_val_loss = float('inf')
        if self.model_type == 'decoder-only':
            self.example_input_array = (torch.randint(0, hparams['src_vocab_size'], (100, hparams['batch_size']), 
                                                   device=device), 
                                                   torch.randint(1, hparams['src_vocab_size'], (hparams['batch_size'],)))
        elif self.model_type == 'encoder-decoder':
            self.example_input_array = (torch.randint(0, hparams['src_vocab_size'], (100, hparams['batch_size']), 
                                                   device=device), 
                                                   torch.randint(1, 50, (hparams['batch_size'],)), 
                                                   torch.randint(0, 50, (100, hparams['batch_size']), 
                                                   device=device))
        

    def forward(self, src, src_len=None, trg=None, inference=False):
        
        if src_len.device != 'cpu':
            src_len = src_len.cpu()

        if trg is None:
            length_size = src.size(0)
            batch_size = src.size(1)
        else:
            length_size = trg.size(0)
            batch_size = trg.size(1)

        if trg is not None:
            outputs = torch.zeros(length_size, batch_size, self.decoder.fc_out.out_features, device=self.device)
        else:
            outputs = torch.zeros(length_size, batch_size, self.decoder.fc_out.out_features, device=self.device)

        #Одна логика для encoder-decoder
        if self.model_type == 'encoder-decoder':

            encoder_outputs, hidden, cell = self.encoder(src, src_len)

            input = trg[0, :]
            eos_generated = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)


            if not self.init_decoder_from_encoder:
                hidden, cell = self.decoder.init_hidden(batch_size)
            else:
                if self.transform_encoder_hidden:
                    hidden = self.transform(hidden)

            for t in range(1, trg.size(0)):
                if self.use_attention:
                    output, hidden, cell = self.decoder(input, hidden, cell, encoder_outputs, src_len)
                else:
                    output, hidden, cell = self.decoder(input, hidden, cell)
                outputs[t] = output
                top1 = output.argmax(1)

                if inference:
                    eos_generated |= (top1 == self.trg_eos_token_id)
                    top1[eos_generated] = self.trg_pad_token_id 
                    input = top1
                else:
                    input = torch.where(torch.le(torch.rand(1, device=self.device), self.teacher_forcing_ratio), trg[t],  top1)
        
        #другая логика для decoder-only
        elif self.model_type == 'decoder-only':
            hidden = self.decoder.init_hidden(batch_size)
            cell = None
            if self.rnn_type == 'lstm':
                hidden, cell = hidden
            input = src[0, :]

            teacher_forcing_ratios = torch.full((batch_size,), self.teacher_forcing_ratio, device=self.device)
            break_token_encountered = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)
            eos_generated = torch.zeros((batch_size,), dtype=torch.bool, device=self.device)

            for t in range(1, src.size(0)):
                output, hidden, cell = self.decoder(input, hidden, cell)
                outputs[t] = output
                top1 = output.argmax(1)

                if inference:
                    break_token_encountered |= (input == self.break_id)
                    teacher_forcing_ratios[break_token_encountered] = 0

                    eos_generated |= (top1 == self.trg_eos_token_id)
                    top1[eos_generated] = self.trg_pad_token_id

                input = torch.where(torch.rand(batch_size, device=self.device) <teacher_forcing_ratios,
                                     src[t, :], top1)

        return outputs

    def extract_tokens_after_break(self, tokens):
        try:
            break_index = tokens.index(self.tokenizer.trg_tokenizer.token_to_id(self.tokenizer.trg_tokenizer.sos_token))
            return tokens[break_index + 1:]
        except ValueError:
            return tokens

    def decode_and_remove_special_tokens(self, tokens, is_source=True):
        decoded_string = self.tokenizer.detokenize(tokens, is_source=is_source, skip_special_tokens=True)
        return decoded_string

    #считаем 100% accuracy (ускорение с помощью numba)
    @staticmethod
    @jit(nopython=True)
    def calculate_full_accuracy(preds, refs):
        correct = 0
        for i in range(min(len(preds), len(refs))):
            if preds[i] == refs[i]:
                correct += 1
        return correct / len(refs)
    
    @staticmethod
    @jit(nopython=True)
    def calculate_partial_accuracy(preds, refs):
        total_chars = 0
        correct_chars = 0
        for pred, ref in zip(preds, refs):
            for i, (p, r) in enumerate(zip(pred, ref)):
                if p != r:
                    break
                correct_chars += 1
            total_chars += len(ref)
        return correct_chars / total_chars 

    #определяем метрики, которые будут ключевыми для выбора лучшей модели и сохраняем их в таблицу гиперпараметров 
    def on_train_start(self):
        print(self.hparams)
        self.logger.log_hyperparams(self.hparams, 
                                    metrics = {"hp/val_bleu": 0, 
                                               "hp/val_full_accuracy": 0, 
                                               "hp/val_partial_accuracy": 0,
                                               "hp/val_loss": float('inf')})
        
    #считаем норму градиентов и тоже сохраняем (на всякий случай, чтобы следить за затуханием/взрывом градиентов)
    def on_before_optimizer_step(self, optimizer):
    # Compute the 2-norm for each layer
    # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    #определяем тренировочный шаг (до получения лосса, optimezr.zero_grad и прочее делать не надо - это происходит под капотом)
    def training_step(self, batch, batch_idx):
        if self.model_type == 'decoder-only':
            src, src_len = batch
            src = src.transpose(0, 1)
            trg = None
        else:
            src, src_len, trg, _ = batch
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)
            src_len = src_len.cpu()

        output = self(src, src_len, trg)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        if self.model_type == 'decoder-only':
            trg = src[1:, :].contiguous().view(-1)
        else:
            trg = trg[1:, :].contiguous().view(-1)
        loss = self.criterion(output, trg)
        self.log('loss/train', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) #сохраняем лосс
        return loss
    
    #определяем валидационный шаг (расчет всех метрик)
    def validation_step(self, batch, batch_idx):
        if self.model_type == 'decoder-only':
            src, src_len = batch
            src = src.transpose(0, 1)
            trg = None
        else:
            src, src_len, trg, _ = batch
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)
            src_len = src_len.cpu()

        output = self(src, src_len, trg, inference=False)
        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        if self.model_type == 'decoder-only':
            trg = src
            trg_all = src[1:, :].contiguous().view(-1)
        else:
            trg_all = trg[1:, :].contiguous().view(-1)
        loss = self.criterion(output, trg_all)
        self.log('loss/val', loss, on_step=True, on_epoch=True, prog_bar=True) #сохраняем лосс

        
        output = self(src, src_len, trg, inference=True)
        pred_tokens = output.argmax(2).transpose(0, 1).tolist()
        trg_tokens = trg.transpose(0, 1).tolist()
        
        if self.model_type == 'decoder-only':
            for i in len(pred_tokens):
                pred_tokens[i] = self.extract_tokens_after_break(pred_tokens[i])
                trg_tokens[i] = self.extract_tokens_after_break(trg[i])

        pred_string, trg_string = [], []

        for pred, target in zip(pred_tokens, trg_tokens):
            pred_string.append(self.decode_and_remove_special_tokens(pred, is_source=False))
            trg_string.append(self.decode_and_remove_special_tokens(target, is_source=False))

        self.validation_preds.extend(pred_string)
        self.validation_refs.extend(trg_string)

    #сохраняем норму весов разный слоев just in case, вдруг будет что интересное (также полезно для дебага)
    def on_train_epoch_end(self):

        for name, param in self.named_parameters():
            norm = torch.norm(param).item()
            self.logger.experiment.add_scalar(f'weight_2.0_norm/{name}', norm, self.current_epoch)

    #оперделяем, что делаем в конце каждой валидационной эаохи - какие метрики считаем
    #если метрики улучшаются (в данном случае лосс на валидации, то обнровляем таблицу гиперпараметров и метрик)
    def on_validation_epoch_end(self):
        bleu_score = self.bleu(self.validation_preds, [[ref] for ref in self.validation_refs])
        full_accuracy = self.calculate_full_accuracy(self.validation_preds, self.validation_refs)
        partial_accuracy = self.calculate_partial_accuracy(self.validation_preds, self.validation_refs)
        val_loss = self.trainer.callback_metrics.get('loss/val_epoch')
        
        if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_bleu_loss = self.bleu
                self.best_full_accuracy = full_accuracy
                self.best_partial_accuracy = partial_accuracy

                self.log('hp/best_vall_loss', val_loss)
                self.log('hp/val_bleu', bleu_score)
                self.log('hp/val_full_accuracy', full_accuracy)
                self.log('hp/val_partial_accuracy', partial_accuracy)
        
        self.log('val/vall_loss', val_loss)
        self.log('val/val_bleu', bleu_score)
        self.log('val/val_full_accuracy', full_accuracy)
        self.log('val/val_partial_accuracy', partial_accuracy)

        for i, (gen_text, corr_text) in enumerate(zip(self.validation_preds[:10], self.validation_refs[:10])):
            text_to_log = f"Generated: {gen_text}\nCorrect: {corr_text}"
            self.logger.experiment.add_text(f'Text_Pair_{i}', text_to_log, self.current_epoch)
        
        self.validation_preds = []
        self.validation_refs = []

    def test_step(self, batch, batch_idx):
        if self.model_type == 'decoder-only':
            src, src_len = batch
            src = src.transpose(0, 1)
            trg = None
        else:
            src, src_len, trg, _ = batch
            src = src.transpose(0, 1)
            trg = trg.transpose(0, 1)
            src_len = src_len.cpu()

        if self.model_type == 'decoder-only':
            trg = src
        
        output = self(src, src_len, trg, inference=True)
        pred_tokens = output.argmax(2).transpose(0, 1).tolist()
        trg_tokens = trg.transpose(0, 1).tolist()

        if self.model_type == 'decoder-only':
            for i in len(pred_tokens):
                pred_tokens[i] = self.extract_tokens_after_break(pred_tokens[i])
                trg_tokens[i] = self.extract_tokens_after_break(trg[i])

        pred_string, trg_string = [], []

        for pred, target in zip(pred_tokens, trg_tokens):
            pred_string.append(self.decode_and_remove_special_tokens(pred, is_source=False))
            trg_string.append(self.decode_and_remove_special_tokens(target, is_source=False))

        self.test_preds.extend(pred_string)
        self.test_refs.extend(trg_string)

    #оперделяем, что делаем для тестовой выборки - какие метрики считаем
    def on_test_epoch_end(self):
        bleu_score = self.bleu(self.test_preds, [[ref] for ref in self.test_refs])
        full_accuracy = self.calculate_full_accuracy(self.test_preds, self.test_refs)
        partial_accuracy = self.calculate_partial_accuracy(self.test_preds, self.test_refs)
        
        self.log('test/bleu', bleu_score, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/full_accuracy', full_accuracy, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/partial_accuracy', partial_accuracy, on_epoch=True, prog_bar=True, logger=True)
        
        self.test_preds = []
        self.test_refs = []

    #оперделяем, как задаются оптимайзеры и шедулеры
    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.lr)

        if self.scheduler_type == 'linear':
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                         num_training_steps=self.trainer.estimated_stepping_batches)
        elif self.scheduler_type == 'cosine':
            scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps,
                                                         num_training_steps=self.trainer.estimated_stepping_batches)
        elif self.scheduler_type == 'constant':
            scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=self.num_warmup_steps)
        else:
            return optimizer

        return [optimizer], [scheduler]
