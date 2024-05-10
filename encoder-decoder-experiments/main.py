import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import pandas as pd
from Dataloaders import get_loaders
from Encoder import Encoder
from Decoder import Decoder
from Seq2Seq import Seq2Seq
from Training import init_weights, count_parameters, train_fn, evaluate_fn, predict
from Metrics import accuracy, bleu_score, predict_valid_eqs



def train(train_dict, device, k=0, n_epochs=50):
    result = {
        "tokenizer_type": [],
        "vocab_size": [],
        "rnn_type": [],
        "optimizer": [],
        "bidirectional": [],
        "hidden_dim": [],
        "n_layers": [],
        "embedding_dim": [],
        "dropout" : [],
        "learning_rate": [],
        "teacher_forcing_ratio": [],
        "epoch": [],
        "val_bleu": [],
        "val_accuracy": [],
        "test_bleu": [],
        "test_accuracy": [],
        "examples" : [],
        "valid_loss" : [],
        "train_loss" : []
    }
    index = []
    for tokenizer_type in train_dict.get('tokenizer_type', []):
        for vocab_size in train_dict.get('vocab_size', []):
            for only_decoder in train_dict.get('only_decoder', []):
                input_dim = int(vocab_size if tokenizer_type=='BPE' else vocab_size+259)
                output_dim = int(vocab_size if tokenizer_type=='BPE' else vocab_size+259)
                train_data_loader, valid_data_loader, test_data_loader, tokenizer, val_references, test_references, pad_idx = get_loaders('pairs_dataset.csv', 
                                                                                                tokenizer_type, int(vocab_size), batch_size=32, only_decoder=only_decoder)
                for rnn_type in train_dict.get('rnn_type_options', []):
                    for teacher_forcing_ratio in train_dict.get('teacher_forcing_ratio_options', []):
                        for optimizer_name in train_dict.get('optimizer_options', []):
                            for bidirectional in [False, True]:
                                for hidden_dim in train_dict.get('hidden_dim_options', []):
                                    for n_layers in train_dict.get('n_layers_options', []):
                                        for lr in train_dict.get('lr_options', []):
                                            for one_hot in [False, True]:
                                                for emb_dim in [0, 128]:
                                                    for dropout in [0, 0.25]:
                                                        encoder_dropout = dropout
                                                        decoder_dropout = dropout
                                                        if (one_hot and emb_dim!=0) or (not one_hot and emb_dim==0):
                                                            continue
                                                        k+=1
                                                        if one_hot:
                                                            encoder_embedding_dim = input_dim
                                                            decoder_embedding_dim = output_dim
                                                        else:
                                                            encoder_embedding_dim = emb_dim
                                                            decoder_embedding_dim = emb_dim
                                                        encoder = Encoder(
                                                            rnn_type,
                                                            input_dim,
                                                            encoder_embedding_dim,
                                                            hidden_dim,
                                                            n_layers,
                                                            encoder_dropout,
                                                            device,
                                                            pad_idx,
                                                            bidirectional,
                                                            one_hot=one_hot,
                                                            dumb=only_decoder
                                                        )
                                                        
                                                        decoder = Decoder(
                                                            rnn_type,
                                                            vocab_size,
                                                            output_dim,
                                                            decoder_embedding_dim,
                                                            hidden_dim,
                                                            n_layers,
                                                            decoder_dropout,
                                                            device,
                                                            pad_idx,
                                                            bidirectional,
                                                            one_hot=one_hot,
                                                        )
                        
                                                        model = Seq2Seq(encoder, decoder, device).to(device)
                                                        model.apply(init_weights)
                                                        print(
                                                            f"The model has {count_parameters(model):,} trainable parameters"
                                                        )
                        
                                                        if optimizer_name == "Adam":
                                                            optimizer = optim.Adam(model.parameters(), lr=lr)
                                                        else:
                                                            optimizer = optim.AdamW(model.parameters(), lr=lr)
                                                        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
                        
                                                        clip = 1.0
                                                        best_valid_loss = float("inf")
                                                        for epoch in range(n_epochs):
                                                            index.append(k)
                                                            print(f"EPOCH {epoch+1}")
                                                            unique_name = f"model_{k}"
                                                            train_loss = train_fn(
                                                                model,
                                                                train_data_loader,
                                                                optimizer,
                                                                criterion,
                                                                clip,
                                                                teacher_forcing_ratio,
                                                                device,
                                                            )
                                                            valid_loss = evaluate_fn(
                                                                model,
                                                                valid_data_loader,
                                                                criterion,
                                                                device,
                                                            )
                                                            if valid_loss < best_valid_loss:
                                                                best_valid_loss = valid_loss
                                                                torch.save(
                                                                    model.state_dict(),
                                                                    f"encoder_decoder_models_bpe/model_{unique_name}.pt",
                                                                )
                                                            print(
                                                                f"\tTrain Loss: {train_loss:7.3f} | Train PPL: {np.exp(train_loss):7.3f}"
                                                            )
                                                            print(
                                                                f"\tValid Loss: {valid_loss:7.3f} | Valid PPL: {np.exp(valid_loss):7.3f}"
                                                            )
                        
                                                            # compute metrics
                                                            val_predictions = [
                                                                predict(
                                                                    valid_data_loader.dataset["eqs"][i],
                                                                    model,
                                                                    tokenizer,
                                                                    tokenizer_type,
                                                                    valid_data_loader.dataset["len_eqs"][i],
                                                                    True, 
                                                                    device=device,
                                                                    only_decoder=only_decoder
                                                                )
                                                                for i in range(len(valid_data_loader.dataset))
                                                            ]
                                                            # print(val_predictions[:10])
                                                            # print(val_references[:10])
                                                            #test_predictions = [
                                                            #    predict(
                                                            #        test_data_loader.dataset["eqs"][i],
                                                            #        model,
                                                            #        test_data_loader.dataset["len_eqs"][i],
                                                            #        True,
                                                            #    )
                                                            #    for i in range(len(test_data_loader.dataset))
                                                            #]
                                                            
                                                            val_bleu = bleu_score(
                                                                preds=val_predictions, refs=val_references
                                                            )
                                                            #test_bleu = bleu_score(
                                                            #    preds=test_predictions, refs=test_references
                                                            #)
                                                            val_accuracy = accuracy(
                                                                preds=val_predictions, refs=val_references
                                                            )
                                                            #test_accuracy = accuracy(
                                                            #    preds=test_predictions, refs=test_references
                                                            #)
                        
                                                            # save preds on valid_eqs
                                                            d = predict_valid_eqs(model, f"pred_{unique_name}", tokenizer, tokenizer_type, device=device, only_decoder=only_decoder)
                        
                                                            print(
                                                                f"\tValid BLEU: {val_bleu:7.3f} | Valid Accuracy: {val_accuracy:7.3f}"
                                                            )
                                                            result["tokenizer_type"].append(tokenizer_type)
                                                            result["vocab_size"].append(vocab_size)
                                                            result["rnn_type"].append(rnn_type)
                                                            result["teacher_forcing_ratio"].append(teacher_forcing_ratio)
                                                            result["optimizer"].append(optimizer_name)
                                                            result["bidirectional"].append(bidirectional)
                                                            result["n_layers"].append(n_layers)
                                                            result["hidden_dim"].append(hidden_dim)
                                                            result["learning_rate"].append(lr)
                                                            result["dropout"].append(dropout)
                                                            result["embedding_dim"].append(emb_dim)
                                                            result["epoch"].append(epoch + 1)
                                                            result["valid_loss"].append(valid_loss)
                                                            result["train_loss"].append(train_loss)
                                                            result["val_bleu"].append(round(val_bleu, 3))
                                                            result["test_bleu"].append(None)
                                                            result["val_accuracy"].append(round(val_accuracy, 3))
                                                            result["test_accuracy"].append(None)
                                                            result["examples"].append(d)

                                                            res_df = pd.DataFrame(result, index=index)
                                                            res_df.to_pickle("results_bpe.pickle")


if __name__ == "__main__":
    d = {
        'only_decoder' : [True, False],
        'rnn_type_options' : ["rnn", "gru", "lstm"],
        'optimizer_options': ["Adam"],
        'teacher_forcing_ratio_options' : [1],
        'lr_options' : [0.001, 0.01],
        'n_layers_options' : [1, 2, 4],
        'hidden_dim_options' : [16, 64, 256],
        'tokenizer_type' : ['BPE', 'byteBPE', 'regexBPE'],
        'vocab_size' : [int(76*i) for i in [1.5, 2, 4]]
    }
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    train(d, device)
   