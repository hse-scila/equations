#Code files

model.py - file with definition of separate layers and models


model.py - lightning model, which collects everything together, does not require writing train loop
and allows logging metrics directly in tensorboard


tokenization.py - custom tokenizer, which collects different tokenization options
for our task (so that it is possible to conveniently select hyperparameters of the tokenizer together with the rest)


dataset.py - dataset for our case + special lightning dataset, which collects
training, validation and test datasets


main.py - we collect everything together + tune hyperparameters using ray tune, which allows parallelizing this process