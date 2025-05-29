import torch
from transformers import AutoTokenizer, MambaForCausalLM
import os
import pandas as pd
from tqdm import tqdm

# Загрузка обученной модели и токенизатора
model_path = "/Users/vovazakharov/equations/mamba/checkpoint-61500"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = MambaForCausalLM.from_pretrained(model_path).to("cuda" if torch.cuda.is_available() else "cpu")

data_dir = os.path.join(os.pardir, 'data')
test_dataset = pd.read_csv(os.path.join(data_dir, 'test.csv'))[:2]

def batch_generate_predictions(dataset, batch_size=64):
    all_predictions = []
    for i in tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[i:i+batch_size]
        texts = [f"Translate {type} differential equation: {equation}. Solution: {answer}" 
             for equation, answer, type in zip(batch['equation'], batch['answer'], batch['type'])]
        tokenized = tokenizer(texts, truncation=True, max_length=512, padding="max_length", return_tensors="pt", padding_side="left")
        input_ids = tokenized.input_ids.to(model.device)
        attention_mask = tokenized.attention_mask.to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **{'input_ids': input_ids, 'attention_mask': attention_mask},
                max_new_tokens=100,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_texts = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        
        for text in generated_texts:
            generated_solution = text.split("Solution:")[1].strip()
            all_predictions.append(generated_solution + "....^^^^^^,,,,,")
    
    return all_predictions

predictions = batch_generate_predictions(test_dataset)
test_dataset['predictions'] = predictions
pd.DataFrame(test_dataset).to_csv("results.csv", index=False, encoding='utf-8')