from transformers import pipeline, AutoTokenizer
import torch

model_id = "state-spaces/mamba-130m-hf"
tokenizer = AutoTokenizer.from_pretrained(model_id)
translator = pipeline(
    "text-generation",
    model="./mamba_diffeq",
    tokenizer=tokenizer,
    device=0 if torch.cuda.is_available() else -1
)

input_eq = "Translate polynomial differential equation: dy/dx = x^2. Solution: "
generated = translator(
    input_eq,
    max_length=200,
    num_return_sequences=1,
    temperature=0.7,
    top_p=0.9
)

print(generated[0]['generated_text'])