import torch
import transformers
from transformers import AutoTokenizer, GPTNeoXForCausalLM

transformers.logging.set_verbosity_info()

model_path = 'EleutherAI/pythia-70m'

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = GPTNeoXForCausalLM.from_pretrained(
    model_path, torch_dtype=torch.float32, device_map='auto'
)

inputs = tokenizer("What is eva?", return_tensors="pt")
tokens = model.generate(**inputs)
print(tokenizer.decode(tokens[0]))
