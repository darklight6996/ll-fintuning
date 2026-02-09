import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

BASE_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH = "D:/AI Dev/ll-fintuning/ll-finetuning/scripts/lora-adapters"

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(model, LORA_PATH)
model.eval()

prompt = "port 443 is open what next"

inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output = model.generate(
        **inputs,
        max_new_tokens=150,
        do_sample=True,
        temperature=0.7
    )

print(tokenizer.decode(output[0], skip_special_tokens=True))