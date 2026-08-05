from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# ==================================================
# CONFIG — paths resolved relative to this script
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_MODEL  = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LORA_PATH   = SCRIPT_DIR / "lora-adapters"

if not LORA_PATH.exists():
    raise FileNotFoundError(
        f"LoRA adapter directory not found: {LORA_PATH}\n"
        "Run train_lora.py first to generate the adapter."
    )

# ==================================================
# LOAD MODEL
# ==================================================

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,  # Fixed: was `dtype=` (invalid argument)
    device_map="auto"
)

model = PeftModel.from_pretrained(model, str(LORA_PATH))
model.eval()

# ==================================================
# INFERENCE
# ==================================================

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