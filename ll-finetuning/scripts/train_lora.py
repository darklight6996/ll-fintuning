from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch
import os

# ==================================================
# CONFIG — Paths resolved relative to this script
# ==================================================

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR   = SCRIPT_DIR.parent  # ll-finetuning/

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
DATA_PATH  = ROOT_DIR / "data" / "portswigger_alpaca.jsonl"
OUTPUT_DIR = str(SCRIPT_DIR / "lora-output")
ADAPTER_DIR = str(SCRIPT_DIR / "lora-adapters")

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

# ==================================================
# LOAD DATASET
# ==================================================
print(f"Loading dataset from: {DATA_PATH}")
dataset = load_dataset("json", data_files=str(DATA_PATH))

# ==================================================
# LOAD TOKENIZER
# ==================================================
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ==================================================
# LOAD MODEL
# ==================================================
print("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# ==================================================
# FORMAT DATASET
# ==================================================
def merge_fields(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
    return {"text": text}

dataset = dataset.map(merge_fields)

# ==================================================
# TOKENIZE
# ==================================================
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=256
    )

tokenized = dataset.map(tokenize, remove_columns=["text"])

# ==================================================
# LORA CONFIG
# ==================================================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ==================================================
# TRAINING SETTINGS
# ==================================================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    num_train_epochs=2,
    logging_steps=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=torch.cuda.is_available(),  # Only enable fp16 if CUDA available
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator
)

# ==================================================
# TRAIN — safe resume_from_checkpoint guard
# ==================================================
# Only resume if a checkpoint already exists; skip on fresh clone
last_checkpoint = None
if os.path.isdir(OUTPUT_DIR):
    checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = True

trainer.train(resume_from_checkpoint=last_checkpoint)

# ==================================================
# SAVE ADAPTER
# ==================================================
model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"\nAdapters saved to: {ADAPTER_DIR}")