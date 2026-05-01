# train_lora_cpu.py

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

# ==================================================
# CONFIG
# ==================================================

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

DATA_PATH = r"C:\Users\UsamaMaqbool\OneDrive - Agency VA\Documents\Documents\ll-fintuning\ll-finetuning\data\htb.jsonl"

OUTPUT_DIR = "lora-output-cpu"
ADAPTER_DIR = "lora-adapters-cpu"

# ==================================================
# LOAD DATASET
# ==================================================

print("Loading dataset...")

dataset = load_dataset(
    "json",
    data_files=DATA_PATH
)

# ==================================================
# LOAD TOKENIZER
# ==================================================

print("Loading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# ==================================================
# LOAD MODEL (CPU SAFE)
# ==================================================

print("Loading model on CPU...")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    dtype=torch.float32
)

model.to("cpu")

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
        padding="max_length",
        max_length=256
    )

tokenized = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

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

# Attach LoRA adapters
model = get_peft_model(model, lora_config)

print("\nTrainable parameters:")
model.print_trainable_parameters()

# ==================================================
# TRAINING SETTINGS (CPU)
# ==================================================

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    logging_steps=1,
    save_strategy="epoch",
    report_to="none",
    use_cpu=True,
    dataloader_pin_memory=False
)

# ==================================================
# DATA COLLATOR
# ==================================================

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ==================================================
# TRAINER
# ==================================================

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator
)

# ==================================================
# TRAIN
# ==================================================

print("\nStarting CPU LoRA training...\n")

trainer.train()

# ==================================================
# SAVE ADAPTER
# ==================================================

print("\nSaving adapters...")

model.save_pretrained(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)

print("\nTraining complete.")
print(f"Adapters saved to: {ADAPTER_DIR}")