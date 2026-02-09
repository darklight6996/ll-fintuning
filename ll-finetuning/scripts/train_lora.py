from datasets import load_dataset
from transformers import ( AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model
import torch

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#Load dataset

dataset = load_dataset(
    "json", 
    data_files="D:/AI Dev/ll-fintuning/ll-finetuning/data/data.jsonl"
    )

#Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token
# model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

def merge_fields(example):
    text = f"""### Instruction:
{example['instruction']}

### Input:
{example.get('input', '')}

### Response:
{example['output']}"""
    return {"text": text}

dataset = dataset.map(merge_fields)

#tokenization

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
       # padding="max_length",
        max_length=256
    )

tokenized = dataset.map(tokenize, remove_columns=["text"])

#LoRA config

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"

)

#attach LoRA adapter

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

#Training setup

training_args = TrainingArguments(
    output_dir="lora-output",
    per_device_train_batch_size=1,
    num_train_epochs=2,
    logging_steps=1,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
    report_to="none",
    dataloader_pin_memory=False

)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator

)

#train

trainer.train()

#save adapter
model.save_pretrained("lora-adapters")
