from datasets import load_dataset
from transformers import ( AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig, get_peft_model

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

#Load dataset

dataset = load_dataset(
    "json", 
    data_files="../data/train.jsonl"
    )

#Load tokenizer and model

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

#tokenization

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
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
    num_train_epochs=3,
    logging_steps=1,
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
