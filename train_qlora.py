"""
QLoRA Fine-tuning — Qwen2.5-7B-Instruct → Cybersecurity/CTF Expert
====================================================================
Hardware target : RTX 3060 Ti  (8.6 GB VRAM)
Quantization    : 4-bit NF4  (BitsAndBytes)
Adapter         : LoRA  r=16, alpha=32
Stack versions  : transformers==5.17, trl==1.13, peft==0.21

Run:
    python train_qlora.py
    python train_qlora.py --resume

Estimated VRAM  : ~7.5 GB
Estimated time  : ~2-4 h  (3 epochs, ~817 examples, seq_len 1024)
"""

import os
import sys
import math
import logging
from pathlib import Path
from datetime import datetime

# Configure PyTorch memory allocator for Windows (max_split_size_mb avoids fragmentation)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, TaskType, get_peft_model
from trl import SFTTrainer

# ──────────────────────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────────────────────
ROOT        = Path(__file__).resolve().parent
DATA_PATH   = ROOT / "data" / "qwen_chatml.jsonl"
OUTPUT_DIR  = ROOT / "lora-output"  / "qwen-qlora"
ADAPTER_DIR = ROOT / "lora-adapters" / "qwen-qlora"
LOG_DIR     = ROOT / "ll-finetuning" / "logs"
LOG_FILE    = LOG_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

for d in [OUTPUT_DIR, ADAPTER_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ──────────────────────────────────────────────────────────────
# LOGGING
# ──────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
    ],
)
log = logging.getLogger(__name__)
log.info(f"Training log: {LOG_FILE}")

# ──────────────────────────────────────────────────────────────
# HYPERPARAMETERS (Hardened for 8GB VRAM & Thermal/Power Stability)
# ──────────────────────────────────────────────────────────────
MODEL_NAME   = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LEN  = 512         # Covers ~87% of data without quadratic attention VRAM explosion
BATCH_SIZE   = 1
GRAD_ACCUM   = 8           # effective batch = 8
NUM_EPOCHS   = 3
LR           = 2e-4
WARMUP_STEPS = 20          # transformers 5.17 uses warmup_steps, not warmup_ratio
LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
# Safe target modules: focus on attention heads to keep activation tensors within 8GB
LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
]

# ──────────────────────────────────────────────────────────────
# SANITY CHECKS
# ──────────────────────────────────────────────────────────────
if not DATA_PATH.exists():
    raise FileNotFoundError(f"Training data not found: {DATA_PATH}")

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    log.info(f"GPU: {torch.cuda.get_device_name(0)}  ({vram_gb:.1f} GB VRAM)")
    # Enforce strict VRAM memory cap: Leaves ~1 GB buffer for Windows display/desktop apps
    torch.cuda.set_per_process_memory_fraction(0.88, device=0)
    log.info("CUDA fail-safe active: Per-process VRAM capped at 88% (OS/WDDM buffer protected)")
else:
    log.warning("CUDA not available — training on CPU will be extremely slow!")

# ──────────────────────────────────────────────────────────────
# 4-BIT QUANTIZATION CONFIG
# ──────────────────────────────────────────────────────────────
log.info("Configuring 4-bit NF4 quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,  # bf16 matches training precision
    bnb_4bit_use_double_quant=True,
)

# ──────────────────────────────────────────────────────────────
# TOKENIZER
# ──────────────────────────────────────────────────────────────
log.info(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ──────────────────────────────────────────────────────────────
# BASE MODEL (4-bit)
# ──────────────────────────────────────────────────────────────
log.info("Loading base model with 4-bit quantization...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager",
)

# Freeze base model parameters
for param in model.parameters():
    param.requires_grad = False

# Enable input gradients for LoRA with gradient checkpointing
if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()
else:
    def make_inputs_require_grad(module, input, output):
        output.requires_grad_(True)
    model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

# ──────────────────────────────────────────────────────────────
# LORA
# ──────────────────────────────────────────────────────────────
log.info(f"Applying LoRA: r={LORA_R}, alpha={LORA_ALPHA}")
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_R,
    lora_alpha=LORA_ALPHA,
    target_modules=LORA_TARGETS,
    lora_dropout=LORA_DROPOUT,
    bias="none",
    inference_mode=False,
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ──────────────────────────────────────────────────────────────
# DATASET — tokenize manually (SFTTrainer 1.13 has no text_field arg)
# ──────────────────────────────────────────────────────────────
log.info(f"Loading dataset: {DATA_PATH}")
raw = load_dataset("json", data_files=str(DATA_PATH), split="train")
log.info(f"Total examples: {len(raw)}")

raw = raw.shuffle(seed=42)
split    = raw.train_test_split(test_size=0.05, seed=42)
train_ds = split["train"]
eval_ds  = split["test"]
log.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")

def tokenize_fn(batch):
    result = tokenizer(
        batch["text"],
        truncation=True,
        max_length=MAX_SEQ_LEN,
        padding=False,
    )
    result["labels"] = result["input_ids"].copy()
    return result

log.info("Tokenizing datasets...")
train_tok = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
eval_tok  = eval_ds.map(tokenize_fn,  batched=True, remove_columns=["text"])

# ──────────────────────────────────────────────────────────────
# CHECKPOINT RESUME
# ──────────────────────────────────────────────────────────────
resume_from = None
if "--resume" in sys.argv and OUTPUT_DIR.exists():
    checkpoints = sorted(
        [d for d in OUTPUT_DIR.iterdir() if d.name.startswith("checkpoint-")],
        key=lambda p: int(p.name.split("-")[-1]),
    )
    if checkpoints:
        resume_from = str(checkpoints[-1])
        log.info(f"Resuming from: {resume_from}")

# ──────────────────────────────────────────────────────────────
# TRAINING ARGUMENTS — Hardened for hardware stability
# ──────────────────────────────────────────────────────────────
steps_per_epoch = math.ceil(len(train_tok) / (BATCH_SIZE * GRAD_ACCUM))
total_steps     = steps_per_epoch * NUM_EPOCHS
log.info(f"Steps/epoch: {steps_per_epoch} | Total steps: {total_steps}")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),

    # Schedule
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},

    # Optimiser
    optim="paged_adamw_8bit",
    learning_rate=LR,
    lr_scheduler_type="cosine",
    warmup_steps=WARMUP_STEPS,
    weight_decay=0.01,
    max_grad_norm=1.0,

    # Precision
    fp16=False,
    bf16=True,

    # Checkpointing (Safe intervals)
    save_strategy="steps",
    save_steps=100,
    save_total_limit=2,

    # Evaluation disabled during training loop to prevent activation spike crashes
    eval_strategy="no",

    # Logging
    logging_steps=5,
    report_to="none",

    # Misc
    dataloader_num_workers=0,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    seed=42,
)

# ──────────────────────────────────────────────────────────────
# DATA COLLATOR
# ──────────────────────────────────────────────────────────────
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
)

# ──────────────────────────────────────────────────────────────
# TRAINER — SFTTrainer 1.13 signature (no text_field/max_seq_len args)
# ──────────────────────────────────────────────────────────────
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_tok,
    eval_dataset=eval_tok,
    processing_class=tokenizer,
    data_collator=data_collator,
)

# ──────────────────────────────────────────────────────────────
# TRAIN
# ──────────────────────────────────────────────────────────────
log.info("=" * 60)
log.info("Starting QLoRA training...")
log.info("=" * 60)

try:
    trainer.train(resume_from_checkpoint=resume_from)
except KeyboardInterrupt:
    log.warning("Training interrupted — saving current adapter...")

# ──────────────────────────────────────────────────────────────
# SAVE ADAPTER
# ──────────────────────────────────────────────────────────────
log.info(f"Saving LoRA adapter to: {ADAPTER_DIR}")
trainer.model.save_pretrained(str(ADAPTER_DIR))
tokenizer.save_pretrained(str(ADAPTER_DIR))

log.info("=" * 60)
log.info("Training complete!")
log.info(f"Adapter: {ADAPTER_DIR}")
log.info("=" * 60)
