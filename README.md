# 🛡️ Cybersecurity Instruction Fine-Tuning & RAG Pipeline (v3.0)

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CUDA%2012.1-red.svg)](https://pytorch.org/)
[![PEFT LoRA](https://img.shields.io/badge/PEFT-LoRA-green.svg)](https://github.com/huggingface/peft)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-DarkLight6996%2FCyberSec--Model--V2-yellow.svg)](https://huggingface.co/DarkLight6996/CyberSec-Model-V2)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

An end-to-end production pipeline for training, evaluating, and deploying specialized Large Language Models (LLMs) tailored for cybersecurity, penetration testing, ethical hacking, and threat intelligence. 

Includes scraping utilities, corpus generation, multi-scenario Alpaca data synthesis, GPU/CPU Parameter-Efficient Fine-Tuning (LoRA), SHA-256 model fingerprint verification, and automated Hugging Face deployment.

---

## 🚀 Hugging Face Live Model

The latest GPU fine-tuned model adapter is hosted publicly on Hugging Face Hub:
* **Model Repo:** [DarkLight6996/CyberSec-Model-V2](https://huggingface.co/DarkLight6996/CyberSec-Model-V2)
* **Base Architecture:** `TinyLlama/TinyLlama-1.1B-Chat-v1.0`
* **Verified SHA-256 Fingerprint:** `d01702e8cc57b7ad690859c3007e9e6949da17dd37ce9e64f3af93f8c2541c87`

---

## 📂 Project Structure

```text
ll-fintuning/
├── data/                                 # Training datasets and corpus
│   ├── comprehensive_cybersec_finetune.jsonl # Complete multi-scenario training dataset
│   ├── redefining_hacking.jsonl          # Core exploitation and methodology dataset
│   ├── structured_data.jsonl             # Parsed instruction pairs
│   └── DATA_DOCUMENTATION.md             # Dataset schema & category documentation
├── Generic Scripts/                      # Utility, parser & audit scripts
│   ├── check_hashes.py                   # SHA-256 model verification tool vs Hugging Face
│   ├── pdf_to_fintune.py                 # Security manual to instruction converter
│   ├── pdf_to_rag_corpus.py              # PDF chunking and corpus generation
│   ├── rag_to_alpaca_multiple.py         # Multi-source RAG to Alpaca converter
│   └── WebPage_chunker.py                # HTB / Web writeup chunking pipeline
├── rag/                                  # RAG scenarios & threat intelligence corpus
│   └── scenarios/                        # 30+ granular attack/defense YAML scenarios
├── scripts/                              # Core Training, Inference & Hub Upload
│   ├── train_lora.py                     # GPU-optimized LoRA training script (CUDA)
│   ├── train_lora_cpu.py                 # Fallback CPU training script
│   ├── infer_lora.py                     # Interactive inference tester
│   └── upload_to_hf.py                   # Automated Hugging Face Hub deployment
├── setup/                                # Environment dependencies
│   ├── windows11/requirements.txt        # Windows 11 CUDA requirements
│   └── ubuntu/requirements.txt           # Linux / Ubuntu requirements
└── README.md                             # Project overview & usage guide
```

---

## ⚡ Quickstart Guide

### 1. Environment Setup

#### Windows 11 (CUDA Enabled)
1. Install [NVIDIA CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads).
2. Install dependencies:
```powershell
pip install -r ll-finetuning/setup/windows11/requirements.txt
```

#### Ubuntu / Linux
```bash
sudo apt update && sudo apt install build-essential python3-dev -y
pip install -r ll-finetuning/setup/ubuntu/requirements.txt
```

---

### 2. Fine-Tuning the Model (GPU)

Run the GPU-optimized training script with automated gradient accumulation and FP16 mixed precision:

```bash
python "ll-finetuning/scripts/train_lora.py"
```

To resume from an existing checkpoint:
```bash
python "ll-finetuning/scripts/train_lora.py" --resume
```

Trained LoRA adapters are automatically saved to `ll-finetuning/scripts/lora-adapters/`.

---

### 3. Verify Model Integrity (SHA-256 Fingerprint)

Compare your local adapter weights against the live Hugging Face repository version:

```bash
python "ll-finetuning/Generic Scripts/check_hashes.py" --repo-id DarkLight6996/CyberSec-Model-V2
```

---

### 4. Deploy to Hugging Face Hub

Upload the trained adapter weights with automatic repository creation and error handling:

```bash
python "ll-finetuning/scripts/upload_to_hf.py" --repo-id DarkLight6996/CyberSec-Model-V2
```

*For a private repository:*
```bash
python "ll-finetuning/scripts/upload_to_hf.py" --repo-id DarkLight6996/CyberSec-Model-V2 --private
```

---

### 5. Run Local Inference

Test the fine-tuned cybersecurity model locally:

```bash
python "ll-finetuning/scripts/infer_lora.py"
```

---

## 🏷️ Version History & Releases

* **`v3.0` (Current - Main)**:
  * Upgraded dataset with comprehensive multi-scenario security vectors (`comprehensive_cybersec_finetune.jsonl`).
  * Automated Hugging Face Hub upload tooling (`upload_to_hf.py`).
  * SHA-256 cryptographic weight verification against Hugging Face (`check_hashes.py`).
  * Optimized CUDA training pipeline and updated documentation.
* **`v2.1` (Legacy Tag)**:
  * Initial GPU compatibility audit, portable path resolution, and assessment modules.
* **`v2.0` (Legacy Release)**:
  * RAG integration, initial benchmarking, and dual CPU/GPU training pipelines.
* **`v1.0` (Legacy Release)**:
  * Initial proof-of-concept scraper and basic Alpaca dataset generation.

---

## ⚖️ Disclaimer

This project and trained models are strictly intended for **educational, authorized security auditing, and defensive research purposes only**. Never execute vulnerability scanning, penetration testing, or exploit payloads without explicit written authorization from system owners.
