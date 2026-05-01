# Cybersecurity Instruction Fine-Tuning & RAG Pipeline

A comprehensive pipeline for building and fine-tuning Large Language Models (LLMs) on cybersecurity and penetration testing datasets. This project includes tools for web scraping, PDF document processing, Retrieval-Augmented Generation (RAG) corpus building, and Parameter-Efficient Fine-Tuning (PEFT) using LoRA.

## 🚀 Overview

This repository provides an end-to-end workflow to:
1. **Scrape & Extract**: Gather cybersecurity knowledge from web writeups and PDF documents.
2. **Process**: Chunk and structure data into RAG-ready formats and Alpaca-style instruction datasets.
3. **Fine-Tune**: Train models (like TinyLlama) using LoRA (Low-Rank Adaptation) on both CPU and GPU.
4. **Augment**: Use RAG to provide contextual information during training or inference.

## 📂 Project Structure

```text
.
├── ll-finetuning
│   ├── data              # Raw datasets (JSONL), Alpaca formatted data
│   ├── Generic Scripts   # Scrapers, PDF parsers, and RAG-to-Alpaca converters
│   ├── models            # Local model storage
│   ├── scripts           # Training scripts (CPU/GPU LoRA)
│   ├── setup             # OS-specific requirements (Windows 11, Ubuntu)
│   └── DATA_DOCUMENTATION.md # Detailed breakdown of the training dataset
├── lora-adapters         # Saved LoRA adapter weights
└── lora-output           # Training checkpoints and logs
```

## 🛠️ Setup & Installation

Detailed requirement files for different operating systems are located in the `ll-finetuning/setup` directory.

### Windows 11
1. Install [NVIDIA CUDA Toolkit 12.1+](https://developer.nvidia.com/cuda-downloads).
2. Install [Visual Studio C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
3. Install dependencies:
   ```powershell
   pip install -r ll-finetuning/setup/windows11/requirements.txt
   ```

### Ubuntu
1. Install NVIDIA CUDA drivers and toolkit.
2. Install build essentials: `sudo apt install build-essential python3-dev`.
3. Install dependencies:
   ```bash
   pip install -r ll-finetuning/setup/ubuntu/requirements.txt
   ```

## 📖 Key Workflows

### 1. Data Collection & Processing
*   **Web Scraping**: Use `scrape_writeups.py` to gather HTB writeups.
*   **PDF Processing**: Use `pdf_to_fintune.py` to convert security manuals/PDFs into instruction pairs.
*   **RAG to Alpaca**: Use `rag_to_alpaca_multiple.py` to convert structured RAG chunks into JSONL format for fine-tuning.

### 2. Fine-Tuning
Training scripts are located in `ll-finetuning/scripts/`:
*   **GPU Training**: `train_lora.py` (Optimized for CUDA).
*   **CPU Training**: `train_lora_cpu.py` (For testing or low-resource environments).

### 3. Dataset Documentation
For a detailed explanation of the categories and tools covered in the training data, refer to [DATA_DOCUMENTATION.md](ll-finetuning/data/DATA_DOCUMENTATION.md).

## 🛡️ Disclaimer
This project is for educational and authorized security research purposes only. Always ensure you have explicit permission before scraping websites or testing systems.
