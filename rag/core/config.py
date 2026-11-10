# ==========================================================
# core/config.py
# Central configuration for CPU + GPU RAG pipelines & Assessment Engine
# ==========================================================

import os

# Resolve paths relative to this config file's directory (core/)
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
ROOT_DIR = os.path.dirname(RAG_DIR)  # project root (ll-finetuning/)

# ==============================
# PATHS
# ==============================

CORPUS_DIR = os.path.join(RAG_DIR, "corpus")

FAISS_INDEX_DIR = os.path.join(RAG_DIR, "faiss_index")
INDEX_PATH = os.path.join(FAISS_INDEX_DIR, "faiss.index")
META_PATH = os.path.join(FAISS_INDEX_DIR, "meta.pkl")

SCENARIOS_DIR = os.path.join(RAG_DIR, "scenarios")
YAML_INDEX_DIR = os.path.join(RAG_DIR, "yaml_index")
YAML_INDEX_PATH = os.path.join(YAML_INDEX_DIR, "yaml.index")
YAML_META_PATH = os.path.join(YAML_INDEX_DIR, "yaml_meta.pkl")

DATABASE_PATH = os.path.join(RAG_DIR, "sessions.db")

# ==============================
# RETRIEVAL SETTINGS
# ==============================

TOP_K = 10
FINAL_K = 3
MAX_CONTENT_CHARS = 3000
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss")  # "faiss", "hnsw", "qdrant", "chroma"

# ==============================
# MEMORY SETTINGS
# ==============================

MEMORY_WINDOW = 6
DEFAULT_SESSION = "default_session"

CPU_SESSION_ID = "cpu_session"
GPU_SESSION_ID = "gpu_session"

# ==============================
# CPU MODEL SETTINGS
# ==============================

# google/flan-t5-large is the base model that was fine-tuned for CPU.
# The PEFT/LoRA adapters can be applied to this base model.
CPU_MODEL_NAME = "google/flan-t5-large"
CPU_MAX_INPUT_TOKENS = 2048
CPU_MAX_NEW_TOKENS = 200

# ==============================
# GPU MODEL SETTINGS
# ==============================

GPU_MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
GPU_MAX_INPUT_TOKENS = 2048
GPU_MAX_NEW_TOKENS = 256

# Qwen target (swap in after download finishes):
# GPU_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_MAX_INPUT_TOKENS = 4096
GPU_MAX_NEW_TOKENS = 1024

# ==============================
# PLANNER MODEL OVERRIDE (M2)
# ==============================
# Optional: use a DIFFERENT model specifically for the autonomous AIPlanner's
# decide_next() calls, while chat, scope-compliance checks, and report synthesis
# keep using CPU_MODEL_NAME / GPU_MODEL_NAME above unchanged.
#
# Left unset (the default), the planner keeps using the exact same shared model --
# this file does NOT force a model change on anyone. Set the environment variable
# below to opt in:
#
#     PLANNER_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct python main.py --autohunt ...
#
# Trade-offs to weigh before setting this (documented here rather than defaulted,
# per AUDIT.md "M2"):
#   - A larger instruction-tuned model generally follows the strict JSON decision
#     contract in ai_planner.py's system prompt far more reliably than a small base
#     model -- directly reducing MAX_JSON_RETRIES exhaustion and the consecutive
#     planner-failure aborts added for MED-1.
#   - That reliability is traded against resident memory: if PLANNER_MODEL_NAME
#     differs from the default model, BOTH models stay loaded in the same process
#     for the lifetime of an autonomous run (chat/report generation still uses the
#     original model), which may not fit in limited VRAM/RAM. If it's the same
#     model, or unset, there is no extra memory cost at all.
PLANNER_MODEL_NAME = os.getenv("PLANNER_MODEL_NAME") or None
PLANNER_MAX_INPUT_TOKENS = int(os.getenv("PLANNER_MAX_INPUT_TOKENS", "0") or 0) or None
PLANNER_MAX_NEW_TOKENS = int(os.getenv("PLANNER_MAX_NEW_TOKENS", "0") or 0) or None

# ==============================
# LORA ADAPTER
# ==============================

# Path to local LoRA adapter directory (used by llm_service when available)
LORA_ADAPTER_PATH = os.path.join(ROOT_DIR, "..", "lora-adapters")

# ==============================
# EMBEDDING + RERANKING
# ==============================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"