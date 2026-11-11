# ==========================================================
# core/config.py
# Central configuration for CPU + GPU RAG pipelines & Assessment Engine
# ==========================================================

import os

# Resolve paths relative to this config file's directory (core/).
#
# BUG FIX (2026-11-11): core/__init__.py (repo root) extends its __path__ to
# include a top-level `rag` entry that is actually a symlink/junction pointing
# at ll-finetuning/rag -- NOT a real directory one level under the true repo
# root. os.path.abspath(__file__) does NOT dereference that symlink, so the old
# code here saw __file__ as "<repo_root>/rag/core/config.py" (via the symlink's
# *logical* location) instead of its real physical path
# "<repo_root>/ll-finetuning/rag/core/config.py" -- undercounting the real
# nesting depth by one directory. That made ROOT_DIR silently equal the TRUE
# repo root instead of the ll-finetuning/ folder the comment below (and
# LORA_ADAPTER_PATH's "..") assume, so LORA_ADAPTER_PATH overshot one level
# above the repo root into a directory that doesn't exist. The practical
# effect: os.path.exists(LORA_ADAPTER_PATH) was always False, so
# ModelRouter's LoRA auto-discovery (core/model_router.py _load_model() step 3)
# silently never fired on ANY past run -- every assessment to date ran the
# bare base model, never the fine-tuned adapter. os.path.realpath() resolves
# the symlink first, fixing the depth count regardless of which path (direct
# or via the `rag` symlink) was used to import this module.
CORE_DIR = os.path.dirname(os.path.realpath(__file__))
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

# Qwen2.5-7B-Instruct + the user's own qwen-qlora fine-tune (lora-adapters/qwen-qlora/,
# trained via train_qlora.py: r=16, alpha=32, dropout=0.05, target_modules
# q/k/v/o_proj, ChatML-formatted data/qwen_chatml.jsonl) is now the single model used
# EVERYWHERE -- chat, report synthesis, scope-compliance checks, and the planner --
# since PLANNER_MODEL_NAME below is left unset and therefore resolves to this same
# GPU_MODEL_NAME (see llm_service.get_llm_service()). ModelRouter auto-discovers and
# attaches the local lora-adapters/qwen-qlora adapter for any "qwen" model_name
# (core/model_router.py _load_model() step 3), so no adapter_path needs passing here.
GPU_MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
GPU_MAX_INPUT_TOKENS = 4096
GPU_MAX_NEW_TOKENS = 1024

# The qwen-qlora adapter was TRAINED in 4-bit NF4 (bitsandbytes QLoRA) specifically
# for an 8GB RTX 3060 Ti. fp16 inference of a bare 7B model needs ~14GB VRAM, which
# will not fit that card -- ModelRouter's GPU loader normally tries fp16 first and
# only falls back to 4-bit NF4 on an OOM exception, which both wastes a doomed load
# attempt and can leave CUDA in a dirty state after a partial OOM. Setting this to
# True skips straight to the 4-bit NF4 path (matching training exactly) instead of
# attempting the fp16 fast path first. Only affects models matched by
# model_router.py's `use_large_model` check (7b/qwen/mistral/PEFT); set to False to
# restore the old try-fp16-then-fallback behavior.
GPU_FORCE_4BIT = True

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