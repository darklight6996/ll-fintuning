# ==========================================================
# core/config.py
# Central configuration for CPU + GPU RAG pipelines & Assessment Engine
# ==========================================================

import os

# Resolve paths relative to this config file's directory (core/)
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)

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

CPU_MODEL_NAME = "google/flan-t5-large"
CPU_MAX_INPUT_TOKENS = 2048
CPU_MAX_NEW_TOKENS = 200

# ==============================
# GPU MODEL SETTINGS
# ==============================

GPU_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
GPU_MAX_INPUT_TOKENS = 2048
GPU_MAX_NEW_TOKENS = 512

# ==============================
# EMBEDDING + RERANKING
# ==============================

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"