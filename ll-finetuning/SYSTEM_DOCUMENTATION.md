# Cybersecurity Instruction Fine-Tuning & RAG Pipeline
## Master System Documentation & Architectural Specification


## Architectural Updates (September 2026)

### 1. Next-Generation Qwen2.5-7B-Instruct QLoRA & ChatML Pipeline
- **Primary Model Evolution**: Transitioning primary offensive/defensive fine-tuning from `TinyLlama-1.1B` to `Qwen/Qwen2.5-7B-Instruct`, targeting the adapter repository `DarkLight6996/CyberSec-Qwen-V1`.
- **ChatML Token Framing**: Migrated from legacy Alpaca 3-field JSONL to strict Qwen ChatML formatting (`<|im_start|>system...<|im_end|>\n<|im_start|>user...<|im_end|>\n<|im_start|>assistant...<|im_end|>`) via `scripts/alpaca_to_chatml.py`.
- **4-Bit QLoRA Optimization**: Engineered for native Windows execution on 8GB consumer VRAM (RTX 3060 Ti) using `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)`.
- **PyTorch Native SDPA**: Enforced `attn_implementation="sdpa"` to eliminate complex C++ compilation and FlashAttention build failures on Windows.
- **LoRA Hyperparameters**: LoRA rank $r=8$, $\alpha=16$, dropout $0.05$ targeting all 7 linear projection layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). Sequence length constrained to `max_seq_length=512`.

### 2. Engineering Verification Gates & Demo Readiness
- **Gate 1 (Go/No-Go)**: Day 1 `bitsandbytes` 4-bit CUDA verification gate (`scripts/verify_bnb.py`) to validate 4-bit quantization kernel execution on Windows.
- **Gate 2 (VRAM Decision Tree)**: Dynamic sequence length and rank throttle: if VRAM peak exceeds 7.5GB during 10-sample testing, `max_seq_length` throttles from 512 to 384 and rank from 8 to 4.
- **Gate 3 (Dual-Track Fallback)**: Standby rehearsals for both the verified `DarkLight6996/CyberSec-Model-V2` (TinyLlama 1.1B adapter) and the deterministic 8-branch rule engine to guarantee zero risk during government demonstrations.
- **Measurable Acceptance Criteria**: Strict pass/fail benchmarks enforced across CTF access control, session reset isolation, and Linux reconnaissance playbooks.

### 3. Hybrid LLM Reasoning & Fallback Engine (`chat_assistant.py`)
- **Transition**: Migrated from a static N-branch decision tree to a **Hybrid LLM + Deterministic Fallback** architecture.
- **LLM First**: User queries are passed to an LLM (`ModelRouter`) powered by a highly structured prompt containing live session intelligence, RAG documents, and YAML scenario tactics.
- **Prompt Synchronization**: Inference prompt construction updated to format system persona, live session intel, untrusted evidence, RAG context, and chat history into native ChatML before generation.
- **Rule Fallback**: The deterministic 8-branch rule engine executes if the LLM is unavailable, offline, or returns empty tokens.
- **Safety**: Context resets are confirmation-gated (`reset confirm`) to prevent accidental session wipes.
- **Prompt Injection Defense**: All untrusted terminal output ingested into the LLM prompt is heavily sandboxed within `<<<BEGIN_EVIDENCE>>>` and `<<<END_EVIDENCE>>>` delimiters and accompanied by strict data-only instructions.

### 4. Command Execution Security (`command_executor.py`, `variable_engine.py`)
- **Shell Escaping**: Completely removed `shell=True` from `CommandExecutor`. All commands are executed via list formats (`shell=False`) after passing through `shlex.split()`.
- **Injection Mitigation**: The `VariableEngine` now routes all substituted variables through `shlex.quote()` prior to command generation.
- **Network Validation**: Strict allow-list regex validation (`^[a-zA-Z0-9\.\-\_\:\/]+$`) enforced on all network parameters (IP, PORT, HOST, DOMAIN) to block unexpected shell characters.

### 5. Engine Configuration & Vector Search (`config.py`, `rag_engine.py`)
- **Context Budgets**: Input context budget dynamically allocated across live intel, recent turns, and RAG chunks.
- **Similarity Normalization**: FAISS L2 distances normalized using `1.0 / (1.0 + distance)`.
- **Noise Thresholds**: Hard negative logit cutoff (`min_score = -2.5`) applied during RAG retrieval to prevent irrelevant knowledge from polluting the prompt context.

### 6. Phase 1 — Global Architecture Refactor (September 2026)

A production-quality services layer (`core/services/`) was introduced to enforce SOLID principles, eliminate duplicate model loading, and apply dependency injection across the entire pipeline.

#### 6.1 New Services Layer (`core/services/`)

| Service | File | Responsibility |
|---|---|---|
| `EmbeddingService` | `embedding_service.py` | Owns `SentenceTransformer` — one instance per process |
| `RerankService` | `rerank_service.py` | Owns `CrossEncoder` — one instance per process |
| `LLMService` | `llm_service.py` | Auto-detects GPU/CPU, owns `ModelRouter`, exposes `generate()` |
| `YAMLService` | `yaml_service.py` | Owns `YAMLLoader + YAMLRegistry + YAMLSearch` stack |
| `RAGService` | `rag_service.py` | Wraps `RAGEngine` with `retrieve()` / `build_context()` interface |
| `MemoryService` | `memory_service.py` | Per-session `ChatMemory` registry — prevents duplicate DB connections |

All services expose a `get_<service>()` module-level singleton accessor. Services are loaded lazily on first call and cached for the process lifetime.

#### 6.2 Refactored Core Files

- **`config.py`**: `GPU_MODEL_NAME` updated to `DarkLight6996/CyberSec-Model-V2`. Added `ROOT_DIR` and `LORA_ADAPTER_PATH` constants.
- **`model_router.py`**: Added `get_global_model_router()` singleton getter. Broadened 4-bit quantization detection to cover all PEFT adapters, not just known model families.
- **`rag_engine.py`**: `_load_components()` now delegates embedding and reranking to shared service singletons — eliminates duplicate VRAM allocation if multiple `RAGEngine` instances exist.
- **`chat_assistant.py`**: Removed 3 private 50-line lazy-loaders (`_get_llm_router`, `_get_rag_engine`, `_get_yaml_search`). Constructor is now 5 lines. Service resolution happens through `get_llm_service()`, `get_rag_service()`, `get_yaml_service()`.

#### 6.3 Backward Compatibility

`main.py` and all existing entry points remain **untouched**. All prior imports (`from core.model_router import ModelRouter`, `from core.rag_engine import get_global_rag_engine`, etc.) continue to work unchanged.

#### 6.4 Verification Results

All 6 smoke tests passed:
- Service imports: ✅
- Config values correct (`DarkLight6996/CyberSec-Model-V2`): ✅
- `ChatAssistant.__init__` has zero side-effects: ✅
- `RAGService` retrieves and builds context: ✅
- `YAMLService` loads 214 scenarios, returns search results: ✅
- `MemoryService` session isolation verified: ✅

> **YAML Note**: 19 of 233 scenario files have syntax errors (bad escape sequences in `adcs-*`, `event-log-tampering`, etc.). These are pre-existing issues in the scenario corpus and do not affect system operation — 214 valid scenarios loaded successfully.

---

### 7. Phase 2 — Vector Store Abstraction Layer (`core/retrieval/`)

A pluggable vector database abstraction was introduced under `core/retrieval/` so the system can swap storage backends without touching any retrieval or service code.

#### 7.1 New Module: `core/retrieval/vector_store.py`

| Backend | Class | Notes |
|---|---|---|
| FAISS | `FAISSVectorStore` | Default. In-process; no server required. Production-verified. |
| HNSW | `HNSWVectorStore` | `hnswlib` adapter; higher throughput at scale. |
| Qdrant | `QdrantVectorStore` | `qdrant_client` adapter; supports persistent remote storage. |
| Chroma | `ChromaVectorStore` | `chromadb` adapter; lightweight embedded option. |

All backends implement the `BaseVectorStore` abstract interface: `search()`, `add()`, `persist()`, `load()`, `count()`. The active backend is selected at startup via `VECTOR_STORE_BACKEND` environment variable (default: `"faiss"`). `config.py` exposes this setting.

#### 7.2 New Module: `core/retrieval/retriever.py`

`Retriever` wraps a `BaseVectorStore` instance and exposes a unified `retrieve(query, top_k)` API. It includes a `ChunkFilter` that prunes low-similarity and duplicate chunks before results reach the reranker.

#### 7.3 New Module: `core/retrieval/reranker.py`

`RetrieverReranker` wraps `RerankService` and re-sorts a short candidate list by cross-encoder score, returning only `final_k` highest-confidence chunks.

#### 7.4 New Module: `core/retrieval/context_builder.py`

`RetrievalContextBuilder` merges reranked chunks into a single formatted string and ensures the combined length stays within the configured `MAX_CONTENT_CHARS` budget.

#### 7.5 `config.py` Addition

```python
VECTOR_STORE_BACKEND = os.getenv("VECTOR_STORE_BACKEND", "faiss")
```

#### 7.6 Verification Results

| Test | Status |
|---|---|
| `test_faiss_vector_store_in_memory` | ✅ PASSED |
| `test_vector_store_factory_switching` | ✅ PASSED |
| `test_chunk_filter` | ✅ PASSED |
| `test_retriever_with_mocks` | ✅ PASSED |
| `test_reranker_with_mock_service` | ✅ PASSED |
| `test_rag_context_builder` | ✅ PASSED |
| `test_rag_engine_backward_compatibility` | ✅ PASSED |

---

### 8. Phase 3 — Intent Router Extraction (`core/routers/intent_router.py`)

The 8-branch deterministic fallback engine was fully extracted from `ChatAssistant` into an independent `IntentRouter` class, eliminating the coupling between conversation orchestration and route logic.

#### 8.1 `Intent` Enum (8 values)

```
RESET | RECON | STEERING | EXPLAIN_OUTPUT | CTF_PRIVESC | AD_NETWORK | WEB_API | GENERAL
```

#### 8.2 Key Methods

| Method | Purpose |
|---|---|
| `detect_operator_steering(query)` | Detects hallucination correction / tangent signals |
| `detect_reset_or_different_challenge(query)` | Detects session reset requests |
| `detect_recon_request(query)` | Detects initial recon / where-do-I-start queries |
| `classify(query, state, intel)` | Returns `Intent` enum value |
| `execute_reset(state)` | Wipes session state after `"reset confirm"` gate |
| `dispatch_fallback(intent, ...)` | Routes to the appropriate deterministic reasoning method |

The intent router is stateless; `ChatAssistant` supplies `state` and `intel` dicts per turn. Fallback reasoning methods (`reason_ctf_query`, `reason_linux_recon_methodology`, etc.) remain embedded in `intent_router.py` for locality.

---

### 9. Phase 4 — Intelligence Extraction Subsystem (`core/intelligence/`)

The monolithic intelligence extraction regex block was decomposed into a plugin-based parser registry under `core/intelligence/`.

#### 9.1 Architecture

```
core/intelligence/
  base.py          — BaseIntelParser abstract class
  models.py        — SessionIntelligence dataclass (typed, frozen)
  extractor.py     — IntelligenceExtractor: orchestrates all parsers
  parsers/
    user.py        — UID / GID / username extraction
    kernel.py      — OS kernel version and architecture
    ports.py       — open TCP/UDP port parsing
    banners.py     — service banner / version extraction
    binaries.py    — interesting binary identification
    suid.py        — SUID / SGID binary detection
    strings.py     — strings(1) output extraction
    flags.py       — CTF flag pattern matching
    filesystem.py  — sensitive file / directory detection
```

#### 9.2 `BaseIntelParser` contract

```python
class BaseIntelParser(ABC):
    @property
    def key(self) -> str: ...       # dict key in SessionIntelligence
    def parse(self, text: str) -> Any: ...
```

`IntelligenceExtractor` iterates all registered parsers, merges results into a `SessionIntelligence` instance, and supports third-party parser registration via `extractor.register_parser(parser)`.

#### 9.3 Verification Results

| Test | Status |
|---|---|
| `test_user_parser` | ✅ PASSED |
| `test_kernel_parser` | ✅ PASSED |
| `test_port_parser` | ✅ PASSED |
| `test_banner_parser` | ✅ PASSED |
| `test_binary_parser` | ✅ PASSED |
| `test_suid_parser` | ✅ PASSED |
| `test_strings_parser` | ✅ PASSED |
| `test_flag_parser` | ✅ PASSED |
| `test_filesystem_parser` | ✅ PASSED |
| `test_intelligence_extractor_e2e` | ✅ PASSED |
| `test_custom_parser_plugin_registration` | ✅ PASSED |

---

### 10. Phase 5 — Prompt Engineering Subsystem (`core/prompts/`)

The monolithic `PromptBuilder` was decomposed into a composable, token-budgeted prompt pipeline under `core/prompts/`.

#### 10.1 Architecture

```
core/prompts/
  budget.py          — PromptBudget: dynamic token allocation across 4 sections
  history_builder.py — HistoryBuilder: rolling summary + semantic deduplication
  intel_builder.py   — IntelBuilder: sandboxed evidence serialization
  knowledge_builder.py — KnowledgeBuilder: deduplicated RAG context compilation
  scenario_builder.py  — ScenarioBuilder: YAML attack step formatting
  templates.py       — PromptTemplate + ModelFamily + detect_model_family()
  prompt_builder.py  — PromptBuilder: unified coordinator (DI-injectable sub-components)
  manager.py         — PromptManager: backward-compatibility wrapper
```

#### 10.2 `PromptBudget` Allocation

`PromptBudget` divides the total token budget across 4 named slots (intel, history, RAG, tactics). Sections that undershoot their allocation return surplus proportionally to other sections, preventing wasted context space.

#### 10.3 Multi-Model Template Support

`detect_model_family()` auto-detects the chat template from a model name or HuggingFace repo ID:

| Pattern | Family | Template |
|---|---|---|
| `qwen`, `cybersec-qwen*` | `QWEN` | `<\|im_start\|>` / `<\|im_end\|>` (ChatML) |
| `llama`, `tinyllama`, `alpaca`, `cybersec-model*`, `darklight*` | `LLAMA` | `<\|begin_of_text\|>` / `<\|eot_id\|>` |
| `mistral`, `zephyr`, `mixtral` | `MISTRAL` | `[INST]` / `[/INST]` |
| `gemma` | `GEMMA` | `<start_of_turn>` / `<end_of_turn>` |
| `deepseek` | `DEEPSEEK` | `<｜begin of sentence｜>` / `<｜thought｜>` |
| anything else | `STANDARD` | Raw text fallback |

> **DarkLight6996 Adapters**: `CyberSec-Model-V2` (TinyLlama base) → `LLAMA`; `CyberSec-Qwen-V1` (Qwen base) → `QWEN`.

#### 10.4 Circular Import Fix

`core/builders/__init__.py` no longer re-exports `PromptBuilder`. The import chain:

```
core.builders.__init__ → core.builders.prompt_builder (alias)
                       → core.prompts.prompt_builder
                       → core.builders.context_builder
                       → core.builders.__init__  ← CIRCULAR
```

was broken by removing the `PromptBuilder` export from `builders/__init__.py`. `PromptBuilder` is now imported directly from `core.prompts` or `core.builders.prompt_builder`.

#### 10.5 Verification Results

| Test | Status |
|---|---|
| `test_prompt_budget_allocation_and_surplus` | ✅ PASSED |
| `test_history_builder_dedup_and_rolling_summary` | ✅ PASSED |
| `test_intel_builder_sandboxing_and_deduplication` | ✅ PASSED |
| `test_intel_builder_with_session_intelligence_model` | ✅ PASSED |
| `test_knowledge_builder_dedup` | ✅ PASSED |
| `test_scenario_builder` | ✅ PASSED |
| `test_multi_model_templates` | ✅ PASSED |
| `test_detect_model_family` | ✅ PASSED |
| `test_prompt_builder_full_pipeline` | ✅ PASSED |
| `test_prompt_manager_backward_compatibility` | ✅ PASSED |

---

### 11. Phase 7 — Tool Calling Subsystem (`core/tool_registry.py`)

An extensible, schema-driven tool calling subsystem was introduced to transform the co-pilot from a reactive chatbot into an autonomous offensive security agent.

#### 11.1 Architecture

```
core/tool_registry.py
  ToolParameter   — Strongly typed parameter definition (type, desc, required, enum)
  ToolDefinition  — Tool metadata and OpenAI function calling schema generator
  ToolResult      — Standardized execution wrapper (success, output, error, latency)
  ToolRegistry    — Thread-safe tool catalog, validator, and dispatcher
```

#### 11.2 Built-In Offensive Operations Tools

| Tool | Parameters | Purpose |
|---|---|---|
| `rag_search` | `query: str`, `top_k: int` | Vector search over cybersecurity manuals & playbooks with reranking |
| `yaml_lookup` | `query: str`, `category: str`, `top_k: int` | Semantic search over YAML assessment playbooks |
| `command_preview` | `command: str`, `variables: dict` | Preview and safety-check shell commands with variable interpolation |
| `session_memory_read` | `session_id: str`, `limit: int` | Read multi-turn conversation history from SQLite |
| `intelligence_extract` | `raw_output: str` | Extract user, kernel, port, binary, flag intel from terminal dumps |

#### 11.3 OpenAI Function Calling Schema Export

`ToolRegistry.get_openai_tools()` exports all registered tools into OpenAI-compatible tool specifications (`{"type": "function", "function": {...}}`), ready for use with tool-calling models like Qwen 2.5 or frontier models.

#### 11.4 Verification Results

| Test | Status |
|---|---|
| `test_parameter_and_tool_definition_schema` | ✅ PASSED |
| `test_tool_execution_success` | ✅ PASSED |
| `test_tool_execution_missing_required_param` | ✅ PASSED |
| `test_tool_execution_exception_handling` | ✅ PASSED |
| `test_openai_tools_schema_export` | ✅ PASSED |
| `test_registry_duplicate_registration` | ✅ PASSED |
| `test_rag_search_builtin_tool` | ✅ PASSED |
| `test_yaml_lookup_builtin_tool` | ✅ PASSED |
| `test_command_preview_builtin_tool` | ✅ PASSED |
| `test_session_memory_read_builtin_tool` | ✅ PASSED |
| `test_intelligence_extract_builtin_tool` | ✅ PASSED |
| `test_custom_tool_registration` | ✅ PASSED |

---

### 12. Phase 8 — Local Agent Framework (`core/agent.py`)

A local agent orchestrator (`CyberAgent`) coordinates conversational AI, deterministic reasoning, RAG context, and autonomous tool calling into a unified agentic runtime.

#### 12.1 Architecture

```
core/agent.py
  AgentRequest   — Strongly typed input contract (query, session_id, session_state, max_iterations)
  AgentResponse  — Strongly typed output contract (text, intent, tools_used, confidence, latency)
  CyberAgent     — Autonomous offensive security agent coordinator
  get_agent()    — Session-scoped agent registry singleton
```

#### 12.2 Key Features

- **Dual-Engine Execution**: Uses primary local LLM (`LLMService`) for dynamic reasoning when available; automatically falls back to deterministic 8-branch `IntentRouter` when offline or on CPU.
- **Autonomous Tool Dispatching**: If the model emits tool calls (in JSON format or structured action blocks), the agent dispatches tools via `ToolRegistry` and loops up to `max_tool_iterations` times.
- **State & Memory Synchronization**: Automatically updates session state, registers extracted intelligence, and persists conversational exchanges to SQLite via `MemoryService`.
- **Per-Session Agent Singletons**: `get_agent(session_id)` ensures thread-safe, isolated state per engagement.

#### 12.3 Verification Results

| Test | Status |
|---|---|
| `test_agent_initialization` | ✅ PASSED |
| `test_agent_lazy_service_loading` | ✅ PASSED |
| `test_agent_run_deterministic_fallback` | ✅ PASSED |
| `test_agent_run_steering_intent` | ✅ PASSED |
| `test_agent_run_reset_intent` | ✅ PASSED |
| `test_agent_run_recon_intent` | ✅ PASSED |
| `test_agent_run_llm_primary_success` | ✅ PASSED |
| `test_agent_run_llm_empty_fallback` | ✅ PASSED |
| `test_agent_tool_execution_loop` | ✅ PASSED |
| `test_agent_max_tool_iterations_guard` | ✅ PASSED |
| `test_agent_session_memory_persistence` | ✅ PASSED |
| `test_agent_intelligence_extraction_in_pipeline` | ✅ PASSED |
| `test_get_agent_singleton_registry` | ✅ PASSED |
| `test_agent_response_metadata_and_latency` | ✅ PASSED |

---

### 13. Full Test Suite Status (September 2026)

**70 / 70 tests passing** across 9 test modules:

| Module | Tests | Status |
|---|---|---|
| `test_embedding_service.py` | 7 | ✅ All passed |
| `test_yaml_index.py` | 5 | ✅ All passed |
| `test_intelligence_parsers.py` | 11 | ✅ All passed |
| `test_prompt_subsystem.py` | 10 | ✅ All passed |
| `test_rag_pipeline.py` | 7 | ✅ All passed |
| `test_core_modules.py` | 0 (import only) | ✅ No errors |
| `test_cli_architecture.py` | 4 | ✅ All passed |
| `test_tool_registry.py` | 12 | ✅ All passed |
| `test_agent.py` | 14 | ✅ All passed |

Run: `python -m pytest ll-finetuning/rag/core/ -v --tb=short`



**Target Audience:** Non-technical stakeholders, compliance auditors, security operations center (SOC) analysts, and newly onboarded operators.

---

# 📖 High-Level Functional Glossary

### 1. Data Ingestion & Extraction Engine (`ll-finetuning/Generic Scripts`)
* **Real-World Purpose:** Ingests raw cybersecurity write-ups, vulnerability bulletins, and training manuals to systematically convert unstructured technical literature into structured instruction pairs. This ensures downstream artificial intelligence models are trained on clean, standardized, and repeatable security knowledge.

### 2. Scenario & Knowledge Vault (`ll-finetuning/rag/scenarios`, `ll-finetuning/rag/corpus`, `ll-finetuning/data`)
* **Real-World Purpose:** Functions as the centralized corporate knowledge repository, housing over 200 standardized offensive/defensive playbooks, compliance audit standards (such as HIPAA, PCI-DSS, SOC 2, and ISO 27001), and penetration testing methodologies. It establishes a single source of truth for organizational security operations and technical compliance requirements.

### 3. Neural Vector Search & Indexing Engine (`ll-finetuning/rag/faiss_index`, `ll-finetuning/rag/yaml_index`, `build_index.py`)
* **Real-World Purpose:** Translates written technical guidance and attack playbooks into high-dimensional semantic search indexes. This enables the system to instantly locate the exact security procedures relevant to an operator's real-time inquiry, even when phrased in conversational language.

### 4. Scenario Registry & Schema Validation Engine (`yaml_loader.py`, `yaml_registry.py`)
* **Real-World Purpose:** Validates, catalogues, and indexes operational scenario playbooks to guarantee that every loaded security procedure strictly meets corporate data and formatting standards. It protects operational pipelines from malformed instructions, missing safety metadata, or broken command definitions.

### 5. Assessment Intent Router & Context Merger (`assessment_router.py`)
* **Real-World Purpose:** Interprets high-level user security queries to match them with the most relevant technical scenarios and compliance standards. It fuses structured operational playbook steps with unstructured background guidance into a single, cohesive context package for execution.

### 6. Automated Security Planner Engine (`planner.py`)
* **Real-World Purpose:** Translates merged assessment context and target constraints into ordered, phased, and risk-managed execution roadmaps. It provides security teams with an auditable, step-by-step game plan prior to running any live operational commands.

### 7. Dynamic Parameter Engine (`variable_engine.py`)
* **Real-World Purpose:** Manages and safely substitutes real-time target parameters—such as host IP addresses, domain names, user credentials, and network ports—into standardized command templates. This eliminates human typing errors and prevents configuration drift during fast-paced security assessments.

### 8. Safe Command Preview & Execution Subsystem (`command_executor.py`)
* **Real-World Purpose:** Acts as a strictly governed execution broker that previews, evaluates safety parameters, enforces execution timeouts, and captures output logs for diagnostic commands. It ensures operations occur within authorized boundaries and prevents runaway or destructive commands.

### 9. Evidence Vault & Audit Trail Manager (`evidence_manager.py`, `memory.py`)
* **Real-World Purpose:** Automatically records all command executions, discoveries, interaction sessions, and operational logs into persistent, structured storage. This provides compliance auditors and incident responders with an immutable, forensically sound audit trail of all assessment activities.

### 10. Dual-Environment RAG Inference Pipeline (`rag_engine.py`, `rag_pipeline_cpu_final.py`, `rag_pipeline_gpu_final.py`, `model_router.py`)
* **Real-World Purpose:** Dynamically routes queries and contextual intelligence to optimal machine learning models across either low-power CPU environments (e.g., FLAN-T5) or high-performance GPU infrastructure (e.g., Mistral-7B). It guarantees responsive, accurate, and factually grounded security guidance tailored to available compute resources.

### 11. Parameter-Efficient Model Fine-Tuning Engine (`ll-finetuning/scripts`, `lora-adapters`)
* **Real-World Purpose:** Customizes and trains base Large Language Models on specialized cybersecurity terminology and tradecraft using Low-Rank Adaptation (LoRA). This achieves domain-expert AI capabilities at a fraction of the computational cost and energy footprint of full model training.

### 12. Environment Provisioning & Pipeline Verification Suite (`ll-finetuning/setup`, `verify_full_pipeline.py`, `test_core_modules.py`)
* **Real-World Purpose:** Standardizes platform requirements across Windows 11 and Ubuntu operating systems while executing automated end-to-end regression tests across all subsystems. It provides stakeholders with quantifiable verification that the software stack is resilient, correctly wired, and production-ready.

---

# MODULE 1: Data Ingestion & Extraction Engine (`ll-finetuning/Generic Scripts`)

---

## 1.1 `WebPage_chunker.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like an expert speed-reader tasked with clipping educational articles out of a cluttered newspaper. Instead of keeping the advertisements, header menus, interactive buttons, or comment sections, the reader neatly scissors out only the core training essay, cuts it into standardized, bite-sized study index cards with consecutive page overlaps, and stamps each card with a unique serial number and original article source citation.
* **Problem Solved:** Web-based cybersecurity training portals (such as PortSwigger Web Security Academy) embed valuable educational material alongside web navigation menus, marketing footers, script trackers, and dynamic UI wrappers. If fed directly into an AI system, this web noise contaminates search relevance and distorts model responses. This script systematically extracts, cleans, fragments, and formats web articles into clean, searchable, citation-backed text chunks for knowledge retrieval.

### 2. The Functional Journey (Plain-English Timeline)
1. **Input Ingestion:** The script opens the user-provided text file and scans every row to identify prospective website targets.
2. **Protocol Validation:** It verifies each row, skipping blank lines or comments and isolating valid URLs that begin with `http`.
3. **Throttled Network Request:** The script contacts each target web server one by one using a custom browser user-agent header. It pauses for a mandatory 2-second cooldown between requests to prevent overwhelming the target web server.
4. **Noise Decontamination:** Upon receiving the HTML page, the parser isolates the core educational article block and destroys all extraneous HTML elements, including headers, navigation menus, sidebars, footers, forms, buttons, embedded iframes, styling, and JavaScript.
5. **Content Verification:** The script checks the extracted plain text against minimum substance criteria. If the extracted body is empty or shorter than 200 characters, it drops the page and issues a warning.
6. **Overlapping Segmentation:** The cleaned text is sliced into manageable text segments. Each segment captures approximately 700 tokens of content (~933 words), maintaining an overlap of 80 tokens (~106 words) with the preceding segment to prevent thoughts or sentences from being abruptly severed at chunk boundaries.
7. **Cryptographic Stamping:** The script constructs a header for each segment containing an 8-character MD5 hash identifier derived from the source URL and sequence index, along with the source URL, article title, and category section.
8. **File Persistence:** The formatted text blocks are written consecutively to the output file, finalizing a RAG-ready document corpus.

### 3. Core Business Rules & System Constraints
* **URL Filtering:** Input lines must start with `http` to be processed; non-matching lines are silently ignored.
* **Network Boundary:** 15-second network timeout per request. On connection failure or non-200 HTTP code, logs warning and moves to next URL without crashing.
* **Extraction Targets:** Looks for `<main>`, `<article>`, or `<div class="content|body|learning">`. If missing, aborts page and logs `⚠️ No main content found`.
* **Length Constraint:** Plain text must be $\ge 200$ characters. Shorter bodies are discarded.
* **Rate Limiting:** 2.0-second delay (`time.sleep(2.0)`) enforced between successive web requests.
* **Token Math:** Word token approximation calculated as $\text{words} = \text{tokens} / 0.75$. Default chunk size = 700 tokens (~933 words); overlap = 80 tokens (~106 words).

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `requests`, `bs4` (BeautifulSoup), `re`, `time`, `hashlib`, `argparse`.
* **CLI Arguments:** `-i / --input` (URL list path), `-o / --output` (RAG corpus output file), `--chunk-size` (default 700), `--overlap` (default 80).
* **Chunk Header Format:**
  ```text
  ### BEGIN CHUNK ###
  # ID: 8-char-hex
  # SOURCE: URL
  # TITLE: Page Title
  # SECTION: Learning Module
  ### CONTENT ###
  ...
  ### END CHUNK ###
  ```

---

## 1.2 `pdf_to_fintune.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like an experienced corporate trainer converting an exhaustive 500-page security compliance manual into a deck of interactive flashcards for new employees. The trainer reads each section, identifies whether it is discussing a technical architecture, a PowerShell command, a compliance standard, or an operational process, and writes a tailored question on the front of the card with the exact manual excerpt on the back.
* **Problem Solved:** Raw PDF manuals and security guides are stored as unstructured visual page layouts. LLMs cannot learn effectively from raw continuous text dumps; they require paired "instruction-and-response" training formats (such as the Stanford Alpaca format). This script bridges this gap by extracting text from PDFs, cleaning low-level character artifacts, chunking by semantic paragraphs, classifying the technical domain, and generating context-aware instructional prompts automatically.

### 2. The Functional Journey (Plain-English Timeline)
1. **File Discovery & Path Traversal:** Discovers all `.pdf` files within specified file paths or directory trees.
2. **Page-Level PDF Extraction:** Opens each PDF via PyMuPDF (`fitz`) and extracts text page by page.
3. **Text Sanitization:** Strips unprintable ASCII control bytes (`[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`), collapses whitespace, and normalizes breaks.
4. **Token-Aware Paragraph Segmentation:** Splits text by paragraph breaks and measures token lengths using OpenAI `tiktoken` (`cl100k_base`), maintaining a 512-token limit per chunk and a 64-token backward overlap.
5. **Content Domain Classification:** Scans vocabulary to classify into `scripting`, `framework`, `procedural`, `technical`, or `general`.
6. **Instruction Synthesis:** Selects a domain-specific instruction prompt template.
7. **Alpaca Record Formatting & Output:** Formats each entry into `{"instruction": ..., "input": "", "output": ...}` and appends to the destination `.jsonl` file.

### 3. Core Business Rules & System Constraints
* **Path Validation:** Fails with exit code 1 if no valid PDF documents are found in input paths.
* **Clean Mode:** `--clean` flag removes pre-existing output files before appending.
* **Chunk Boundaries:** Strict 512 token ceiling per chunk with 64 token backward preservation.
* **Domain Priority Order:** Evaluated sequentially: (1) Scripting $\rightarrow$ (2) Framework $\rightarrow$ (3) Procedural $\rightarrow$ (4) Technical $\rightarrow$ (5) General fallback.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `fitz` (`PyMuPDF`), `tiktoken`, `tqdm`, `argparse`, `json`, `re`.
* **Output Format:** Stanford Alpaca JSONL (`{"instruction": str, "input": str, "output": str}`).
* **Edge Cases:** Scanned/image-only PDFs with no OCR yield empty text and are gracefully skipped without exceptions.

---

## 1.3 `pdf_to_rag_corpus.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like an archivist microfilming an encyclopedia for an intelligence library. The archivist examines each page under high resolution; if standard text scanning fails due to complex multi-column layouts, they switch to geometric block scanning. They then stamp each microfiche card with the book's title, exact physical page number, and token density estimate before cataloging it in the searchable archives.
* **Problem Solved:** Security standard manuals (e.g., NIST, ISO, CIS benchmarks) are distributed as multi-page PDFs with diverse visual layouts (tables, sidebars, multi-column text). Naive text extraction often mashes adjacent columns together or misses text blocks completely. This script dynamically adapts its extraction strategy (standard stream vs. geometric layout blocks), decontaminates character artifacts, and slices text into page-attributed RAG chunks.

### 2. The Functional Journey (Plain-English Timeline)
1. **Document Initialization:** Initializes configuration (600 tokens/chunk, 90 tokens overlap, 50-character minimum page threshold, `auto` mode) and opens PDF.
2. **Adaptive Page Extraction:** Extracts text stream; if string length $< 20$ chars, automatically switches to layout block extraction (`get_text("blocks")`).
3. **Character Decontamination:** Strips binary control characters and standardizes whitespace.
4. **Minimum Viability Filter:** Skips pages with less than 50 total characters.
5. **Dual-Tier Segmentation:** Splits by paragraphs; falls back to sentence punctuation boundaries (`.`, `!`, `?`) if no paragraph breaks exist.
6. **Sliding Character Window:** Aggregates text up to ~1,800 characters (600 tokens $\times 3$ chars/token) with a 270-character overlap buffer.
7. **Page-Attributed Header Stamping:** Wraps chunk in metadata header with document name, page number, chunk index, and estimated token count.
8. **Corpus File Writing:** Writes formatted blocks to the `.txt` output file.

### 3. Core Business Rules & System Constraints
* **Auto-Mode Fallback:** If extracted page text $< 20$ characters, layout block extraction is automatically engaged.
* **Page Cutoff:** Pages with $< 50$ characters after cleaning are omitted.
* **Paragraph Filter:** Paragraphs must be $> 10$ characters to be included in chunks.
* **Chunk Window:** 600 token window (~1,800 characters) with 90 token overlap (~270 characters).

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `fitz` (`PyMuPDF`), `argparse`, `logging`, `pathlib.Path`.
* **Output Format:** Plaintext file formatted with `### BEGIN CHUNK ###`, `# ID: <stem>_p<page>_c<idx>`, `# PAGE: <page>`, `# TOKENS: ~<tokens>`, and `### CONTENT ###`.

---

## 1.4 `rag_to_alpaca.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like an industrial recycling plant that takes pallets of shredded reference documents and transforms them into standardized question-and-answer exam books. It inspects each shredded passage, determines whether it looks like a recipe, a definition, a script, or a threat analysis, writes a custom exam prompt on top, and stamps the document citation beneath it.
* **Problem Solved:** While structured text files (`.txt` chunks) are ideal for vector search databases, generative AI models require structured instruction datasets (`.jsonl`) to learn cybersecurity reasoning. This script reads delimited RAG text files line-by-line using a memory-efficient state machine, parses embedded metadata, classifies the content type, and outputs fine-tuning instruction pairs.

### 2. The Functional Journey (Plain-English Timeline)
1. **State Machine Activation:** Initializes parser state and opens the input file in streaming generator mode.
2. **Chunk Boundary Detection:** Transitions state to active collection upon encountering `### BEGIN CHUNK ###`.
3. **Metadata Extraction:** Splits `# KEY: VALUE` comment headers into a metadata dictionary.
4. **Content Isolation:** Buffers all body lines until `### END CHUNK ###`.
5. **Length & Integrity Verification:** Discards chunks shorter than 50 or longer than 4,000 characters.
6. **Contextual Intent Classification:** Evaluates keywords to select appropriate prompt template (Procedural, Conceptual, Syntax, Risk/Threat, or General).
7. **Context Enrichment:** Embeds source and section citations into instruction and input fields.
8. **JSONL Serialization:** Serializes record to a JSON line and appends to output file.

### 3. Core Business Rules & System Constraints
* **Length Constraints:** Content must satisfy $50 \le \text{length} \le 4000$ characters. Out-of-bounds chunks are silently dropped.
* **Metadata Schema:** Expects `# KEY: VALUE` formatting inside chunk headers.
* **Memory Footprint:** Uses Python streaming generator pipeline ($O(1)$ memory consumption).

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `json`, `argparse`, `logging`, `pathlib.Path`.
* **Output Format:** JSON Lines (`{"instruction": str, "input": str, "output": str, ["_meta": dict]}`).

---

## 1.5 `rag_to_alpaca_multiple.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like an automated regional post office that collects mail sacks from dozens of local branch offices, unpacks each parcel, verifies that every letter meets standardized postal address formatting rules, and packages them into a single consolidated freight container ready for international delivery.
* **Problem Solved:** When building a large-scale training corpus, knowledge chunks are often generated across dozens of separate text files (one per manual, tool, or write-up). Manually converting each file one by one is slow and error-prone. This script automates folder-wide batch conversion, scanning an entire directory of `.txt` chunk files, validating chunk boundaries, and merging them into a single unified `.jsonl` fine-tuning dataset.

### 2. The Functional Journey (Plain-English Timeline)
1. **Directory Verification:** Validates input folder existence and halts if missing.
2. **Alphabetical File Queueing:** Discovers all `*.txt` files and sorts them alphabetically.
3. **Fault-Tolerant File Parsing:** Opens each file with UTF-8 decoding while suppressing invalid byte errors (`errors='ignore'`).
4. **Delimiter State Extraction:** Extracts `# ` metadata headers and content blocks between `### CONTENT ###` and `### END CHUNK ###`.
5. **Length Validation Filter:** Discards chunks under 50 characters.
6. **Contextual Prompt Formulation:** Matches heuristic keywords to format prompt instruction with source context.
7. **Record Serialization:** Appends valid JSON records to the master `.jsonl` file.
8. **Batch Metrics Reporting:** Logs total converted records across all directory files.

### 3. Core Business Rules & System Constraints
* **Input Directory:** Must exist and contain $\ge 1$ `.txt` files.
* **Encoding Resilience:** Uses `errors='ignore'` to prevent malformed encoding crashes.
* **Content Filter:** Drops chunks $< 50$ characters.
* **Key Flexibility:** Supports uppercase (`SOURCE`, `SECTION`) and lowercase (`source`, `section`) metadata tags.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `os`, `json`, `argparse`, `pathlib.Path`.
* **Output Format:** Consolidated `.jsonl` dataset aggregating multiple source files.

---

## 1.6 `writeup_to_corpus.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like an editor converting informal laboratory field notebooks written by security researchers into standardized, peer-reviewed textbook chapters. The editor takes raw notes detailing how a specific challenge or computer system was penetrated, strips out file artifacts, divides the narrative into readable 512-word segments with smooth overlapping transitions, and files each one into a dedicated archive.
* **Problem Solved:** Penetration testing write-ups (such as HackTheBox or CTF walkthroughs) are typically stored as unstructured, informal `.txt` files. They contain rich, real-world exploit chains but lack the standardized delimitations required by RAG vector stores. This script batch-processes directories of raw write-up text files, automatically derives human-readable titles, segments them into uniform word-count windows, and outputs standardized `_rag.txt` files for vector indexing.

### 2. The Functional Journey (Plain-English Timeline)
1. **Folder Verification:** Validates source directory and auto-creates output directory if absent.
2. **Writeup File Discovery:** Collects all `.txt` documents in the input path.
3. **Text Reading:** Reads file content with UTF-8 encoding; skips empty files.
4. **Title Derivation:** Converts filename (e.g. `forest_box.txt`) to clean Title Case (`Forest Box`).
5. **Word-Based Window Slicing:** Segments words into 512-word windows with 64-word overlaps.
6. **Metadata Stamping:** Wraps chunks in delimiter headers with derived ID, source title, and `Writeup` section tag.
7. **File Generation:** Saves output to `<filename>_rag.txt` in the target directory.
8. **Summary Reporting:** Logs created chunk counts per file and cumulative totals.

### 3. Core Business Rules & System Constraints
* **Empty File Rule:** Blank or whitespace-only files are skipped (`returns 0`).
* **Chunk Window Math:** 512 target words per chunk; step forward size = $512 - 64 = 448$ words.
* **Directory Auto-Creation:** `output_dir.mkdir(parents=True, exist_ok=True)` automatically creates output folder tree.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `argparse`, `pathlib.Path`, `logging`, `os`, `sys`.
* **Output Format:** Individual `<stem>_rag.txt` corpus files formatted for FAISS ingestion.

---

## 1.7 `check_hashes.py` (LoRA Adapter SHA-256 Hash Verification)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `check_hashes.py` like a high-security bank vault teller verifying the wax seal and serial numbers on an incoming armored transport bag against the master central bank ledger before placing it in the safety deposit box. If even a single digit does not match, the bag is quarantined as tampered.
* **Problem Solved:** When distributing and updating machine learning models and fine-tuned adapters, weight corruption, silent truncation during downloads, or supply-chain tampering can compromise model behavior. `check_hashes.py` computes chunked SHA-256 cryptographic checksums across local adapter safetensors and compares them directly against the upstream Hugging Face Hub repository metadata.

### 2. The Functional Journey (Plain-English Timeline)
1. **Repository Query:** Contacts the Hugging Face Hub API for the target model repository (`DarkLight6996/CyberSec-Model-V2` by default) to retrieve remote LFS file metadata.
2. **Remote Hash Extraction:** Extracts the official SHA-256 fingerprint from Hugging Face LFS metadata, or downloads the file directly to compute the authoritative reference checksum.
3. **Local Directory Discovery:** Scans all potential adapter storage roots (`lora-adapters/`, `lora-adapters-cpu/`, `lora-output/`, and workspace roots) for `adapter_model.safetensors`.
4. **Chunked Hashing:** Computes SHA-256 checksums in 8 MB binary chunks (`chunk := f.read(8 * 1024 * 1024)`) to prevent RAM exhaustion on large tensor files.
5. **Exact Match Comparison:** Compares local hashes against the Hugging Face Hub hash in a case-insensitive match evaluation.
6. **Audit Verification Output:** Displays a visual matching report indicating `MATCHES HF! (Live Version)` or `NO MATCH (DIFF)`.

### 3. Core Business Rules & System Constraints
* **Chunked Buffer Boundary:** Hashes files in 8 MB chunks to safely process multi-gigabyte models without high memory overhead.
* **Deduplication:** Uses a path resolution set to avoid calculating hashes multiple times for identical file references.
* **Failure Safety:** Falls back to direct Hub file download if LFS metadata attributes are missing or unparseable.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `argparse`, `hashlib`, `pathlib.Path`, `huggingface_hub.HfApi`, `huggingface_hub.hf_hub_download`, `huggingface_hub.model_info`.
* **Execution:** `python "Generic Scripts/check_hashes.py" --repo-id DarkLight6996/CyberSec-Model-V2`

---

## 1.8 `fix_jsonl.py` & `fix_jsonl2.py` (Dataset Sanitization & Repair)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of these repair tools like automated proofreaders for a publishing house. If an author submitted manuscript pages with broken quotation marks, missing paragraph separators, or escaped formatting tags, the automated proofreader cleans the formatting rules so the typesetting machine runs without jamming.
* **Problem Solved:** JSONL datasets collected from diverse sources often contain malformed escape sequences, unclosed braces, or trailing commas that cause dataset loaders to crash. `fix_jsonl.py` and `fix_jsonl2.py` parse and sanitize line-by-line JSON structures, enforcing valid JSON schema compliance.

### 2. The Functional Journey (Plain-English Timeline)
1. **File Reading:** Streams JSONL lines with UTF-8 encoding.
2. **Syntax Repair:** Cleans broken JSON delimiters, fixes escaped quotes, and validates `instruction`, `input`, and `output` keys.
3. **Validation Filter:** Drops lines that cannot be parsed into valid JSON objects.
4. **Clean Serialization:** Writes formatted JSON records back to disk with standard newline terminators.

### 3. Core Business Rules & System Constraints
* **Atomicity:** Validates every line independently; malformed lines do not corrupt valid adjacent entries.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** Standard library `json`, `os`, `sys`, `pathlib.Path`.

---

## 1.9 `add_scenarios.py` & `add_more_scenarios.py` (Scenario Generator Utilities)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of these scripts like legal template generators for corporate compliance officers. When a new cybersecurity standard or audit regulation is enacted, the generator produces standardized assessment questionnaire templates pre-populated with required audit fields.
* **Problem Solved:** Manually writing dozens of multi-step cybersecurity playbooks is time-consuming and error-prone. These generator utilities programmatically construct structured YAML playbooks and JSONL fine-tuning pairs from defined attack taxonomies.

### 2. The Functional Journey (Plain-English Timeline)
1. **Taxonomy Ingestion:** Reads attack taxonomies (AD, Cloud, Web, GRC).
2. **Template Synthesis:** Structures metadata, steps, commands, and expected outputs conforming to the scenario schema.
3. **Output Export:** Emits validated YAML files into `rag/scenarios/` and JSONL pairs into `data/`.

### 3. Core Business Rules & System Constraints
* **Schema Conformity:** Generated playbooks strictly contain `id`, `name`, `category`, `description`, and `steps`.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `json`, `yaml`, `pathlib.Path`.

---

# MODULE 2: RAG Core Orchestration & Schema Engine (`ll-finetuning/rag/core`)

---

## 2.1 `config.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `config.py` like the central mission control dispatch board of an international airport. It holds all master operational parameters: runways in use, gate assignments, radio frequencies, speed limits, and aircraft size classifications. Rather than every pilot and ground crew inventing their own rules, everyone references this single authoritative dashboard.
* **Problem Solved:** Complex AI and retrieval systems involve dozens of interdependent parameters (file paths, database locations, model names, token thresholds, retrieval depths). Hardcoding these values across individual files creates severe configuration drift, broken paths, and deployment failures across different environments. `config.py` provides a unified, cross-platform source of truth.

### 2. The Functional Journey (Plain-English Timeline)
1. **Dynamic Path Anchoring:** Automatically determines the absolute filesystem location of the `core/` folder and its parent `rag/` directory regardless of where the script is executed from.
2. **Corpus & Index Path Resolution:** Resolves absolute paths to text corpora (`corpus/`), FAISS vector indexes (`faiss_index/`), YAML scenario templates (`scenarios/`), and SQLite audit databases (`sessions.db`).
3. **Retrieval Depth Definition:** Sets two-stage search limits: initial broad vector retrieval (`TOP_K = 10`) and final reranked document retention (`FINAL_K = 3`).
4. **Conversational Memory Boundaries:** Defines rolling memory windows (`MEMORY_WINDOW = 6` conversational turns) and default session identifiers.
5. **Compute Profile Configuration:** Configures dedicated models and token capacities for low-power CPU mode (`google/flan-t5-large`, 2048 max input tokens, 200 max new tokens) and high-performance GPU mode (`mistralai/Mistral-7B-Instruct-v0.2`, 2048 max input tokens, 512 max new tokens).
6. **Neural Model Specifiers:** Specifies the standard vector embedding model (`sentence-transformers/all-MiniLM-L6-v2`) and cross-encoder neural reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`).

### 3. Core Business Rules & System Constraints
* **Path Resolution:** All internal directory paths are dynamically computed relative to `__file__`, guaranteeing functional execution across Windows and Linux environments without hardcoded drive letters.
* **Two-Tier Retrieval Ratio:** Enforces a 10:3 retrieval-to-rerank filtering ratio ($10 \rightarrow 3$), balancing retrieval recall against computational inference cost.
* **Context Budgeting:** Limits maximum content character payload to 3,000 characters and caps model token inputs at 2,048 tokens to prevent context window exhaustion.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Constants Defined:** `CORE_DIR`, `RAG_DIR`, `CORPUS_DIR`, `FAISS_INDEX_DIR`, `INDEX_PATH`, `META_PATH`, `SCENARIOS_DIR`, `YAML_INDEX_DIR`, `YAML_INDEX_PATH`, `YAML_META_PATH`, `DATABASE_PATH`, `TOP_K`, `FINAL_K`, `MAX_CONTENT_CHARS`, `MEMORY_WINDOW`, `DEFAULT_SESSION`, `CPU_SESSION_ID`, `GPU_SESSION_ID`, `CPU_MODEL_NAME`, `CPU_MAX_INPUT_TOKENS`, `CPU_MAX_NEW_TOKENS`, `GPU_MODEL_NAME`, `GPU_MAX_INPUT_TOKENS`, `GPU_MAX_NEW_TOKENS`, `EMBED_MODEL`, `RERANK_MODEL`.
* **Dependencies:** Standard library `os`.

---

## 2.2 `yaml_loader.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `YAMLLoader` like the head librarian at a legal reference library. When the library opens, the head librarian unlocks the vaults, instructs the cataloging clerks (`YAMLRegistry`) to inventory every law book on the shelves, verifies that no books have torn pages or missing titles, and stands ready at the front desk to quickly fetch books whenever lawyers or judges request them.
* **Problem Solved:** Assessment scenarios are stored as hundreds of standalone YAML files. Other subsystems (routers, planners, API endpoints) need a simple, high-level interface to load, reload, inspect, and filter templates without having to manage raw dictionary indexing or low-level file traversal. `YAMLLoader` provides a clean lifecycle wrapper around the underlying registry.

### 2. The Functional Journey (Plain-English Timeline)
1. **Configuration Initialization:** Accepts the root directory path of the YAML scenario library, recursion preferences, and verbosity flags.
2. **Library Loading Execution:** Instantiates `YAMLRegistry`, triggers disk discovery, reads and parses all YAML files, and tracks success counts.
3. **Integrity Validation:** Triggers structural validation across all loaded templates, flagging missing mandatory fields or malformed step definitions.
4. **Summary & Health Reporting:** Prints summary statistics including total loaded templates, categories, frameworks, and any failed files.
5. **On-Demand Search Delegation:** Exposes fast lookup shortcuts (`get`, `require`, `search_by_category`, `search_by_tag`, `filter`) directly to consuming applications.
6. **Hot Reloading:** Provides a `reload()` mechanism that resets cached state and refreshes templates from disk when playbooks are modified at runtime.

### 3. Core Business Rules & System Constraints
* **State Gate:** Methods (`get`, `require`, `all`, `stats`, `validate`) raise a `RuntimeError("Loader not loaded. Call load() first.")` if called before `load()`.
* **Strict Lookup Requirement:** `require(template_id)` raises a `KeyError` if a requested template ID is absent from the registry, preventing silent execution failures.
* **Non-Blocking Warnings:** Templates with validation errors are still loaded to preserve availability, but their issues are logged and tracked inside `__validation_errors`.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `pathlib.Path`, `core.yaml_registry.YAMLRegistry`.
* **Interface Methods:** `load()`, `reload()`, `is_loaded()`, `get(id)`, `require(id)`, `all()`, `stats()`, `summary()`, `search_by_category()`, `search_by_subcategory()`, `search_by_tag()`, `search_by_platform()`, `search_by_framework()`, `search_by_difficulty()`, `filter(**kwargs)`, `validate()`.

---

## 2.3 `yaml_registry.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `YAMLRegistry` like the master inventory and indexing system of a massive warehouse. When thousands of crates arrive, the warehouse manager inspects each crate's manifest, verifies required barcodes and warning labels, logs defective crates in an issue ledger, and indexes every item across multiple cross-referenced filing cabinets: one by category, one by manufacturer, one by destination, and one by hazard level.
* **Problem Solved:** Loading 200+ complex operational YAML playbooks without indexing would require slow $O(N)$ sequential file scans every time an operator asks for a specific test (e.g., "Show all Azure Privilege Escalation playbooks" or "Find all HIPAA compliance tests"). `YAMLRegistry` parses files safely, validates schema conformity, and constructs high-speed inverted lookup indexes in memory.

### 2. The Functional Journey (Plain-English Timeline)
1. **Directory Discovery:** Scans the scenario folder (flat or recursively) for files ending in `.yaml` or `.yml`.
2. **Safe Parsing:** Reads each file using `yaml.safe_load`. If syntax errors occur, it logs an error and records the file in `failed_files`.
3. **Mandatory Schema Validation:** Verifies the presence of required root keys: `id`, `name`, `category`, `description`, and `steps`.
4. **Step Structure Inspection:** Validates that `steps` is formatted as a list of dictionaries and verifies that each step contains a `phase` declaration.
5. **Multi-Dimensional Index Construction:** Indexes the template ID across six specialized inverted search indexes:
   * Category Index (e.g., `cloud_security`, `web_application`, `active_directory`).
   * Subcategory Index (e.g., `privilege_escalation`, `reconnaissance`, `credential_access`).
   * Tag Index (e.g., `azure`, `kerberos`, `sql_injection`, `mimikatz`).
   * Platform Index (e.g., `azure`, `aws`, `windows`, `linux`, `kubernetes`).
   * Compliance Framework Index (e.g., `hipaa`, `pci_dss`, `soc2`, `iso27001`, `cis`).
   * Difficulty Index (e.g., `beginner`, `intermediate`, `advanced`).
6. **Query & Filter Execution:** Serves sub-millisecond lookups and composite multi-field filters for downstream planning and retrieval.

### 3. Core Business Rules & System Constraints
* **Mandatory Field Rule:** Every scenario must define `id`, `name`, `category`, `description`, and `steps`. Missing fields trigger validation warnings.
* **Step Schema Constraint:** `steps` must be a list of dicts where each item explicitly contains `phase`.
* **Case-Insensitive Indexing:** All index entries (categories, tags, platforms, frameworks) are normalized to lowercase, preventing search misses caused by typographical differences.
* **Fault Tolerance:** A single corrupted YAML file will never abort the registry loading process; errors are captured in `failed_files` while all valid templates continue loading.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `pathlib.Path`, `collections.defaultdict`, `yaml` (PyYAML).
* **Data Structures:** Primary store `self.templates` (`Dict[str, dict]`), secondary inverted indexes `defaultdict(set)`.
* **Supported Filter Attributes:** `category`, `subcategory`, `difficulty`, `platform`, `framework`, `tags`.

---

## 2.4 `yaml_indexer.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `YAMLIndexer` like a team of intelligence analysts creating a mathematical encyclopedia of military field tactics. The analysts read every operational playbook, synthesize its tactical objectives, compliance controls, and required tools into a cohesive tactical summary, and feed it into a supercomputer that converts the meaning into a mathematical coordinate in space, saving the finished coordinate map onto high-speed solid-state storage.
* **Problem Solved:** Keyword search fails when an operator uses synonyms (e.g., searching for "stealing cloud credentials" when the playbook is titled "Azure Service Principal Misconfiguration"). `YAMLIndexer` converts all YAML scenario metadata into high-dimensional dense vector embeddings and builds a standalone FAISS semantic index file for fast vector search.

### 2. The Functional Journey (Plain-English Timeline)
1. **Environment Initialization:** Resolves scenario and index directory paths, instantiates the `SentenceTransformer` embedding model, and loads the YAML scenario library.
2. **Searchable Document Synthesis:** For each YAML template, it compiles a rich plain-text representation combining:
   * Basic attributes: `id`, `name`, `category`, `subcategory`, `description`, `difficulty`.
   * Tactical objectives: `learning_objectives`.
   * Classifications & Platforms: `tags`, `platforms`, `mitre_attack` tactics/techniques.
   * Compliance mappings: `standard`, `control`, `description` for all mapped standards.
3. **Batch Vector Embedding:** Generates 384-dimensional dense semantic vector embeddings across all synthesized scenario documents.
4. **L2 Vector Normalization:** Normalizes all embedding vectors to unit length ($L_2$ norm) to enable cosine similarity calculations via inner product.
5. **FAISS Vector Index Assembly:** Constructs a `faiss.IndexFlatIP` (Inner Product) vector index and populates it with normalized embeddings.
6. **Artifact Serialization:** Writes the FAISS index binary to `yaml.index` and serializes template metadata to `yaml_meta.pkl`.

### 3. Core Business Rules & System Constraints
* **Vector Dimension:** Fixed at 384 dimensions matching `sentence-transformers/all-MiniLM-L6-v2`.
* **Cosine Similarity Metric:** Uses `IndexFlatIP` combined with `faiss.normalize_L2` to ensure cosine similarity scoring between $0.0$ and $1.0$.
* **Missing Scenarios Guard:** If no YAML templates are discovered in the target directory, index generation aborts gracefully with a warning without creating corrupted index files.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `faiss`, `numpy`, `pickle`, `sentence_transformers.SentenceTransformer`, `pathlib.Path`, `core.yaml_loader.YAMLLoader`, `core.config`.
* **Persisted Files:** `rag/yaml_index/yaml.index` (binary FAISS index), `rag/yaml_index/yaml_meta.pkl` (pickled list of raw template dictionaries).

---

## 2.5 `yaml_search.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `YAMLSearch` like an expert tactical advisor stationed in an emergency command room. When a field commander shouts a problem (e.g., "The target network has an open Redis database with no password!"), the advisor instantly computes the semantic intent of the request, compares it against the playbook coordinate map, and hands the commander the top 3 most relevant operational playbooks ranked by relevance.
* **Problem Solved:** Operators describe security tasks in varied natural language (e.g., "audit cloud storage permissions", "test s3 bucket leaks", "aws data exfil"). `YAMLSearch` performs vector similarity searches directly against loaded scenario playbooks, ranking and scoring the best matching procedures for any conversational prompt.

### 2. The Functional Journey (Plain-English Timeline)
1. **Engine Initialization:** Binds to an active `YAMLRegistry` and initializes the `SentenceTransformer` embedding model.
2. **In-Memory Index Compilation:** If templates are present in the registry, it transforms each template into a searchable textual summary (including title, category, description, tags, and step actions) and encodes them into normalized float32 vectors.
3. **Query Vectorization:** When a query arrives (e.g., *"Assess Azure tenant for privilege escalation"*), the query string is converted into a normalized 384-dimensional semantic embedding vector.
4. **Cosine Similarity Computation:** Calculates the dot product between the query vector and all template vectors (`np.dot(self._embeddings, query_vec)`).
5. **Ranked Sorting & Thresholding:** Identifies the top-k highest scoring indices using descending array sorting (`np.argsort(scores)[::-1][:top_k]`).
6. **Result Formatting:** Injects the computed similarity score into each template record under the `_score` key and returns the ranked list.

### 3. Core Business Rules & System Constraints
* **Auto-Indexing:** Automatically triggers index building if the registry is already loaded upon initialization, or on the first search invocation if not yet indexed.
* **Relevance Scoring:** Result templates include a `_score` float representing cosine similarity (higher is more relevant).
* **Graceful Degradation:** Returns an empty list (`[]`) if the registry contains zero templates or embeddings failed to generate.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `numpy`, `typing`, `sentence_transformers.SentenceTransformer`, `core.yaml_registry.YAMLRegistry`, `core.config.EMBED_MODEL`.
* **Input Parameters:** `query` (search string), `top_k` (number of templates to return, default 3).
* **Return Format:** List of raw template dictionaries enriched with `_score` float attribute.

---

## 2.6 `assessment_router.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `AssessmentRouter` like the chief triage physician in an emergency room. When a patient arrives with complex symptoms, the triage physician simultaneously sends two orders: one to the medical library to pull research papers explaining the underlying pathology (Document Knowledge), and one to the surgical ward to pull the exact step-by-step operating checklist (Procedural Playbooks). The physician then staples both packets together so the surgical team has full context.
* **Problem Solved:** Answering a complex security assessment request requires two distinct types of information: *declarative knowledge* (explaining what a vulnerability or compliance standard means) and *procedural knowledge* (exact commands and phases to execute an audit). `AssessmentRouter` coordinates simultaneous retrieval from both the Document RAG Engine and the YAML Procedure Search Engine, merging them into a unified context payload.

### 2. The Functional Journey (Plain-English Timeline)
1. **Query Ingestion:** Receives the operator's high-level assessment inquiry along with configurable retrieval quotas (`top_k_docs`, `top_k_yamls`).
2. **Document Retrieval Dispatch:** If a `RAGEngine` is configured, it queries the document vector index, executes cross-encoder reranking, and constructs formatted source citations. If unavailable, it captures a descriptive status string.
3. **Procedural Scenario Dispatch:** If a `YAMLSearch` engine is configured, it queries the scenario semantic index to identify the top matching YAML playbooks.
4. **Context Fusion:** Merges the declarative text snippets and procedural scenario definitions into a unified `MergedContext` data object.
5. **Prompt Formatting:** Provides a unified formatting method (`format_combined_prompt_context()`) that organizes the retrieved intelligence into clearly delimited sections (`=== DOCUMENTATION CONTEXT ===` and `=== RELEVANT ASSESSMENT PROCEDURES ===`) ready for LLM consumption.

### 3. Core Business Rules & System Constraints
* **Independent Failure Resilience:** If either the document engine or the YAML search engine fails or is unconfigured, the router captures the error gracefully and continues fulfilling the request with the available subsystem.
* **Dynamic Retrieval Override:** Temporarily overrides `final_k` on the RAG engine during execution and restores the original setting upon completion.
* **Audit Serialization:** `MergedContext` provides a `.to_dict()` method for structured logging, session tracking, and audit trail serialization.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `typing`, `core.rag_engine.RAGEngine`, `core.yaml_search.YAMLSearch`.
* **Primary Classes:**
  * `MergedContext`: Container object holding `query`, `documents`, `doc_context_str`, and `yaml_templates`.
  * `AssessmentRouter`: Core router with `.route(query, top_k_docs=3, top_k_yamls=3)` method.

---

## 2.7 `planner.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `Planner` like an elite construction project manager combining individual blueprints from electrical, plumbing, and structural subcontractors into a single, chronologically ordered master construction schedule. The manager ensures that foundation excavation happens before framing, framing happens before wiring, and inspections happen before painting, stamping each task with a sequential job number.
* **Problem Solved:** When a security assessment requires multiple playbooks (e.g., external reconnaissance, Active Directory enumeration, and privilege escalation), running the steps in arbitrary order causes operational failure (e.g., trying to escalate privileges before finding the server IP). `Planner` merges multiple playbook templates and reorganizes all steps into an industry-standard cybersecurity methodology order.

### 2. The Functional Journey (Plain-English Timeline)
1. **Template Extraction:** Ingests a collection of matching YAML scenario templates and extracts all operational step blocks.
2. **Source Lineage Tracking:** Annotates every individual step with its originating template ID and human-readable playbook name (`source_yaml_id`, `source_yaml_name`).
3. **Methodology Phase Ranking:** Evaluates the `phase` attribute of every step against a standardized cybersecurity lifecycle hierarchy:
   * Rank 0: Reconnaissance / Recon
   * Rank 1: Enumeration
   * Rank 2: Initial Access
   * Rank 3: Privilege Escalation / PrivEsc
   * Rank 4: Lateral Movement
   * Rank 5: Persistence
   * Rank 6: Exfiltration
   * Rank 7: Evidence Collection
   * Rank 8: Cleanup
   * Rank 99: Fallback for custom/unrecognized phases
4. **Hierarchical Sorting:** Sorts all accumulated steps primarily by phase rank and secondarily by original template order.
5. **Sequential Numbering:** Re-indexes all steps with a continuous 1-based sequential number (`sequence_number`).
6. **Plan Synthesis:** Packages the organized steps into an `AssessmentPlan` object providing phase groupings and clean text summaries.

### 3. Core Business Rules & System Constraints
* **Phase Hierarchy Enforcement:** Steps are strictly sorted according to the 9-stage lifecycle hierarchy, preventing out-of-order execution during complex multi-scenario audits.
* **Lineage Preservation:** Every step permanently retains its `source_yaml_id`, guaranteeing full auditability back to the original playbook source.
* **Unknown Phase Handling:** Unrecognized phases receive fallback rank 99, placing them safely at the end of the assessment plan rather than causing sort errors.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `typing.List`, `typing.Dict`, `typing.Any`, `typing.Optional`.
* **Classes & Functions:**
  * `get_phase_rank(phase: str) -> int`: Normalizes phase names (spaces/hyphens to underscores) and computes sort rank.
  * `AssessmentPlan`: Container class with `.summary()` method outputting formatted textual plans.
  * `Planner`: Core synthesizer with `.create_plan(templates, custom_name=None)` method.

---

## 2.8 `variable_engine.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `VariableEngine` like a legal document mail-merge engine. A law firm writes a standardized commercial lease agreement containing placeholders like `[TENANT_NAME]`, `[PROPERTY_ADDRESS]`, and `[MONTHLY_RENT]`. Before signing, the engine replaces every placeholder across all 50 pages with the exact client details, ensuring zero typos and flagging any blanks that were forgotten.
* **Problem Solved:** Security playbooks contain generic command templates (e.g., `nmap -sV [TARGET_IP] -p [PORT]`). Manually copying, pasting, and retyping target IPs, tenant names, and usernames across 30 assessment commands leads to catastrophic typos and target drift. `VariableEngine` centrally manages, validates, and injects runtime target variables into commands and nested data structures.

### 2. The Functional Journey (Plain-English Timeline)
1. **Variable Dictionary Initialization:** Accepts default or initial environment parameters (e.g., target IP, domain, username).
2. **Key Normalization:** Strips enclosing brackets/braces, trims whitespace, and converts all variable keys to uppercase (e.g., `[tenant_name]` $\rightarrow$ `TENANT_NAME`).
3. **Placeholder Detection:** Scans text using regular expressions to discover all bracketed `[VAR_NAME]` and braced `{VAR_NAME}` placeholder tokens.
4. **Missing Variable Audit:** Compares extracted placeholders against active session variables, compiling a list of missing variables that require operator input prior to execution.
5. **Multi-Format String Substitution:** Performs case-tolerant string replacement for both uppercase and lowercase placeholder variants (e.g., `[TARGET_IP]` and `[target_ip]`).
6. **Recursive Structure Traversal:** Recursively traverses nested data structures (lists, dictionaries, strings), substituting variables across complex playbook objects.

### 3. Core Business Rules & System Constraints
* **Dual Placeholder Syntax:** Equivocally supports both square bracket (`[VAR]`) and curly brace (`{VAR}`) token syntax.
* **Case Tolerance:** Variable lookup and substitution are case-tolerant, correctly matching `[target_ip]`, `[TARGET_IP]`, `{Target_Ip}`, etc.
* **Immutability of Template Data:** Substitution creates clean data copies without mutating original template dictionaries in place.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `re`, `typing`.
* **Interface Methods:**
  * `set_variable(name, value)` / `set_variables(dict)`
  * `get_variable(name, default=None)`
  * `extract_variables(text) -> List[str]`
  * `get_missing_variables(text) -> List[str]`
  * `substitute(content, custom_vars=None) -> Any` (supports `str`, `list`, `dict`).

---

## 2.9 `command_executor.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `CommandExecutor` like a nuclear launch control safety console. The system does not allow any missile to launch just because an automated target coordinate was calculated. Instead, the console displays the exact missile trajectory, the intended target, the expected blast radius, and the collateral damage risk on a clear screen, demanding that the human commander physically turn a physical authorization key before any rocket motor ignites.
* **Problem Solved:** Allowing an AI system or automated script to execute shell commands directly on production systems without human inspection is extremely dangerous. A single unvalidated command could crash a production server, corrupt a database, or violate compliance scope. `CommandExecutor` acts as a safety-gated execution broker that operates in read-only mode by default, formats human-readable previews, and requires explicit operator authorization for every execution.

### 2. The Functional Journey (Plain-English Timeline)
1. **Safety Mode Initialization:** Initializes the executor with `read_only = True` by default.
2. **Command Step Extraction:** Parses a plan step to extract shell commands, operational purpose, expected output, and risk/detection indicators.
3. **Variable Interpolation:** Injects runtime variables into the command string via `VariableEngine` if provided.
4. **Preview Generation:** Generates structured `CommandPreview` objects displaying the step ID, finalized command string, purpose, expected result, and associated operational risks.
5. **Execution Gate Evaluation:** When execution is requested, the executor checks two strict security gates:
   * Gate 1: Is the executor configured with `read_only = False`?
   * Gate 2: Did the human operator provide explicit authorization (`approved = True`)?
6. **Isolated Subprocess Execution:** If both gates pass, executes the command via `subprocess.run` inside a protected shell with output capture and a strict 60-second execution timeout.
7. **Structured Result Reporting:** Captures and returns standard output (`stdout`), standard error (`stderr`), status codes (`returncode`), and execution state (`success`, `failed`, `skipped`, `declined`, or `error`).

### 3. Core Business Rules & System Constraints
* **Default Read-Only Safety:** By default, `read_only = True`. Any attempt to execute returns `{"status": "skipped", "reason": "Executor is in Read-Only Mode."}`.
* **Mandatory Authorization Gate:** If `approved = False`, execution is immediately halted with `{"status": "declined", "reason": "Execution was not approved by user."}`.
* **Execution Timeout Boundary:** Shell commands are bounded by a strict 60-second timeout (`timeout=60`), preventing hung processes or infinite loops from blocking the pipeline.
* **Full Stream Capture:** Both standard output and standard error streams are captured (`capture_output=True`) to feed the evidence vault and audit logs.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `subprocess`, `shlex`, `typing`.
* **Classes:**
  * `CommandPreview`: Data class with `.display()` and `.to_dict()` methods.
  * `CommandExecutor`: Safety execution broker with `.prepare_step_commands(step, variable_engine)` and `.execute_command(command_str, approved=False)`.

---

## 2.10 `evidence_manager.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `EvidenceManager` like a forensic evidence locker in a police department. Whenever an officer collects fingerprint dusting, blood samples, or surveillance footage from a crime scene, the locker clerk places each item in a tamper-evident bag, stamps it with the case number, officer badge ID, precise timestamp, and chain-of-custody seal, and files it in a fireproof vault so that it can be presented to a judge during a court trial.
* **Problem Solved:** Security audits and penetration tests have zero legal or compliance value if the testing team cannot prove what commands were run, when they were executed, and what raw output was returned. Storing outputs in ephemeral chat logs causes evidence loss. `EvidenceManager` provides structured, persistent evidence logging organized by assessment, playbook, and step IDs for compliance audits and report generation.

### 2. The Functional Journey (Plain-English Timeline)
1. **Vault Initialization:** Instantiates the manager, linking it to an optional JSON persistence file path on disk.
2. **Existing Vault Ingestion:** If a storage file exists on disk, it automatically loads and deserializes previously recorded evidence records.
3. **Evidence Record Creation:** When an assessment step completes, creates an `EvidenceRecord` capturing:
   * `assessment_id` (unique audit engagement identifier).
   * `yaml_id` (source scenario playbook ID).
   * `step_id` (sequence or step identifier).
   * `command` (exact shell command executed).
   * `output` (raw console stdout/stderr output).
   * `status` (`success`, `failed`, or `error`).
   * `timestamp` (POSIX epoch timestamp).
   * `artifacts` (list of generated filenames, screenshots, or logs).
4. **In-Memory Ledger Update:** Appends the record to the active evidence registry.
5. **Disk Persistence Synchronization:** Immediately flushes the updated evidence registry to disk in structured JSON format.
6. **Report Summary Export:** Generates executive-ready summary dictionaries (`export_summary`) containing total evidence counts and truncated preview snippets for report generation.

### 3. Core Business Rules & System Constraints
* **Permanent Audit Lineage:** Every evidence entry is immutably stamped with `assessment_id`, `yaml_id`, and `step_id`.
* **Automatic Epoch Stamping:** If no timestamp is supplied, the system automatically applies current high-precision epoch time (`time.time()`).
* **Safe Output Truncation in Summaries:** Summary export limits output previews to 200 characters (`output[:200] + "..."`) to prevent report generator memory bloat while preserving full output in raw records.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `time`, `json`, `pathlib.Path`, `typing`.
* **Storage Schema:**
  ```json
  [
    {
      "assessment_id": "ASSESS-2026-001",
      "yaml_id": "AZ-02",
      "step_id": "step_1",
      "command": "az role assignment list ...",
      "output": "{...}",
      "status": "success",
      "timestamp": 1787654321.12,
      "artifacts": ["roles.json"]
    }
  ]
  ```

---

## 2.11 `memory.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `ChatMemory` like a court stenographer's official transcript ledger. Every statement made by the defense attorney (user) and every statement made by the presiding judge (assistant) is transcribed in chronological order, indexed with the session case number and precise ISO timestamp, and bound in an official archive ledger so that anyone can review exactly what was said earlier in the trial.
* **Problem Solved:** LLMs are stateless; without a conversational history mechanism, the AI forgets what target was discussed in the previous question, forcing the operator to re-explain context on every turn. In-memory Python lists vanish if the process restarts. `ChatMemory` persists conversational dialogue into an embedded SQLite database (`sessions.db`), enabling multi-turn context retention across sessions.

### 2. The Functional Journey (Plain-English Timeline)
1. **Database Connection:** Connects to the local SQLite database (`sessions.db`) using the specified `session_id`.
2. **Schema Verification & Creation:** Automatically creates the `messages` table if it does not yet exist (`id`, `session_id`, `role`, `content`, `timestamp`).
3. **Turn Recording:** When a dialogue turn occurs, writes separate records for the `user` inquiry and the `assistant` response with ISO-8601 timestamps.
4. **Recent Context Retrieval:** Queries the database for the last $N$ messages for the active session (`ORDER BY id DESC LIMIT N`) and reverses the result list into natural chronological order.
5. **Full History Export:** Provides full historical transcript dumps for audit reviews or session replay.
6. **Session Reset:** Allows operators to cleanly purge history for a specific session without impacting other sessions.

### 3. Core Business Rules & System Constraints
* **Session Isolation:** All queries filter strictly by `session_id = ?`, ensuring complete data isolation between different users or assessment sessions.
* **Chronological Inversion:** Recent message queries fetch records descending by ID to capture the newest messages, then reverse them in memory to feed the LLM prompt in correct chronological reading order.
* **ACID Persistence:** Executes `conn.commit()` after every write, ensuring conversational data is immediately saved against unexpected process termination.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `sqlite3`, `datetime.datetime`.
* **Database Schema:**
  ```sql
  CREATE TABLE IF NOT EXISTS messages (
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      session_id TEXT,
      role TEXT,
      content TEXT,
      timestamp TEXT
  );
  ```

---

## 2.12 `build_index.py` & `query_index.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `build_index.py` and `query_index.py` like the cartographer and the navigator of an exploration ship. The cartographer (`build_index.py`) surveys hundreds of uncharted islands (text documents), measures their exact latitudes and longitudes (mathematical vector coordinates), and draws a master nautical map. The navigator (`query_index.py`) uses the map in real time to calculate the exact distance to any requested landmark.
* **Problem Solved:** To perform semantic search across unstructured cybersecurity documents (OWASP guides, PortSwigger manuals, penetration testing write-ups), text must be pre-indexed into dense vector spaces. `build_index.py` constructs the primary document FAISS vector index, while `query_index.py` provides an interactive terminal verification tool to test search accuracy and distance scoring.

### 2. The Functional Journey (Plain-English Timeline)
1. **Directory Traversal (`build_index.py`):** Scans `rag/corpus/` for all `.txt` and `.md` documentation files.
2. **Embedding Computation:** Passes document texts into `SentenceTransformer(EMBED_MODEL)` to compute 384-dimensional dense semantic vectors.
3. **L2 Index Construction:** Initializes a `faiss.IndexFlatL2` index and adds the vector embeddings.
4. **Serialization:** Writes the vector index binary to `faiss.index` and pickled metadata (filename, text, chunk_id) to `meta.pkl`.
5. **Interactive Querying (`query_index.py`):** Loads the saved index and metadata, captures user query strings from the terminal, encodes the query, executes vector nearest-neighbor search, and prints the top 3 matching text snippets with $L_2$ distance metrics.

### 3. Core Business Rules & System Constraints
* **File Types Ingested:** Ingests only files ending in `.txt` or `.md`.
* **Missing Index Handling:** `query_index.py` verifies file existence and warns the operator to run `build_index.py` if index files are missing.
* **Distance Metric:** Uses standard Euclidean ($L_2$) distance metric in `IndexFlatL2` (lower distance indicates higher vector similarity).

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `faiss`, `pickle`, `os`, `sys`, `sentence_transformers.SentenceTransformer`, `core.config`.
* **Index Files:** `rag/faiss_index/faiss.index` and `rag/faiss_index/meta.pkl`.

---

## 2.13 `rag_engine.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `RAGEngine` like a two-tier academic research department. When a professor asks a complex research question, junior research assistants quickly search the library stacks to pull 10 potentially relevant textbooks (`FAISS Retrieval`). Then, a senior lead researcher carefully reads all 10 books side-by-side, scores them for exact factual relevance, discards the bottom 7, and prepares a concise 3-book summary packet for the professor (`CrossEncoder Reranker`).
* **Problem Solved:** Standard vector search (bi-encoders) is fast but frequently retrieves documents containing superficial semantic overlap. Passing too many irrelevant passages to an LLM causes hallucination and high compute cost. `RAGEngine` implements a state-of-the-art two-tier retrieval and neural reranking architecture that maximizes precision and factual grounding.

### 2. The Functional Journey (Plain-English Timeline)
1. **Component Initialization:** Loads the bi-encoder embedding model (`all-MiniLM-L6-v2`), cross-encoder neural reranker (`ms-marco-MiniLM-L-6-v2`), FAISS vector index, and metadata pickle file.
2. **Query Vectorization:** Encodes the user inquiry into a normalized float32 embedding vector.
3. **First-Stage Fast Retrieval:** Searches the FAISS index to retrieve the top $K=10$ candidate document chunks based on vector proximity.
4. **Candidate Pair Synthesis:** Pairs the user's exact query string with each of the 10 candidate document texts.
5. **Second-Stage Neural Reranking:** Feeds the query-document pairs through the `CrossEncoder` model, which performs deep cross-attention to calculate a relevance score for each passage.
6. **Descending Sorting & Filtering:** Sorts candidate documents by neural score and retains only the top $K=3$ highest-scoring chunks.
7. **Prompt Context Formulation:** Formats the final 3 chunks into a structured context block with source filenames, chunk IDs, and relevance scores.

### 3. Core Business Rules & System Constraints
* **Two-Tier Filtering Limits:** Initial retrieval candidate pool `top_k = 10`; final reranked retention `final_k = 3`.
* **Out-of-Bounds Index Guard:** Ignores any FAISS returned index equal to `-1` or exceeding metadata array bounds.
* **Empty Result Handling:** If zero documents are found, returns a safe fallback string (`"No relevant context retrieved."`) rather than failing.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `faiss`, `pickle`, `numpy`, `sentence_transformers.SentenceTransformer`, `sentence_transformers.CrossEncoder`, `core.config`.
* **Interface Methods:**
  * `retrieve(query: str) -> list[tuple[dict, float]]`
  * `build_context(ranked_docs) -> str`
  * `info() -> dict`

---

## 2.14 `prompts.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `PromptManager` like the master wardrobe and teleprompter room of a major theatrical production. Depending on whether the actor is stepping on stage as an educational mentor, a technical penetration tester, or a strict compliance auditor, the manager hands the actor the exact script, rules of engagement, background evidence, and conversation history so they deliver their lines in perfect character.
* **Problem Solved:** Hardcoding LLM system prompts inside execution scripts leads to inconsistent AI personas, formatting discrepancies, and maintenance headaches. `PromptManager` centralizes persona definitions, formats conversation history, structures retrieved RAG knowledge, and synthesizes clean, injection-resistant prompts.

### 2. The Functional Journey (Plain-English Timeline)
1. **Persona Registry Initialization:** Initializes specialized system persona prompts:
   * `rag`: Standard educational cybersecurity mentor and assistant.
   * `pentest_mentor`: Methodology-driven penetration testing guide focused on critical thinking.
   * `compliance`: Objective compliance analyst mapping technical findings to regulatory controls.
   * `agent_readonly`: Safety-focused assistant recommending commands with explicit approval requirements.
2. **History Assembly:** Formats SQLite conversational history turns into a clean `CONVERSATION HISTORY` block.
3. **Knowledge Assembly:** Formats retrieved document chunks into an indexed `RETRIEVED KNOWLEDGE` block with source citations and rerank scores.
4. **Final Prompt Synthesis:** Combines system persona instructions, conversation history, retrieved knowledge, and the user question into a structured prompt layout.

### 3. Core Business Rules & System Constraints
* **Context Grounding Rule:** System instructions explicitly require the model to answer only from provided context whenever possible and acknowledge when information is missing.
* **Educational Priority:** Directs models to prefer structured methodology and educational explanation over blunt answers.
* **Safety Mandate:** Read-only personas strictly instruct the model never to claim automatic execution and always demand explicit approval.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Prompt Modes:** `rag`, `pentest_mentor`, `compliance`, `agent_readonly`.
* **Method Interface:** `build_prompt(mode, query, retrieved_docs, memory_rows) -> str`.

---

## 2.15 `model_router.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `ModelRouter` like an intelligent power transmission station. Whether electricity is flowing from a local low-voltage generator (CPU) or a high-capacity industrial power plant (GPU), and whether the machinery uses alternating current (Seq2Seq T5), direct causal base weights (Mistral/Qwen), or plug-in voltage transformers (PEFT LoRA adapters), the power station standardizes the output voltage so that household appliances plug into the exact same wall outlet seamlessly.
* **Problem Solved:** CPU environments, GPU environments, base foundation models, and parameter-efficient LoRA adapters require completely different loading routines, precision types (`bfloat16`/`float16`), quantization configs (`bitsandbytes`), and device placement strategies. Writing separate generation logic throughout the codebase creates code duplication and brittleness. `ModelRouter` abstracts all model and adapter loading and exposes a unified `.generate(prompt)` interface.

### 2. The Functional Journey (Plain-English Timeline)
1. **Model & Adapter Identification:** Inspects the requested model name or repository string. Checks whether the target is a PEFT LoRA repository (e.g. `DarkLight6996/CyberSec-Model-V2` or `DarkLight6996/CyberSec-Qwen-V1`) via `PeftConfig.from_pretrained()`.
2. **Base Architecture Resolution:** If a LoRA adapter is detected, extracts the underlying `base_model_name_or_path` (e.g., `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `Qwen/Qwen2.5-7B-Instruct`) and loads the base model and tokenizer.
3. **Tokenizer Loading & Chat Template Alignment:** Loads the tokenizer and aligns padding tokens (`pad_token = eos_token`). In causal chat architectures (ChatML), verifies special token delimiters (`<|im_start|>`, `<|im_end|>`).
4. **Hardware & Quantization-Aware Loading:**
   * *CPU Mode:* Loads model weights onto system RAM in float32 precision and sets evaluation mode.
   * *GPU Mode:* Validates CUDA availability. If 4-bit quantization is configured, applies `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` and attaches weights to GPU memory via `device_map="auto"`.
5. **Adapter Attachment:** If a LoRA adapter was specified, attaches adapter layers to the active model using `PeftModel.from_pretrained(model, adapter_path)`.
6. **Input Tokenization & Truncation:** Converts incoming prompt text into PyTorch tensors, enforcing `max_input_tokens` limits.
7. **Inference Execution:** Executes `model.generate()` within a `torch.no_grad()` context using specified temperature and top-p sampling.
8. **Output Decoding & Causal Token Slicing:** For causal models, slices output tokens past the input sequence length (`outputs[0][input_length:]`) and decodes into clean text, stripping internal generation artifacts.

### 3. Core Business Rules & System Constraints
* **CUDA Hardware Validation:** If `mode="gpu"` is requested on a system without CUDA, immediately raises `RuntimeError("GPU mode requested, but CUDA is not available.")`.
* **Memory Optimization:** GPU mode loads weights in half-precision (`torch.float16`) to minimize VRAM consumption.
* **Causal Token Slicing:** For causal models (e.g., Mistral), slices output tensors past the input length (`outputs[0][input_length:]`) so prompt text is not duplicated in the generated response.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `transformers.AutoTokenizer`, `transformers.AutoModelForSeq2SeqLM`, `transformers.AutoModelForCausalLM`, `torch`.
* **Interface Methods:** `generate(prompt, temperature=0.2, top_p=0.9, do_sample=True) -> str`, `info() -> dict`.

---

## 2.16 `rag_pipeline_cpu_final.py` & `rag_pipeline_gpu_final.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of these pipeline scripts like two complete, ready-to-drive security vehicles. One is a lightweight, fuel-efficient patrol scooter designed to navigate tight city streets on standard gasoline (`CPU Pipeline` using FLAN-T5). The other is a high-speed, heavy-duty armored pursuit vehicle equipped with a turbocharged V8 engine (`GPU Pipeline` using Mistral-7B). Both vehicles have the exact same steering wheel and dashboard controls.
* **Problem Solved:** Users need standalone, turn-key interactive terminal applications to run end-to-end RAG workflows without manual setup. These two scripts assemble all core modules (retrieval, memory, prompting, and inference) into unified, interactive CLI applications optimized for their respective compute environments.

### 2. The Functional Journey (Plain-English Timeline)
1. **Subsystem Bootstrapping:** Initializes the RAG retrieval engine, SQLite chat memory, prompt manager, and model router.
2. **Operator Command Loop:** Enters an interactive terminal loop listening for operator input.
3. **Special Command Interception:** Intercepts control commands:
   * `exit`: Closes SQLite database connection and terminates application.
   * `reset`: Clears session history from SQLite.
   * `history`: Displays complete multi-turn dialogue history.
4. **Contextual Retrieval:** Queries the FAISS index and cross-encoder reranker for relevant documentation.
5. **Memory Window Retrieval:** Pulls the last 6 conversational turns from SQLite.
6. **Prompt Assembly:** Synthesizes system persona, history, retrieved knowledge, and the current question.
7. **LLM Generation:** Generates the response via CPU or GPU model router.
8. **Memory Commit:** Persists the user question and assistant answer to the SQLite database and prints the answer to the terminal.

### 3. Core Business Rules & System Constraints
* **Session Segregation:** CPU pipeline uses session ID `cpu_session`; GPU pipeline uses session ID `gpu_session`.
* **Graceful Exit:** Typing `exit` or issuing `KeyboardInterrupt` (Ctrl+C) guarantees that SQLite connections are closed cleanly.
* **Prompt Grounding:** Uses standard `rag` mentor persona mode to ensure grounded responses.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `core.rag_engine`, `core.memory`, `core.prompts`, `core.model_router`, `core.config`.
* **Execution:** Run via `python core/rag_pipeline_cpu_final.py` or `python core/rag_pipeline_gpu_final.py`.

---

## 2.17 Verification Suite: `verify_full_pipeline.py`, `test_core_modules.py`, `benchmark.py`

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of these test scripts like the comprehensive pre-flight checklist and flight simulator used by aviation safety inspectors before certifying an aircraft for commercial passenger flight. Every hydraulic valve, emergency oxygen system, radio frequency, and engine thrust limit is stressed and measured under simulated emergency conditions to ensure zero failures in the air.
* **Problem Solved:** In complex multi-component systems, a failure in variable interpolation, scenario indexing, or retrieval scoring could cause silent operational failures during live security audits. These verification tools provide automated integration testing, subsystem unit validation, and retrieval speed/relevance benchmarking.

### 2. The Functional Journey (Plain-English Timeline)
1. **Document Retrieval Verification (`verify_full_pipeline.py`):** Executes technical queries (e.g., Kerberos, Pass-the-Hash) and validates that `RAGEngine` retrieves and scores document chunks.
2. **YAML Search Verification:** Queries the scenario vector index with audit intents (e.g., *"Assess Azure AD"*, *"SOC2 review"*) and verifies template ranking and scoring.
3. **End-to-End Pipeline Integration Test:** Simulates a complete 9-step audit journey:
   * Step 1: User question input.
   * Step 2: Multi-turn SQLite memory commit and retrieval.
   * Step 3 & 4: Assessment routing across documents and YAML playbooks.
   * Step 5: Sequential assessment plan synthesis via `Planner`.
   * Step 6: Dynamic target parameter substitution via `VariableEngine`.
   * Step 7: Safety-gated read-only preview and execution via `CommandExecutor`.
   * Step 8: Evidence record creation and report summary export via `EvidenceManager`.
   * Step 9: Combined prompt context formatting for LLM inference.
4. **Subsystem Unit Testing (`test_core_modules.py`):** Runs isolated functional checks across all 6 core assessment modules.
5. **Retrieval Speed & Precision Benchmarking (`benchmark.py`):** Measures query latency and cross-encoder relevance scores across standard security queries.

### 3. Core Business Rules & System Constraints
* **End-to-End Determinism:** Verifies that all 9 pipeline steps pass with zero unhandled exceptions.
* **Read-Only Safety Verification:** Confirms that `CommandExecutor` strictly declines unapproved execution during automated testing.
* **Evidence Integrity:** Confirms that recorded evidence matches the generated plan step and parameter values.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Execution:**
  * Full integration verification: `python core/verify_full_pipeline.py`
  * Core modules unit test: `python core/test_core_modules.py`
  * Retrieval speed benchmark: `python core/benchmark.py`

---

## 2.18 `session_manager.py` (Assessment Session & State Persistence)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `SessionManager` like a dedicated physical case binder and file cabinet maintained by a chief court clerk. Whenever an investigation or trial begins, the clerk creates a labeled file folder (`.assessments/<session_name>/`), records the official case jurisdiction and boundaries (`scope`), logs every piece of physical evidence in an indexed ledger (`state.json`), and stores forensic disk images in an attached evidence vault (`artifacts/`).
* **Problem Solved:** Security assessments take hours or days and require persistent state tracking across restarts. Storing state in temporary RAM variables leads to catastrophic data loss if the terminal closes or crashes. `SessionManager` provides persistent, isolated session management with JSON serialization, target extraction, notes logging, and planner step tracking.

### 2. The Functional Journey (Plain-English Timeline)
1. **Directory Discovery:** Verifies or initializes the master `.assessments/` root directory.
2. **Session Initialization:** `create_session(name, scope)` sanitizes the session slug, creates `.assessments/<name>/` and `.assessments/<name>/artifacts/`, extracts target IPs/domains from scope text, and writes an initial `state.json`.
3. **Session Resumption:** `load_session(name)` loads previously recorded session metadata, targets, evidence, and completed steps into memory.
4. **Target Entity Extraction:** Automatically extracts and deduplicates IPv4/CIDR addresses and domain names from input scope and ingested findings into `state["targets"]`.
5. **Atomic Persistence:** `save_current_state()` writes state data to a temporary file (`state.json.tmp`) and performs an atomic filesystem replace to prevent state corruption during sudden crashes.

### 3. Core Business Rules & System Constraints
* **Session Folder Isolation:** All session artifacts, notes, evidence, and reports are isolated inside `.assessments/<session_name>/`.
* **Atomic State Commits:** Uses temporary file writing and atomic replace to guarantee data integrity against process interruption.
* **Target Normalization:** Auto-deduplicates targets across `ips`, `ports`, `domains`, `services`, and `cloud_resources`.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `pathlib.Path`, `json`, `datetime.datetime`, standard library `re`.
* **Interface Methods:** `create_session(name, scope)`, `load_session(name)`, `save_current_state()`, `list_sessions()`, `add_note()`, `add_evidence()`, `add_ingested_findings()`, `set_current_step()`, `complete_current_step()`.

---

## 2.19 `ingestion.py` (Extensible File Ingestion Engine)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `FileIngestionEngine` like a multi-lingual forensic document laboratory. Whether a field operative brings in a technical medical chart (`Nmap XML`), an intelligence agency wiretap transcript (`Burp Suite export`), a satellite communications dispatch (`Cloud CLI JSON`), or handwritten field notes (`Generic text`), the laboratory routes each item to a specialist, extracts key names, locations, and danger signals, and files the standardized facts into the active case docket.
* **Problem Solved:** Security operators use disparate scanning and audit tools that output proprietary, complex XML or JSON structures. Manually copying and pasting open ports, endpoints, and misconfigurations into planning tools is tedious and error-prone. `FileIngestionEngine` automatically detects file formats, extracts open ports, live hosts, and vulnerabilities, and normalizes them so the dynamic planner adapts immediately.

### 2. The Functional Journey (Plain-English Timeline)
1. **File Read & Inspection:** Inspects file extension and first 4KB of text to select the appropriate parser (`NmapParser`, `BurpParser`, `CloudCLIParser`, `GenericTextParser`).
2. **Nmap Extraction:** Slices XML hosts, extracts IPv4/IPv6 addresses, resolves hostnames, and collects all open ports, protocols, services, and script outputs.
3. **Burp Suite Extraction:** Parses XML `<issue>` items, normalizes host URLs to clean domains, and categorizes vulnerabilities by severity.
4. **Cloud CLI Extraction:** Analyzes JSON dumps from Azure or AWS CLI, detecting high-privilege roles (`Owner`, `Contributor`, `*:*`), storage accounts, and Key Vaults.
5. **Entity Normalization:** Formats findings into standardized entity buckets (`ips`, `ports`, `domains`, `services`, `cloud_resources`, `vulnerabilities`).
6. **Provenance Archiving:** Copies the original raw scanner file into the session's `.assessments/<session_name>/artifacts/` folder for immutable chain-of-custody proof.

### 3. Core Business Rules & System Constraints
* **Parser Fallback:** If an XML or JSON file has malformed tags, the engine automatically falls back to regex-based text parsing without raising unhandled exceptions.
* **Domain Normalization:** Strips URL protocols (`https://`), trailing paths, and port numbers from Burp hosts to produce clean domain names.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `xml.etree.ElementTree`, `re`, `json`, `shutil`, `pathlib.Path`.
* **Classes:** `FileIngestionEngine`, `NmapParser`, `BurpParser`, `CloudCLIParser`, `GenericTextParser`, `IngestionResult`.

---

## 2.20 `report_generator.py` (Markdown Assessment Report Synthesizer)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `ReportGenerator` like an expert legal stenographer and senior paralegal drafting the final certified trial brief. They pull together the indictment charges (`scope`), the timeline of police and detective actions (`chronological timeline`), the forensic lab ballistics tests (`ingested findings`), and the judge's final verdict into an authoritative, beautifully formatted leather-bound volume ready to be presented to the board of directors.
* **Problem Solved:** Manually compiling assessment reports from terminal logs, disparate text files, and scanner outputs takes hours of repetitive formatting. `ReportGenerator` transforms session state into an executive-ready Markdown report with tables, finding severity icons, and structured evidence blocks.

### 2. The Functional Journey (Plain-English Timeline)
1. **State Ingestion:** Reads the active session dictionary containing scope, target entities, scanner findings, completed steps, notes, and evidence.
2. **Executive Summary Compilation:** Formulates high-level engagement metrics (status, timeline, completed actions, finding counts).
3. **Attack Surface Table Generation:** Formats clean Markdown tables for open ports, live hosts, and services discovered by Nmap.
4. **Vulnerability Matrix:** Groups Burp Suite and cloud misconfiguration findings with visual severity indicators (`🔴 HIGH`, `🟠 MEDIUM`, `🟡 LOW`).
5. **Action & Evidence Alignment:** Outlines every planner step executed along with exact commands run and captured console outputs.
6. **Automated Export:** Exports the Markdown file directly to `.assessments/<session_name>/report.md`.

### 3. Core Business Rules & System Constraints
* **Output Truncation:** Caps individual tool console outputs in the report at 1,500 characters to ensure clean document readability while preserving raw logs on disk.
* **Self-Contained File:** Generates pure GitHub-flavored Markdown requiring zero external CSS or JavaScript dependencies.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `datetime`, `pathlib.Path`.
* **Interface Method:** `generate(session_state: Dict[str, Any], output_path: Optional[str] = None) -> str`.

---

## 2.21 Interactive Assessment CLI (`main.py`)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `main.py` like the flight director's console at NASA Mission Control. The flight director sits before a single unified command terminal. From this console, they can initiate a new mission (`new`), pull telemetry from radar stations (`upload`), ask the tactical computer for the single next flight burn maneuver (`next`), authorize engine ignition (`run`), record flight logs (`paste`), and compile the final mission briefing (`report`).
* **Problem Solved:** Security tools often require memorizing dozens of disconnected shell commands, switching between separate file viewers, and manually tracking assessment progress. `main.py` provides a unified, structured interactive CLI environment with a step-by-step methodology planner and file ingestion support.

### 2. The Functional Journey (Plain-English Timeline)
1. **CLI Launch & Banner Display:** Initializes `AssessmentCLI` and prints the interactive banner.
2. **Session Creation & Scope Setup (`new <name>`):** Prompts for engagement scope, initializes isolated session storage, and extracts initial targets.
3. **Scanner Ingestion (`upload <path>`):** Ingests Nmap, Burp, Cloud, or generic files, updates global targets, and archives raw artifacts.
4. **Dynamic Step Query (`next`):** Queries `Planner.get_next_step()`, evaluating scope and ingested findings to present exactly ONE actionable step with commands and rationale.
5. **Execution & Evidence Gate (`run`):** Prompts for operator confirmation, executes commands safely via `CommandExecutor`, captures stdout/stderr, and logs evidence.
6. **Manual Observations (`paste`):** Captures multi-line notes, manual testing findings, and observations.
7. **Report Generation (`report`):** Generates and saves the certified Markdown report to `.assessments/<session_name>/report.md`.
8. **Clean Termination (`quit`):** Atomically commits state to disk and shuts down.

### 3. Core Business Rules & System Constraints
* **Active Session Enforcement:** Commands requiring state (`next`, `run`, `paste`, `upload`, `report`, `status`) strictly enforce an active session guard (`_require_session()`).
* **Operator Authorization:** Command execution requires explicit operator sign-off (`Y/N`) before spawning subprocesses.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `cmd`, `sys`, `pathlib.Path`, `core.session_manager`, `core.ingestion`, `core.planner`, `core.report_generator`, `core.command_executor`.
* **Execution:** Run from project root via `python main.py` or from `ll-finetuning/` via `python main.py`.

---

# MODULE 3: Knowledge Corpus Vault (`ll-finetuning/rag/corpus`)

---

## 3.1 Penetration Testing Methodology Series (Sections 1–19)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this 19-volume curriculum like a comprehensive flight academy ground school syllabus for commercial airline pilots. It begins with the fundamental physics of flight and cockpit instrumentation, progresses through standard operating flight checklists, navigates emergency crosswind maneuvers, and culminates in strict international aviation laws, air traffic communication protocols, and safety reporting duties.
* **Problem Solved:** When an AI model answers security questions, generic knowledge leads to vague, unsafe, or non-compliant advice. This curated 19-chapter corpus provides the RAG pipeline with grounded, authoritative doctrine covering the full penetration testing lifecycle from legal scoping to final executive reporting.

### 2. The Functional Journey (Plain-English Timeline)
1. **Engagement Scoping & Authorization (Sections 1–4):** Establishes the foundational distinction between passive scanning and authorized exploitation, outlines rules of engagement (Black-Box, White-Box, Gray-Box), and defines strict pre-engagement scoping agreements.
2. **Reconnaissance & Threat Modeling (Sections 5–7):** Details passive OSINT information gathering techniques, active port and service enumeration, and threat surface modeling prior to launching exploits.
3. **Exploitation Domains (Sections 8–12):** Provides specialized procedural guidance across Web Applications (OWASP Top 10), Network Infrastructure (Kill Chain execution), Active Directory domain environments, Cloud Infrastructure (AWS/Azure/GCP), API endpoints, and Mobile platforms.
4. **Vulnerability Scoring & Triage (Section 14):** Formulates quantitative risk analysis using Common Vulnerability Scoring System (CVSS v3.1) metrics (Base, Temporal, and Environmental metrics).
5. **Post-Exploitation & Impact Demonstration (Section 16):** Guides operators in safely proving business impact (data access, privilege escalation) without disrupting operational systems.
6. **Reporting, Compliance & Ethics (Sections 15, 17, 18, 19):** Dictates executive reporting structure, maps technical findings to compliance standards (PCI-DSS, HIPAA, SOC 2), catalogues essential tool references, and defines legal and ethical boundaries.

### 3. Core Business Rules & System Constraints
* **Legal Authorization Prerequisite (Section 19):** Unambiguously establishes that testing without explicit written authorization violates computer crime legislation (e.g., CFAA); mandates signed Rules of Engagement (RoE).
* **Controlled Exploitation:** Directs operators to perform minimal necessary actions to validate vulnerabilities, prohibiting destructive payload execution (e.g., DoS attacks or unauthorized data modification).
* **Factual Grounding for RAG:** Acts as the primary semantic corpus retrieved when operators ask theoretical, methodological, or governance-related questions.

### 4. Technical Reference & Edge Cases (For Engineers)
* **File Inventory:** `Section 1 What is PT.txt` through `Section 19 Legal And Ethical Boundaries In PT.txt`, `SECTION 6 ACTIVE RECONNAISSANCE.txt`, `SECTION 7 THREAT MODELING BEFORE EXPLOITATION.txt`.
* **Corpus Format:** Plaintext documents indexed directly by `core/build_index.py` into `rag/faiss_index/faiss.index`.

---

## 3.2 Technical Enumeration & Exploitation Reference Guides

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of these tactical guides like an automotive technician's rapid field diagnostic charts. When a check-engine light illuminates, the mechanic doesn't re-read the entire engineering textbook; they flip directly to the sensor code table, check the multimeter voltage tolerances, and follow the exact step-by-step diagnostic tree to isolate the blown fuse.
* **Problem Solved:** Operators running live reconnaissance need fast, exact technical lookups for command flags, directory brute-forcing wordlists, HTTP response code implications, and exploit verification protocols. These focused reference files supply concise, high-density facts for vector retrieval.

### 2. The Functional Journey (Plain-English Timeline)
1. **Web Directory Enumeration (`Directory Enumeration.txt`, `Web Directory Enumeration Basics.txt`):** Documents URL discovery methodology using tools like Gobuster and Feroxbuster, wordlist strategies, and rate-limiting guidelines.
2. **Protocol & Response Analysis (`HTTP Status Code Interpretation.txt`):** Explains the security implications of HTTP status codes during fuzzing (e.g., interpreting `401 Unauthorized` vs. `403 Forbidden`, detecting soft `404` errors, and tracking `302` redirects).
3. **Advanced Enumeration (`Advanced Enumeration Techniques.txt`):** Details DNS zone transfers, SNMP enumeration, SMB share querying, and Active Directory LDAP searches.
4. **Exploitation Basics & Techniques (`Exploitation Basics.txt`, `Exploitation Techniques.txt`):** Catalogs vulnerability verification patterns, public exploit customization, and payload staging.
5. **Exploitation Validation (`Exploitation Validation Process.txt`):** Establishes the standard operating procedure for validating that an exploit succeeded without damaging the host.

### 3. Core Business Rules & System Constraints
* **Safe Validation Standard:** Validates vulnerabilities using benign proof-of-concept markers (e.g., `id`, `whoami`, `cat /etc/hostname`, non-destructive SQL queries) before attempting deeper access.
* **Status Code Triaging:** Enforces systematic HTTP status code evaluation during fuzzing to prevent missed endpoints caused by misinterpreting redirect responses.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Files:** `Advanced Enumeration Techniques.txt`, `Directory Enumeration.txt`, `Web Directory Enumeration Basics.txt`, `HTTP Status Code Interpretation.txt`, `Exploitation Basics.txt`, `Exploitation Techniques.txt`, `Exploitation Validation Process.txt`.

---

## 3.3 Large Enterprise Knowledge Corpora (`owasp_rag.txt`, `portswigger_rag.txt`)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of these two master corpora like the comprehensive international encyclopedias of web security: one is the complete published encyclopedia from the World Health Organization (`owasp_rag.txt`), and the other is the intensive combat flight training simulator manual from the test pilot school (`portswigger_rag.txt`).
* **Problem Solved:** High-level summaries often lack the deep edge-case details required to understand complex modern web vulnerabilities (such as blind SQL injection with time delays, XML external entity encoding bypasses, or prototype pollution). These large pre-processed corpora supply deep reference material.

### 2. The Functional Journey (Plain-English Timeline)
1. **OWASP Comprehensive Corpus (`owasp_rag.txt` ~1.06 MB):** Contains thousands of pre-delimited chunks covering the entire OWASP Testing Guide, web application vulnerability classes, defensive architectures, and code-level remediation patterns.
2. **PortSwigger Academy Corpus (`portswigger_rag.txt` ~64.8 KB):** Contains structured RAG chunks detailing specialized laboratory exploit mechanics, HTTP request smuggling variants, and cross-site scripting bypasses.
3. **Ingestion & Indexing:** `build_index.py` reads these chunk blocks, extracts embedded metadata (`# ID`, `# SOURCE`, `# PAGE`), and embeds the text into the FAISS neural vector database.

### 3. Core Business Rules & System Constraints
* **Chunk Formatting Schema:** Enforces standard delimiter boundaries (`### BEGIN CHUNK ###` ... `### END CHUNK ###`).
* **High Token Capacity:** Supplies dense technical context blocks spanning thousands of pages of curated application security literature.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Storage Footprint:** Total combined raw text exceeds 1.12 MB (~30,000 lines of structured Markdown and plain text).
* **Chunk Delimiters:** Fully compatible with both `build_index.py` and `rag_to_alpaca.py` parsers.

---

# MODULE 4: Neural Vector Index Stores (`ll-finetuning/rag/faiss_index` & `ll-finetuning/rag/yaml_index`)

---

## 4.1 Document Knowledge Vector Store (`rag/faiss_index/`)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this index store like a 384-dimensional satellite navigation coordinate system for knowledge. Instead of sorting documents alphabetically by title, every paragraph of security knowledge is placed at a precise spatial coordinate. When an operator asks a question, the system converts the question into a location pin and instantly finds all textbook paragraphs located within walking distance of that pin.
* **Problem Solved:** Scanning gigabytes of text sequentially during a live chat conversation takes several seconds or minutes. FAISS (Facebook AI Similarity Search) provides vector similarity search that queries tens of thousands of document embeddings in under 5 milliseconds.

### 2. The Functional Journey (Plain-English Timeline)
1. **Vector Index Ingestion (`faiss.index`):** Loads a compiled `IndexFlatL2` binary representing all documentation paragraphs from `rag/corpus/`.
2. **Metadata Array Deserialization (`meta.pkl`):** Loads a pickled array of metadata objects mapped 1-to-1 with the FAISS internal integer vector IDs.
3. **Nearest-Neighbor Search:** When a 384-dimensional query vector is submitted, FAISS calculates the Euclidean ($L_2$) distance between the query point and all stored document vectors.
4. **Metadata Hydration:** Returns the top-$K$ lowest-distance vector indices and immediately retrieves the corresponding raw document text, filename, and chunk ID from `meta.pkl`.

### 3. Core Business Rules & System Constraints
* **Dimensional Alignment:** Vector space must exactly equal 384 dimensions matching `sentence-transformers/all-MiniLM-L6-v2`.
* **1-to-1 Index-Metadata Parity:** The length of `metadata` in `meta.pkl` must exactly match the number of vectors stored in `faiss.index`.
* **Distance Metric:** Euclidean distance ($L_2$): a distance of $0.0$ represents identical meaning, while larger numbers represent increasing semantic divergence.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Files:** `rag/faiss_index/faiss.index` (43,053 bytes), `rag/faiss_index/meta.pkl` (1,196,277 bytes).
* **Index Class:** `faiss.IndexFlatL2`.

---

## 4.2 Procedural Playbook Vector Store (`rag/yaml_index/`)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this index store like a mathematical radar screen tracking 210 distinct tactical military operations. When a field commander issues an order in natural language (e.g., "Find all ways to silently compromise an Azure Key Vault"), the radar sweeps across all 210 tactical flight paths and locks onto the 3 closest matching flight vectors.
* **Problem Solved:** Operators need to find relevant operational scenario playbooks even when they don't know the exact file name or YAML ID (e.g., finding `AZ-05-key-vault-secret-dump.yaml` by asking "how to extract credentials from cloud key store"). `yaml.index` provides sub-millisecond semantic search across all scenario playbooks.

### 2. The Functional Journey (Plain-English Timeline)
1. **Vector Index Ingestion (`yaml.index`):** Loads a compiled `IndexFlatIP` (Inner Product) binary representing all 210+ YAML scenario playbooks.
2. **Metadata Deserialization (`yaml_meta.pkl`):** Loads the pickled list of full scenario dictionaries.
3. **Normalized Cosine Search:** Compares the normalized query embedding vector against the normalized playbook embeddings using vector dot products.
4. **Playbook Hydration:** Returns the top-$K$ highest-scoring template IDs and full playbook schemas for automated assessment planning.

### 3. Core Business Rules & System Constraints
* **Similarity Metric:** Cosine similarity via `IndexFlatIP` on unit-normalized vectors ($L_2$ norm = 1.0).
* **Full Playbook Hydration:** `yaml_meta.pkl` stores the entire structured YAML schema (steps, commands, variables, compliance controls), eliminating secondary disk I/O during planning.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Files:** `rag/yaml_index/yaml.index` (185,901 bytes), `rag/yaml_index/yaml_meta.pkl` (860,229 bytes).
* **Index Class:** `faiss.IndexFlatIP`.

---

# MODULE 5: Scenario & Compliance Knowledge Vault (`ll-finetuning/rag/scenarios`)

---

## 5.1 Scenario Architecture & Standard Schema

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of a Scenario YAML file like a standardized pre-flight checklist and flight plan used by commercial aviation pilots. Every single flight plan in the world follows the exact same format: flight ID, destination weather constraints, required fuel variables, sequential cockpit actions (taxi, takeoff, cruise, land), emergency warning thresholds, and post-flight maintenance sign-offs.
* **Problem Solved:** Security testing without standard schemas leads to inconsistent testing, forgotten steps, dangerous unverified commands, and audit failures. The 210+ YAML files in `rag/scenarios` enforce an auditable, machine-readable standard across every security assessment.

### 2. The Functional Journey (Plain-English Timeline)
1. **Metadata Declaration:** Specifies unique scenario ID (e.g., `AZ-02`), descriptive title, high-level category (`cloud_security`), subcategory (`privilege_escalation`), operational difficulty (`intermediate`), and time estimates.
2. **Threat Framework Mapping:** Maps the assessment directly to MITRE ATT&CK tactics, techniques, and sub-technique IDs (e.g., `T1078.004`).
3. **Regulatory Control Alignment:** Maps procedural actions directly to regulatory standards (e.g., SOC 2 `CC6.1`, PCI-DSS `Req 7.2`, HIPAA `164.312(a)(1)`).
4. **Dynamic Variable Binding:** Declares required runtime target parameters (e.g., `[TENANT_ID]`, `[TARGET_IP]`, `[KEY_VAULT_NAME]`).
5. **Multi-Phase Step Execution:** Outlines ordered tactical actions, shell commands, expected output patterns, operational risk indicators, and defensive telemetry detection patterns.
6. **Remediation & Educational Guidance:** Provides remediation checklists and comprehension questions for post-assessment debriefing.

### 3. Core Business Rules & System Constraints
* **Mandatory Field Schema:** Every playbook must contain `id`, `name`, `category`, `description`, and `steps`.
* **Step Structure Contract:** Every step item must be a dictionary explicitly defining `phase` (e.g., `reconnaissance`, `enumeration`, `privilege_escalation`).
* **Variable Token Standard:** Placeholders must adhere to `[VARIABLE_NAME]` or `{VARIABLE_NAME}` syntax for `VariableEngine` interpolation.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Directory Path:** `ll-finetuning/rag/scenarios/` (210 YAML files).
* **Schema Example:**
  ```yaml
  id: AZ-05
  name: Azure Key Vault Secret Dump
  category: cloud_security
  subcategory: credential_access
  difficulty: intermediate
  platforms: [azure, cloud]
  mitre_attack: [T1552.004, T1078.004]
  compliance_controls:
    - standard: PCI-DSS-v4
      control: "Req 3.5.1"
      description: "Protect cryptographic keys against disclosure"
  variables:
    KEY_VAULT: "Target Key Vault name"
  steps:
    - phase: enumeration
      action: "List Accessible Key Vaults"
      commands: ["az keyvault list --query '[].name' -o table"]
      expected_output: "List of key vault resource names"
  ```

---

## 5.2 Scenario Taxonomy & Domain Breakdown

The scenario vault contains 210 specialized operational procedures organized into 6 core domains:

```
+----------------------------------------------------------------------------------------------------+
|                                    SCENARIO VAULT (210+ PLAYBOOKS)                                 |
+----------------------------------------------------------------------------------------------------+
| Active Directory & Identity     : ADCS (ESC1-ESC13), Kerberoasting, AS-REP, DCSync, PetitPotam...  |
| Cloud Infrastructure & IAM      : Azure (AZ-01 - AZ-08), AWS IAM/S3/Forensics, GCP IAP/Storage...  |
| Web Application Security        : SQLi (18 variants), XSS (20 variants), SSRF, SSTI, GraphQL...    |
| Governance & Compliance Audits  : HIPAA, PCI-DSS v4, SOC 2, ISO 27001, CIS AWS/Win/Linux...        |
| Network & Perimeter Security    : VLAN Hopping, ARP Spoofing, DNS Poisoning, BGP Hijacking...     |
| Forensics & Incident Response   : Windows Memory Forensics, Linux Forensics, SOC Malware Playbook..|
+----------------------------------------------------------------------------------------------------+
```

### 1. Active Directory & Identity Security
* **Coverage:** AD Certificate Services exploitation (`adcs-esc1-attack.yaml` through `adcs-esc13-attack.yaml`), Kerberoasting (`kerberoasting-attack.yaml`), AS-REP Roasting (`asrep-roasting-attack.yaml`), Pass-the-Hash (`pass-the-hash-attack.yaml`), Pass-the-Certificate (`pass-the-certificate.yaml`), DCSync credential dumping (`dcsync-attack.yaml`), Golden Ticket forge (`golden-ticket-attack.yaml`), Zerologon (`Zerologon (CVE-2020-1472).yaml`), PetitPotam (`PetitPotam.yaml`), Shadow Credentials (`Shadow Credentials Attack (msDS-KeyCredentialLink).yaml`), and BloodHound enumeration (`ad-bloodhound-enumeration.yaml`).
* **Operational Impact:** Validates domain controller security, Kerberos ticket signing integrity, and certificate template access controls.

### 2. Cloud Infrastructure & Identity (AWS, Azure, GCP)
* **Coverage:**
  * **Azure Series:** Azure AD Recon (`AZ-01`), Privileged Role PIM Bypass (`AZ-02`), Service Principal Misconfigurations (`AZ-03`), Storage Account Exfiltration (`AZ-04`), Key Vault Secret Dump (`AZ-05`), Subscription Takeover (`AZ-06`), Conditional Access Policy Bypass (`AZ-07`), VM Lateral Movement (`AZ-08`), and Public Storage Misconfiguration (`cloud-misconfig-azure-storage-public.yaml`).
  * **AWS Series:** AWS IAM Privilege Escalation (`aws-iam-privesc.yaml`), S3 Bucket Enumeration (`aws-s3-enumeration.yaml`), AWS Cloud Forensics (`aws-cloud-forensics.yaml`), Lambda Privilege Escalation (`aws-lambda-privesc.yaml`), EC2 Metadata Exploitation (`AWS EC2 Metadata Service Exploitation.yaml`), and CloudFormation Template Injection (`cloudformation-template-injection.yaml`).
  * **GCP Series:** GCP IAM Privilege Escalation (`gcp-iam-privesc.yaml`), Compute Metadata (`gcp-compute-metadata.yaml`), Cloud Functions (`gcp-cloud-functions-enum.yaml`), Storage Enumeration (`gcp-storage-enumeration.yaml`), and IAP Bypass (`gcp-iap-bypass.yaml`).
* **Operational Impact:** Evaluates multi-cloud control planes, IAM role boundary enforcement, and serverless permission scoping.

### 3. Web Application Security
* **Coverage:**
  * **SQL Injection (18 Playbooks):** Visible Error-Based, UNION-Based column counting/value extraction, Blind Conditional Responses, Blind Conditional Errors, Blind Time Delays, Blind Out-of-Band Exfiltration, and XML/Filter Bypasses across MySQL, Oracle, and Microsoft SQL Server (`sql-server-xp-cmdshell.yaml`).
  * **Cross-Site Scripting (20+ Playbooks):** Reflected, Stored, and DOM-based variants, AngularJS sandbox escapes, CSP bypasses, cookie theft, password capture, and template literal unicode evasions.
  * **Advanced Web Vectors:** Server-Side Request Forgery (`ssrf-testing.yaml`), Server-Side Template Injection (`ssti-testing.yaml`), HTTP Request Smuggling (`http-request-smuggling.yaml`), Web Cache Poisoning (`web-cache-poisoning.yaml`), Web Cache Deception (`web-cache-deception.yaml`), GraphQL Introspection/Batching/Injection (`graphql-injection.yaml`), WebSocket Security Testing (`WebSocket Security Testing.yaml`), CRLF Injection, CORS Misconfigurations, and Race Conditions (`race-condition.yaml`).
* **Operational Impact:** Validates application layer controls, input sanitization, output encoding, and web application firewall (WAF) effectiveness.

### 4. Governance, Risk & Compliance (GRC) Audits
* **Coverage:** HIPAA Security Assessment (`hipaa-security-assessment.yaml`), PCI-DSS v4 Assessment (`pci-dss-v4-assessment.yaml`), SOC 2 Type II Controls Review (`soc2-controls-assessment.yaml`), ISO 27001:2022 Assessment (`iso27001-2022-assessment.yaml`), NIST Cybersecurity Framework Assessment (`nist-csf-assessment.yaml`), and Center for Internet Security (CIS) Benchmarks for AWS (`cis-aws-foundations-benchmark.yaml`), Windows Server 2022 (`cis-windows-server-2022-benchmark.yaml`), and Ubuntu 22.04 (`cis-ubuntu-2204-benchmark.yaml`).
* **Human Resources & Change Management Audits:** New Hire Provisioning (`HR-01`), Termination Offboarding (`HR-02`), Role Change Review (`HR-03`), Contractor Vendor Review (`HR-04`), Privileged Recertification (`HR-05`), Unauthorized Change Detection (`CM-01`), Code Review Bypass (`CM-02`), CI/CD Secret Leakage (`CM-03`), Production Direct Access (`CM-04`), and Backup Integrity Test (`CM-05`).
* **Operational Impact:** Delivers objective compliance gap assessments and generates audit-ready control verification evidence.

### 5. Network, Wireless & Perimeter Infrastructure
* **Coverage:** External Network Scanning (`external-network-scanning.yaml`), Internal Network Enumeration (`internal-network-enumeration.yaml`), Network Segmentation Testing (`network-segmentation-testing.yaml`), VLAN Hopping (`vlan-hopping.yaml`), ARP Spoofing (`arp-spoofing.yaml`), DNS Poisoning (`dns-poisoning.yaml`), DNS Zone Transfers (`dns-zone-transfer.yaml`), DHCP Starvation (`dhcp-starvation-rogue.yaml`), BGP Route Hijacking (`bgp-route-hijacking.yaml`), WPA2 Cracking (`wpa2-cracking.yaml`), Evil Twin Attacks (`evil-twin-attack.yaml`), and C2 Framework Red Teaming (`c2-framework-redteam.yaml`).
* **Operational Impact:** Validates network perimeter boundaries, firewall segmentation rules, and internal broadcast domain isolation.

### 6. Forensics, Incident Response & Threat Hunting
* **Coverage:** Windows Memory Forensics (`windows-memory-forensics.yaml`), Linux Forensics (`linux-forensics.yaml`), Threat Hunting Techniques (`threat-hunting-techniques.yaml`), Threat Intelligence Operations (`threat-intelligence-operations.yaml`), SOC Incident Response Malware Playbook (`soc-ir-malware-playbook.yaml`), Sysmon Evasion (`sysmon-evasion.yaml`), Event Log Tampering (`event-log-tampering.yaml`), AMSI/AV Evasion (`amsi-av-evasion.yaml`), YARA Rule Creation (`yara-rule-creation.yaml`), and Sigma Rule Creation (`sigma-rule-creation.yaml`).
* **Operational Impact:** Equips blue teams with structured investigation playbooks and detection engineering rules.

---

# MODULE 6: Model Fine-Tuning & Inference Pipeline (`ll-finetuning/scripts`)

---

## 6.1 `train_lora.py` (GPU LoRA Fine-Tuning)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of full model training like sending a medical doctor back through 12 years of elementary school, high school, and university from scratch just to teach them a new cardiology surgical technique. LoRA (Low-Rank Adaptation) fine-tuning is like having that already-qualified doctor wear specialized augmented-reality glasses that project the new cardiology steps directly into their visual field during surgery, achieving mastery in hours at a fraction of the cost.
* **Problem Solved:** Retraining an entire 1.1-billion-parameter language model requires tens of thousands of dollars in enterprise GPU compute clusters. Standard fine-tuning also risks "catastrophic forgetting," where the model forgets general English reasoning. `train_lora.py` freezes 99.5% of the base model's weights and trains low-rank adapter matrices ($r=8$) on the attention layers (`q_proj`, `v_proj`), adapting TinyLlama to cybersecurity reasoning on a single consumer GPU.

### 2. The Functional Journey (Plain-English Timeline)
1. **Environment & Path Resolution:** Dynamically anchors root directory paths and confirms the existence of the training dataset (`data/portswigger_alpaca.jsonl`).
2. **Dataset Ingestion:** Loads the JSONL dataset into a Hugging Face streaming dataset table.
3. **Tokenizer Loading & EOS Alignment:** Loads the base model tokenizer (`TinyLlama-1.1B-Chat-v1.0`) and aligns missing pad tokens with the end-of-sequence token (`pad_token = eos_token`).
4. **Base Model Instantiation:** Loads the 1.1-billion parameter causal LM in half-precision (`torch.float16`) with automated GPU device mapping (`device_map="auto"`).
5. **Alpaca Prompt Formatting:** Merges raw `instruction`, `input`, and `output` fields into standardized training prompts:
   ```text
   ### Instruction:
   [Task description]

   ### Input:
   [Contextual details]

   ### Response:
   [Expected command & explanation]
   ```
6. **Tokenization & Truncation:** Encodes prompt strings into token IDs with strict truncation capped at 256 tokens.
7. **LoRA Adapter Injection:** Configures PEFT (Parameter-Efficient Fine-Tuning) with rank $r=8$, scaling factor $\alpha=16$, and dropout $0.05$ targeting query and value projection layers (`q_proj`, `v_proj`).
8. **Gradient Accumulation & Training:** Executes 2 training epochs with a per-device batch size of 1 and 8 gradient accumulation steps (effective batch size = 8), logging step-by-step training loss.
9. **Checkpoint Guard & Resume:** Checks for existing checkpoints in `lora-output/` to resume interrupted training automatically.
10. **Adapter Serialization:** Saves the trained adapter weights and tokenizer configuration directly to `lora-adapters/`.

### 3. Core Business Rules & System Constraints
* **Dataset Prerequisite:** Raises `FileNotFoundError` if the training dataset file does not exist on disk.
* **Effective Batch Sizing:** Batch size = 1 with 8 gradient accumulation steps ($1 \times 8 = 8$), maintaining low VRAM requirements while stabilizing gradient descent.
* **Token Budget Limit:** Caps training sequence length at 256 tokens (`max_length=256`), optimizing memory usage for concise command-and-response instruction pairs.
* **Precision Policy:** Automatically enables `fp16 = True` only when CUDA is active to prevent floating-point CPU incompatibilities.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `datasets`, `transformers`, `peft` (LoraConfig, get_peft_model), `torch`, `pathlib.Path`.
* **Hyperparameters:** Model: `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, Epochs: 2, Learning Rate: `2e-4`, LoRA Rank: $r=8$, LoRA Alpha: $\alpha=16$, Target Modules: `["q_proj", "v_proj"]`, Dropout: $0.05$.
* **Saved Outputs:** `lora-adapters/adapter_model.safetensors`, `lora-adapters/adapter_config.json`.

---

## 6.2 `train_lora_cpu.py` (CPU-Optimized LoRA Fine-Tuning)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of this script like running a high-precision watchmaking workshop on a portable workbench with hand tools instead of an automated industrial robotic assembly line. It accomplishes the exact same delicate mechanical adjustments, adhering to stricter safety tolerances so that standard household electricity (CPU RAM) does not overheat or trip the circuit breakers.
* **Problem Solved:** Many development, CI/CD, and low-resource audit environments lack dedicated NVIDIA CUDA GPUs. Standard GPU training scripts immediately crash on CPU systems due to half-precision (`float16`) instructions and CUDA memory calls. `train_lora_cpu.py` provides a deterministic CPU training pipeline utilizing single-precision (`torch.float32`), fixed max-length padding, and explicit CPU process allocation.

### 2. The Functional Journey (Plain-English Timeline)
1. **Configuration & Data Verification:** Anchors script paths and verifies the existence of the training dataset (`data/htb.jsonl`).
2. **Dataset Ingestion:** Reads the dataset records using the Hugging Face `datasets` loader.
3. **Single-Precision Model Loading:** Loads `TinyLlama-1.1B-Chat-v1.0` explicitly using `torch.float32` precision and binds the tensor graph directly to the system CPU (`model.to("cpu")`).
4. **Alpaca Text Synthesis:** Combines instruction, input context, and expected output into standard prompt blocks.
5. **Fixed-Length Padding Tokenization:** Tokenizes prompt text enforcing `padding="max_length"` and `max_length=256` to ensure uniform tensor shapes on CPU memory buses.
6. **LoRA Adapter Wrapping:** Hooks PEFT low-rank matrices ($r=8, \alpha=16$) onto attention projections and prints the trainable parameter count.
7. **CPU-Enforced Training Arguments:** Sets `use_cpu=True`, `fp16=False`, `dataloader_pin_memory=False`, and learning rate `2e-4`.
8. **Trainer Execution & Checkpoint Recovery:** Checks for existing checkpoints in `lora-output-cpu/` and begins training.
9. **Adapter Export:** Saves trained CPU adapter weights to `lora-adapters-cpu/`.

### 3. Core Business Rules & System Constraints
* **Precision Mandate:** `fp16` is strictly set to `False` and `torch_dtype` is strictly set to `torch.float32`; attempting `float16` on CPU triggers execution crashes.
* **Explicit Padding:** Requires `padding="max_length"` to avoid dynamic tensor reshaping overhead on CPU threads.
* **Pin Memory Prohibition:** `dataloader_pin_memory=False` avoids invalid CUDA pinned memory allocations on host RAM.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `datasets`, `transformers`, `peft`, `torch`, `pathlib.Path`.
* **Output Paths:** Checkpoints: `scripts/lora-output-cpu/`, Adapters: `scripts/lora-adapters-cpu/`.

---

## 6.3 `infer_lora.py` (Adapter Inference & Evaluation)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `infer_lora.py` like testing a pair of custom racing goggles on a sports car driver before a championship race. The mechanic clips the specialized polarized lenses onto the driver's standard helmet, takes the car onto the test track, asks a rapid-fire navigational question, and verifies that the driver spots the optimal cornering line immediately.
* **Problem Solved:** After fine-tuning LoRA adapters, operators need a lightweight verification tool to validate that the trained adapter weights load correctly on top of the base model and generate accurate, cybersecurity-specific technical responses without running full RAG retrieval.

### 2. The Functional Journey (Plain-English Timeline)
1. **Path Verification:** Confirms that the compiled adapter directory (`lora-adapters/`) exists; if missing, raises an instructive error directing the user to run training first.
2. **Base Model Loading:** Instantiates the base tokenizer and loads `TinyLlama-1.1B-Chat-v1.0` in `torch.float16` with automated device placement.
3. **PEFT Adapter Merging:** Loads and mounts the LoRA adapter weights onto the base model architecture via `PeftModel.from_pretrained()`.
4. **Evaluation Mode Engagement:** Sets the model to evaluation mode (`model.eval()`) to disable dropout and freeze weights.
5. **Prompt Tokenization:** Tokenizes the test prompt (e.g., `"port 443 is open what next"`) and moves input tensors to the active device.
6. **Autoregressive Generation:** Generates up to 150 new tokens within `torch.no_grad()` using sampling with temperature $0.7$.
7. **Response Decoding:** Decodes output token IDs into plain text, stripping special tokens and printing the response to the terminal.

### 3. Core Business Rules & System Constraints
* **Adapter Directory Guard:** Halts immediately with `FileNotFoundError` if `lora-adapters/` is absent, preventing confusing downstream CUDA memory errors.
* **Inference Freeze:** Enforces `torch.no_grad()` and `model.eval()`, preventing memory leakages from gradient computation during testing.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `transformers.AutoTokenizer`, `transformers.AutoModelForCausalLM`, `peft.PeftModel`, `torch`, `pathlib.Path`.
* **Execution:** Run via `python scripts/infer_lora.py`.

---

## 6.4 `upload_to_hf.py` (Automated Hugging Face Deployment)

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of `upload_to_hf.py` like an automated international shipping courier and distribution warehouse manager. Once a new precision flight navigation system (fine-tuned adapter) is tested and certified, the courier safely packs the equipment, attaches customs declarations and serial numbers, securely authenticates with the global central depot (Hugging Face Hub), and publishes the device for verified field technicians worldwide to download.
* **Problem Solved:** Manually uploading multi-file model adapter artifacts (safetensors, tokenizers, configs) via web browsers is slow, error-prone, and lacks automated repository creation and access-token authentication. `upload_to_hf.py` provides a parameterized CLI pipeline that verifies local adapter existence, creates remote repositories on demand, and pushes versioned model checkpoints to Hugging Face Hub.

### 2. The Functional Journey (Plain-English Timeline)
1. **Target Adapter Resolution:** Checks the local filesystem for the adapter directory (defaults to `scripts/lora-adapters` or a custom path). If the folder is missing, raises `FileNotFoundError`.
2. **API Authentication:** Initializes `HfApi(token=token)` using either an explicit command-line token or cached environment credentials.
3. **Repository Verification & Creation:** Contacts Hugging Face Hub to verify or create the target repository (e.g., `DarkLight6996/CyberSec-Model-V2`), supporting public or private repository visibility.
4. **Folder Staging & Upload:** Uploads all adapter weights (`adapter_model.safetensors`), configuration files (`adapter_config.json`), and tokenizer definitions with a standardized commit message.
5. **Confirmation & URL Output:** Prints the public/private access URL upon successful upload.

### 3. Core Business Rules & System Constraints
* **Path Existence Gate:** Enforces `adapter_path.exists()`, preventing empty repository initialization.
* **Idempotent Repository Creation:** Uses `exist_ok=True` to safely target existing repositories without failing or overwriting untracked branch assets.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Dependencies:** `argparse`, `pathlib.Path`, `huggingface_hub.HfApi`.
* **CLI Parameters:** `--repo-id` (default `DarkLight6996/CyberSec-Model-V2`), `--adapter-dir` (optional path), `--private` (boolean flag), `--token` (optional auth token).
* **Execution:** `python scripts/upload_to_hf.py --repo-id <username>/<model-name>`

---

# MODULE 7: Training Datasets & Alpaca Knowledge Corpus (`ll-finetuning/data`)

---

## 7.1 Dataset Schema & Stanford Alpaca Specification

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of the Alpaca dataset like a standardized three-part flashcard format used in medical board exams. Side 1 has the Clinical Scenario / Question (`instruction`), Side 2 has the Patient History / Context (`input`), and Side 3 has the Correct Diagnosis & Treatment Plan (`output`). Every student in the country studies the exact same card layout.
* **Problem Solved:** Raw technical documentation and exploit write-ups lack interactive conversational structure. To train an LLM to act as an assistant, data must be structured into prompt-context-response triplets. The repository standardizes all training data on the JSON Lines (`.jsonl`) Stanford Alpaca specification.

### 2. The Functional Journey (Plain-English Timeline)
1. **Record Serialization:** Every row in a `.jsonl` file represents one independent JSON object containing `instruction`, `input`, and `output`.
2. **Instruction Formulation:** Encapsulates the specific operational task or cybersecurity question (e.g., *"How do you test for SQL injection vulnerabilities in a user input field?"*).
3. **Input Contextualization:** Provides target-specific constraints, environment details, or source citations (e.g., *"You suspect the username field on the login page might be vulnerable..."*).
4. **Output Synthesis:** Supplies the exact terminal command, explanation, and expected outcome (e.g., `Command: sqlmap -u "..." --risk=3 --level=5 --dbs\nExplanation: ...`).

### 3. Core Business Rules & System Constraints
* **Strict JSON Lines Syntax:** Every row must be valid standalone JSON terminated with a newline (`\n`).
* **UTF-8 Character Encoding:** All files are encoded in UTF-8 to support technical symbols, escape codes, and international character sets.

---

## 7.2 Dataset Inventory & Domain Coverage

| Dataset File | File Size | Description & Operational Domain |
| :--- | :--- | :--- |
| **`comprehensive_cybersec_finetune.jsonl`** | **195 KB** | **Master Multi-Scenario Dataset (v3.0):** Comprehensive end-to-end technical scenarios covering Active Directory, ADCS (ESC1–ESC16), Shadow Credentials, Kerberoasting, RBCD, full OWASP testing guides (XSS, SSRF, XXE, Deserialization, Auth/Session, Business Logic, API security). |
| **`htb.jsonl`** | **27.8 MB** | Comprehensive collection of HackTheBox penetration testing walkthroughs, privilege escalation chains, Linux/Windows box exploits, and CTF tradecraft. |
| **`owasp_alpaca.jsonl`** | **252 KB** | Structured application security instruction pairs derived from the OWASP Testing Guide, covering input validation, authentication, and secure architectures. |
| **`portswigger_alpaca.jsonl`** | **70.9 KB** | High-density Web Security Academy lab instruction pairs covering SQLi, XSS, CSRF, SSRF, and HTTP request smuggling. |
| **`redefining_hacking.jsonl`** | **676 KB** | Offensive tradecraft and modern methodology instruction pairs focusing on network security, credential dumping, and lateral movement. |
| **`structured_data.jsonl`** | **92.7 KB** | Curated compliance audit checks, SOC 2 control mappings, and standardized command reference instructions. |
| **`data.jsonl` & `finetune_data.jsonl`** | **68.8 KB / 87.2 KB** | Benchmark fine-tuning instruction sets covering tool usage (`nmap`, `hydra`, `sqlmap`, `bloodhound`, `certipy`, `evil-winrm`). |
| **`train.jsonl`** | **3.8 KB** | Lightweight development test dataset for rapid smoke-testing of training pipelines. |

---

# MODULE 8: Environment Setup & Platform Provisioning (`ll-finetuning/setup` & root)

---

## 8.1 Multi-Platform Runtime Configuration

### 1. Executive Concept & Real-World Analogy
* **Real-World Analogy:** Think of the setup configuration like the specialized electrical adapter kit and transformer carried by an international film production crew. When filming in North America (Windows 11 with CUDA 12.1), they plug into 120V 60Hz outlets; when filming in Europe (Ubuntu with standard Linux system libraries), they switch to 230V 50Hz plugs. The cameras and lighting equipment (codebase) run identically without burning out.
* **Problem Solved:** Machine learning dependencies (PyTorch, torchvision, torchaudio, FAISS, CUDA drivers, C++ build tools) have notorious operating system incompatibilities. Installing the wrong binary wheels causes broken CUDA bindings, segmentation faults, or silent CPU fallbacks. The `setup/` directory isolates exact multi-platform dependency manifests.

### 2. The Functional Journey (Plain-English Timeline)
1. **Operating System Detection:** Identifies target runtime platform (Windows 11 vs. Ubuntu Linux).
2. **Hardware Acceleration Alignment:**
   * *Windows 11:* Requires NVIDIA CUDA Toolkit 12.1+, Microsoft Visual C++ Build Tools, and installs PyTorch via the official CUDA 12.1 index:
     ```powershell
     pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
     pip install -r ll-finetuning/setup/windows11/requirements.txt
     ```
   * *Ubuntu Linux:* Requires `build-essential`, `python3-dev`, and installs Linux-compiled wheels:
     ```bash
     sudo apt install build-essential python3-dev
     pip install -r ll-finetuning/setup/ubuntu/requirements.txt
     ```
3. **Core Library Installation:** Installs Hugging Face ecosystem (`transformers`, `datasets`, `peft`, `accelerate`, `bitsandbytes`), retrieval stack (`faiss-cpu`, `sentence-transformers`), document parsers (`PyMuPDF`, `tiktoken`, `beautifulsoup4`), and operational utilities (`pyyaml`, `tqdm`).

### 3. Core Business Rules & System Constraints
* **Index URL Pinning:** Windows 11 installation pins `--index-url https://download.pytorch.org/whl/cu121` to guarantee CUDA-enabled PyTorch binaries rather than default CPU wheels.
* **Build Tools Prerequisite:** Windows requires MSVC C++ build tools for compiling low-level C extensions.

### 4. Technical Reference & Edge Cases (For Engineers)
* **Configuration Files:**
  * `ll-finetuning/setup/windows11/requirements.txt`
  * `ll-finetuning/setup/ubuntu/requirements.txt`
  * `ll-finetuning/requirements.txt` (Central package manifest)
  * `.vscode/settings.json` (IDE workspace configuration)
  * `.gitignore` (Excludes checkpoints, large datasets, and virtual environment folders)

---

# MODULE 9: Operational Interface Layer & End-to-End Data Ingestion Architecture

---

## 9.1 The Operator Experience (Console & Terminal Interfaces)

While the software operates without a graphical user interface (GUI), it provides a suite of interactive command-line consoles, safety inspection consoles, and ingestion routes designed for clarity, safety, and deterministic control.

### 1. Interactive RAG Mentor Console (`rag_pipeline_cpu/gpu_final.py`)
* **What the Operator Sees & Inputs:**
  * The operator starts the pipeline and is greeted with status initialization logs (`Loading RAG Engine...`, `Loading Memory...`, `Loading Model...`, `System Ready`).
  * The terminal prompts the operator with:
    ```text
    Enter query (exit/reset/history): _
    ```
  * The operator can type plain-English questions (e.g., *"How do I test for Kerberoasting in an Active Directory domain?"*).
  * The operator can issue system session control commands:
    * `history`: Displays the full chronological transcript of previous questions and answers.
    * `reset`: Clears the conversation history in the SQLite database without restarting the script.
    * `exit`: Safely disconnects the SQLite database and shuts down the pipeline.
* **Backend Mapping:**
  ```
  Terminal Input: "How do I test for Kerberoasting?"
         │
         ├──► 1. ChatMemory: Fetches last 6 conversational turns from SQLite.
         ├──► 2. RAGEngine: Queries FAISS index for top 10 candidate document chunks.
         ├──► 3. CrossEncoder: Reranks candidate chunks, discarding the bottom 7.
         ├──► 4. PromptManager: Stitches System Persona + History + Top 3 Chunks + Query.
         ├──► 5. ModelRouter: Passes prompt to LLM (FLAN-T5 or Mistral-7B).
         └──► 6. Output: Streams formatted answer to terminal screen and commits turn to database.
  ```

---

### 2. Safety-Gated Command Console (`command_executor.py`)
* **What the Operator Sees & Inputs:**
  * When a security assessment plan step requires executing a diagnostic command against a target, the console displays a structured safety preview:
    ```text
    --------------------------------------------------
    Step ID:         AZ-02_step_1
    Command:         az role assignment list --tenant ACME-Prod-Tenant --resource-group rg-east
    Purpose:         Enumerate Role Assignments for Azure Tenant
    Expected Output: JSON list of user role assignments and privileges
    Possible Risks:  CloudTrail / Azure Activity Log event for GetRoleAssignment
    --------------------------------------------------
    Approve execution? (Y/N): _
    ```
  * The operator explicitly types `Y` (Yes) or `N` (No).
* **Backend Mapping:**
  * If `Y`: The executor launches an isolated subprocess with a 60-second execution boundary, captures console outputs, and deposits the evidence into `EvidenceManager`.
  * If `N`: The executor declines execution, returns `{"status": "declined"}`, and logs the decision in the audit trail without executing any network traffic.

---

### 3. Data Ingestion & Transformation Routes (CLI Pipelines)
* **What the Operator Triggers:**
  * Web Harvesting: `python "Generic Scripts/WebPage_chunker.py" -i urls.txt -o corpus/web_rag.txt`
  * PDF Conversion: `python "Generic Scripts/pdf_to_fintune.py" manual.pdf -o data/finetune.jsonl`
  * Directory Batching: `python "Generic Scripts/rag_to_alpaca_multiple.py" corpus_dir/ -o data/master.jsonl`
* **What the Operator Sees:**
  * Real-time `tqdm` progress bars, extracted character counts per page, chunk boundary confirmations, and total generated training pairs.

---

## 9.2 The End-to-End Data Journey

The repository governs how unstructured external cybersecurity data is transformed, indexed, reasoned upon, and presented back to the user:

```
[ STAGE 1: EXTERNAL DATA CAPTURE ]
   ├── PortSwigger Academy Articles (WebPage_chunker.py)
   ├── Security Standard PDFs & Manuals (pdf_to_rag_corpus.py)
   └── CTF / Penetration Testing Write-ups (writeup_to_corpus.py)
                                │
                                ▼
[ STAGE 2: SANITIZATION & METADATA ENRICHMENT ]
   ├── Noise Stripping (Removes HTML menus, ads, headers, scripts)
   ├── ASCII Scrubbing (Removes unprintable binary control bytes)
   └── Cryptographic & Section Stamping (Assigns MD5 hash IDs and page tags)
                                │
                                ▼
[ STAGE 3: MATHEMATICAL INDEXING & STORAGE ]
   ├── Unstructured Text  ──► SentenceTransformer (384-dim) ──► FAISS Index (faiss.index)
   ├── Structured Scenarios ──► Schema Validator & Inverted Indexes ──► YAML Index (yaml.index)
   └── Dialogue / Audit Logs ──► SQLite (sessions.db) & Evidence Vault (JSON)
                                │
                                ▼
[ STAGE 4: RUNTIME QUERY & MULTI-ENGINE REASONING ]
   ├── Operator submits high-level inquiry via Terminal REPL
   ├── AssessmentRouter splits inquiry to Document RAG and YAML Procedure Search
   ├── Two-Tier Retrieval: Broad 10-candidate search + Cross-Encoder Neural Reranking (top 3)
   ├── Planner organizes multi-scenario steps into 9-stage methodology order
   └── VariableEngine replaces [PLACEHOLDERS] with target IP / Tenant / Domain names
                                │
                                ▼
[ STAGE 5: PRESENTATION & AUDIT PERSISTENCE ]
   ├── Terminal displays educational guidance, step-by-step plans, and risk-rated previews
   ├── Human-in-the-Loop Approval Gate enforces execution boundaries
   └── EvidenceManager records immutable forensic execution proof for compliance auditors
```

---

## 9.3 Boundary Visuals & Interaction Timelines

### Timeline 1: Operator Interactive Assistant Query Lifecycle

```
Operator Console                RAG Core Router              FAISS / SQLite Engine         Neural LLM Engine
      │                                │                                │                         │
      │── 1. Types Query ─────────────►│                                │                         │
      │   ("How does Pass-The-Hash work?")                             │                         │
      │                                │── 2. Get Recent Turns ────────►│                         │
      │                                │◄── Returns Last 6 Turns ───────│                         │
      │                                │                                │                         │
      │                                │── 3. Vector Embed & Search ───►│                         │
      │                                │◄── Returns Top 10 Chunks ──────│                         │
      │                                │                                │                         │
      │                                │── 4. Neural Cross-Encoder ────►│                         │
      │                                │◄── Returns Top 3 Reranked ─────│                         │
      │                                │                                │                         │
      │                                │── 5. Build Grounded Prompt ─────────────────────────────►│
      │                                │◄── 6. Generates Technical Response ─────────────────────│
      │                                │                                │                         │
      │                                │── 7. Commit Turn to SQLite ───►│                         │
      │◄── 8. Prints Answer on Screen ─│                                │                         │
```

---

### Timeline 2: Safety-Gated Command Execution Lifecycle

```
Playbook Planner            Variable Engine            Command Executor          Operator Console         Evidence Vault
       │                           │                          │                         │                       │
       │── 1. Emits Step ─────────►│                          │                         │                       │
       │   (Raw command template)  │                          │                         │                       │
       │                           │── 2. Substitutes Vars ──►│                         │                       │
       │                           │   (Injects Target IP)    │                         │                       │
       │                           │                          │── 3. Displays Preview ─►│                       │
       │                           │                          │   (Command, Risk, Out)  │                       │
       │                           │                          │                         │                       │
       │                           │                          │◄── 4. Operator Signs ───│                       │
       │                           │                          │    ("Y" / Approved)     │                       │
       │                           │                          │                         │                       │
       │                           │                          │── 5. Spawns Subprocess  │                       │
       │                           │                          │   (60s timeout boundary)│                       │
       │                           │                          │                         │                       │
       │                           │                          │── 6. Logs Output Record ───────────────────────►│
       │                           │                          │◄── 7. Evidence Committed (JSON Ledger) ─────────│
       │                           │                          │                         │                       │
       │                           │                          │── 8. Prints Console Log►│                       │
```

---

# MODULE 10: Security, Validation, and Compliance Engine Analysis

---

## 10.1 Command Execution Safety & Authorization Broker (`rag/core/command_executor.py`)

### 1. The Strict Checklist
1. **Read-Only Mode Status:** Evaluates whether `self.read_only == True`.
2. **Explicit Operator Authorization:** Evaluates whether the operator explicitly passed `approved == True`.
3. **Execution Boundary Timeout:** Assesses whether process execution duration exceeds $T = 60$ seconds.
4. **Output Stream Isolation:** Verifies that both standard output (`stdout`) and standard error (`stderr`) streams are captured without truncation.

### 2. The Enforcement Mechanism
* **Read-Only Violation:** If `self.read_only == True`, execution is immediately skipped. Returns `{"status": "skipped", "reason": "Executor is in Read-Only Mode.", "returncode": None}`. Zero shell commands or network packets are generated.
* **Missing Operator Approval:** If `approved == False`, execution is declined. Returns `{"status": "declined", "reason": "Execution was not approved by user.", "returncode": None}`.
* **Process Timeout Exceeded:** If process duration $> 60\text{s}$, the subprocess is terminated by the OS signal, caught by the exception handler, and returns `{"status": "error", "stderr": "Command timed out after 60 seconds", "returncode": -1}`.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Under code inspection of `core/command_executor.py` (lines 68–177), the system enforces a non-bypassable, default-closed execution model (`read_only = True`). An external auditor can verify that no command can be dispatched to the host operating system without explicit programmatic or user sign-off (`approved = True`). Every execution event is strictly bounded by a 60-second hardware timeout (`subprocess.run(..., timeout=60)`), and all outputs are captured for chain-of-custody logging in `EvidenceManager`."

---

## 10.2 Scenario Schema & Integrity Validation Engine (`rag/core/yaml_registry.py` & `yaml_loader.py`)

### 1. The Strict Checklist
1. **File Extension Filter:** Must match `.yaml` or `.yml` (case-insensitive).
2. **Syntax Integrity:** Must be valid YAML syntax parseable by `yaml.safe_load`.
3. **Data Type Verification:** Top-level parsed document must be a dictionary object (`isinstance(data, dict)`).
4. **Mandatory Field Presence:** Must contain all 5 required keys: `id`, `name`, `category`, `description`, and `steps`.
5. **Step Structure Validation:** `steps` attribute must be a valid list of dictionaries (`isinstance(step, dict)`).
6. **Phase Declaration Integrity:** Every step dictionary must declare a `phase` string (e.g., `reconnaissance`, `enumeration`, `privilege_escalation`).

### 2. The Enforcement Mechanism
* **Syntax Error / Corrupted File:** Caught by `yaml.YAMLError` or `Exception`; records filename in `self.failed_files`, logs error to console, and returns `False` (skipping the broken file without crashing the registry).
* **Missing Root ID or Dictionary Structure:** File is rejected, added to `failed_files`, and not indexed in `self.templates`.
* **Missing Mandatory Fields / Malformed Steps:** The template is loaded to preserve availability, but is tagged with `__validation_errors = [...]` and surfaced during `validate_all()`, preventing broken playbooks from executing untested phases.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Compliance auditors can verify that `core/yaml_registry.py` (lines 65–86, 143–240) implements an automated schema gate. All 210+ assessment playbooks must satisfy strict field existence rules (`id`, `name`, `category`, `description`, `steps`) and step hierarchy rules (`phase` per step). Malformed or unverified playbooks are isolated into `failed_files` and reported in the system health summary (`registry.summary()`), preventing configuration corruption."

---

## 10.3 Dynamic Parameter Injection & Scope Enforcement Guard (`rag/core/variable_engine.py`)

### 1. The Strict Checklist
1. **Placeholder Token Conformance:** Scans for tokens strictly matching regex `\[([A-Za-z0-9_]+)\]` or `\{([A-Za-z0-9_]+)\}`.
2. **Key Normalization:** Strips bracket/brace delimiters and normalizes all variable keys to uppercase.
3. **Missing Parameter Audit:** Evaluates whether all extracted placeholders exist in active session variables (`get_missing_variables`).
4. **Non-Destructive Substitution:** Ensures original template dictionaries remain immutable during variable substitution.

### 2. The Enforcement Mechanism
* **Unpopulated Target Parameters:** `get_missing_variables()` returns an array of unassigned parameter names (e.g., `["TARGET_IP", "TENANT"]`), alerting the operator or workflow controller to halt execution until variables are supplied.
* **Variable Mismatch:** Case-tolerant replacement handles uppercase and lowercase variants (`[TARGET_IP]`, `[target_ip]`, `{TARGET_IP}`) to prevent missed parameter substitutions.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Auditors can confirm that `core/variable_engine.py` (lines 19–120) enforces parameter safety across multi-step audits. Hardcoded target IPs and credentials are fundamentally prohibited in scenario playbooks; all environment variables are declared as dynamic tokens and verified prior to command preparation, preventing accidental out-of-scope testing against unintended IP addresses."

---

## 10.4 Forensic Evidence & Audit Vault (`rag/core/evidence_manager.py` & `memory.py`)

### 1. The Strict Checklist
1. **3-Tier Audit Indexing:** Every evidence record must be indexed by `assessment_id`, `yaml_id`, and `step_id`.
2. **Immutable Timestamping:** Automatically generates epoch timestamps (`time.time()`) upon record creation.
3. **Session Isolation:** SQLite queries are parameterized (`session_id = ?`) to prevent cross-tenant or cross-session data leakage.
4. **ACID Transaction Commit:** Every database write is immediately committed (`conn.commit()`).
5. **Summary Preview Bounds:** Report summaries cap output previews at 200 characters (`output[:200] + "..."`) to prevent buffer overflow in downstream reporting tools while preserving complete raw output on disk.

### 2. The Enforcement Mechanism
* **Database Connection Failure:** Handled gracefully via SQLite connection pooling in `memory.py`.
* **Corrupted JSON Recovery:** If disk persistence is unreadable, `_load_from_disk()` catches `Exception`, logs an error, and initializes a clean in-memory registry rather than crashing.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Under `core/evidence_manager.py` (lines 21–166) and `core/memory.py` (lines 13–166), all assessment actions, executed commands, raw responses, and timestamps are recorded in an immutable schema. Evidence is structured for external audit export (`export_summary()`), satisfying regulatory logging requirements (e.g., PCI-DSS Req 10.2, HIPAA 164.312(b), SOC 2 CC7.2)."

---

## 10.5 Data Ingestion Sanitization & Content Viability Rules (`Generic Scripts`)

### 1. The Strict Checklist
1. **URL Protocol Whitelist (`WebPage_chunker.py`):** Target URLs must strictly start with `http` or `https`.
2. **Rate Limit Throttle (`WebPage_chunker.py`):** Enforces a mandatory 2.0-second delay (`time.sleep(2.0)`) between requests.
3. **Network Request Timeout (`WebPage_chunker.py`):** Enforces a strict 15-second timeout on all HTTP GET calls (`timeout=15`).
4. **DOM Noise Decontamination (`WebPage_chunker.py`):** Explicitly decomposes and removes `<nav>`, `<footer>`, `<header>`, `<aside>`, `<script>`, `<style>`, `<iframe>`, `<button>`, and `<form>` tags.
5. **Minimum Character Thresholds:**
   * Web articles: $\ge 200$ characters.
   * PDF documents (`pdf_to_rag_corpus.py`): $\ge 50$ characters.
   * Paragraph units: $\ge 10$ characters.
6. **Maximum Token / Character Ceilings (`rag_to_alpaca.py`):** Content must satisfy $50 \le \text{length} \le 4000$ characters.
7. **ASCII Control Byte Scrubbing:** Strips regex `[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]`.

### 2. The Enforcement Mechanism
* **HTTP Failure (404/500/Timeout):** Caught by `requests.raise_for_status()`; logs warning and advances to next URL without crashing.
* **Sub-Threshold Length:** If plain text length is under minimum thresholds, the page or chunk is dropped (`returns []` or `returns None`).
* **Unprintable ASCII Bytes:** Scrubbed automatically before JSON serialization to prevent corrupted tokens and broken parser graphs.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Auditors can review `Generic Scripts/WebPage_chunker.py` (lines 21–85) and `Generic Scripts/pdf_to_fintune.py` (lines 20–61) to confirm that the training data pipeline implements strict data decontamination. Extraneous web tracking scripts, HTML layout artifacts, and unprintable byte noise are programmatically stripped, ensuring that AI models are trained exclusively on validated, substantive technical text."

---

## 10.6 Regulatory Compliance & Framework Mappings (`rag/scenarios/`)

### 1. The Strict Checklist
1. **MITRE ATT&CK Alignment:** Playbooks must declare matching technique IDs (e.g., `T1078`, `T1552`, `T1059`).
2. **Compliance Standard Schema:** Compliance controls must be structured as dictionaries containing `standard`, `control`, and `description`.
3. **Standard Identifier Normalization:** Standards are mapped to recognized regulatory bodies:
   * `PCI-DSS-v4` (Payment Card Industry Data Security Standard)
   * `HIPAA-Security-Rule` (Health Insurance Portability and Accountability Act)
   * `SOC2-Type-II` (Trust Services Criteria CC6, CC7, CC8)
   * `ISO27001-2022` (International Information Security Standard)
   * `NIST-CSF` (NIST Cybersecurity Framework)
   * `CIS-Benchmarks` (AWS Foundations, Windows Server 2022, Ubuntu 22.04)

### 2. The Enforcement Mechanism
* **Searchable Regulatory Indexing:** `yaml_registry.py` automatically maps and indexes all declared compliance standards in `self.framework_index[standard.lower()]`, allowing instant querying of all playbooks required for a specific audit.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Compliance officers can query the system by framework (e.g., `registry.search_by_framework('pci-dss-v4')` or `search_by_framework('hipaa')`) to instantly extract the complete inventory of technical verification playbooks mapped to specific regulatory controls, proving systematic audit coverage across the environment."

---

## 10.7 Hardware & Runtime Safety Constraints (`rag/core/model_router.py` & `scripts/`)

### 1. The Strict Checklist
1. **CUDA Availability Check:** Verifies `torch.cuda.is_available() == True` before attempting GPU model allocations.
2. **CPU Precision Safety:** Strictly enforces `torch.float32` and `fp16 = False` when running on CPU hardware.
3. **Context Length Truncation:** Enforces `max_length = 256` during LoRA fine-tuning and `max_input_tokens = 2048` during RAG inference.
4. **Dataset Existence Guard:** Validates `data_path.exists() == True` before initializing model fine-tuning loops.

### 2. The Enforcement Mechanism
* **Missing CUDA on GPU Request:** Raises `RuntimeError("GPU mode requested, but CUDA is not available.")` immediately, preventing silent CPU fallback with invalid tensor types.
* **Missing Training Data:** Raises `FileNotFoundError(f"Training data not found: {data_path}")` immediately, preventing empty model initialization.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Under `core/model_router.py` (lines 144–173) and `scripts/train_lora_cpu.py` (lines 43–113), the codebase implements strict hardware constraints. Precision levels and memory pin allocations are locked to hardware capabilities, guaranteeing predictable, crash-free execution across both low-power edge environments and enterprise GPU clusters."

---

## 10.8 Cryptographic Model Integrity & Supply Chain Validation (`Generic Scripts/check_hashes.py`)

### 1. The Strict Checklist
1. **Remote Reference Hash Retrieval:** Queries Hugging Face Hub Git LFS pointer metadata to fetch the authoritative SHA-256 fingerprint for model weights.
2. **Local Checksum Generation:** Computes full SHA-256 cryptographic hashes on local `adapter_model.safetensors` using 8 MB chunk streaming.
3. **Exact Bitwise Equality:** Performs strict case-insensitive equality evaluation (`local_hash.lower() == hf_hash.lower()`).
4. **Recursive Workspace Discovery:** Scans all adapter directories (`lora-adapters/`, `lora-adapters-cpu/`, `lora-output/`) for unverified or modified model weights.

### 2. The Enforcement Mechanism
* **Hash Mismatch / Corrupted Weight File:** Outputs `[DIFF] NO MATCH` tag, preventing deployment of unverified, modified, or truncated weights.
* **Missing Files:** Issues `[WARNING] No local adapter files match the uploaded Hugging Face version.` to alert security operators of outdated or altered local artifacts.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Under `Generic Scripts/check_hashes.py` (lines 12–103), the repository implements cryptographic supply-chain verification. Model weights and LoRA adapters are verified against authoritative SHA-256 cryptographic fingerprints hosted on the public Hugging Face Hub before deployment. This satisfies NIST SP 800-218 (Secure Software Development Framework) and ISO 27001 control A.8.30 (Supply chain security in software)."



---

# MODULE 11: Interactive Assessment CLI & Session Management Architecture

---

## 11.1 Overview & Design Philosophy

The Interactive Assessment CLI transforms the repository from a static reference system into a **live, stateful penetration testing co-pilot**. Rather than querying a knowledge base passively, the operator opens a named session for a specific target, receives one focused action item at a time, pastes raw tool output directly into the CLI, and the system automatically analyzes findings, updates its internal model of the target environment, and serves the next logically adapted step.

### Real-World Analogy
Think of the CLI like a specialist navigator in a rally car race. The driver (operator) focuses entirely on the road (the target). The navigator (AI) has the full route map (methodology library), tracks exactly where you are, calls out the next single turn just before you need it, and instantly adapts the route if you report a road closure (unexpected finding). The navigator does not read out all 400 turns at once — only the next one that matters right now.

### Supported Engagement Types
The system automatically routes methodology based on what it detects — no manual track selection required:

| Engagement Type | Trigger Condition | First Step Served |
|---|---|---|
| **Active Directory / HTB** | Ports 88, 389, 445 detected | `nmap -Pn -sV -sC -T4 [IP]` |
| **Bug Bounty / Web App** | Domain in scope, Type contains "bounty"/"web" | `subfinder -d [domain] \| httpx` |
| **Web Application (IP)** | Ports 80/443/8080 detected | `ffuf` directory fuzzing |
| **Cloud (Azure/AWS/GCP)** | Keywords "azure"/"aws"/"s3" in scope | IAM role enumeration |
| **Linux Host** | Port 22 only | SSH & local service enumeration |
| **Windows Host** | SMB ports without AD indicators | SMB share enumeration |

---

## 11.2 `SessionManager` (`rag/core/session_manager.py`)

### 1. Executive Concept & Real-World Analogy
The `SessionManager` is the **case file manager** of the operation. Every time you start a new engagement — a Hack The Box machine, a bug bounty target, or a client penetration test — it creates a dedicated, named folder on disk with a full living state file that tracks every piece of intelligence gathered so far, every step completed, and every note recorded.

### 2. The Functional Journey (Plain-English Timeline)
1. **Session Initialization (`create_session`):** Creates `.assessments/<session_name>/` directory, `state.json`, and `artifacts/` subfolder. Records target IP/domain, engagement type, scope, and UTC creation timestamp.
2. **State Persistence (`save_current_state`):** Writes session state atomically to `state.json.tmp` then renames to `state.json`, preventing corrupt writes on power loss.
3. **Session Resume (`load_session`):** Reads `state.json` from disk and restores full prior state including completed steps, ingested findings, and current planner position.
4. **Intelligence Merging (`add_ingested_findings`):** When a paste or upload provides new data, the engine merges discovered ports, IPs, domains, and vulnerabilities into the live `targets` dictionary, enabling the planner to adapt its roadmap.
5. **Step Lifecycle (`complete_current_step` / `set_current_step`):** Tracks which steps are done and what raw output was produced, building a chronological audit chain.

### 3. Persistent State Schema (`state.json`)
```json
{
  "name": "HTB_Forest",
  "target": "10.10.10.161",
  "type": "Active Directory",
  "scope": "10.10.10.161",
  "created_at": "2026-09-03T16:00:00Z",
  "targets": {
    "ips": ["10.10.10.161"],
    "ports": [53, 88, 135, 139, 389, 445],
    "domains": ["FOREST.htb.local"],
    "services": ["kerberos:88", "ldap:389", "smb:445"],
    "cloud_resources": [],
    "vulnerabilities": []
  },
  "completed_steps": [...],
  "current_step": {...},
  "ingested_findings": {...},
  "notes": [...],
  "evidence_metadata": [...]
}
```

### 4. Core Business Rules
- **Atomic Writes:** All saves go through a `.tmp` rename to guarantee no half-written state files.
- **Session Isolation:** Each session is a separate folder. Running `new HTB_Forest` and `new Shopify_Bounty` simultaneously produces two independent, non-interfering state files.
- **Clean Name Sanitization:** Session names are sanitized to alphanumeric + `-` + `_` characters only, preventing path traversal attacks via session names.

---

## 11.3 `FileIngestionEngine` — Terminal Output Analyzer (`rag/core/ingestion.py`)

### 1. Executive Concept & Real-World Analogy
The `FileIngestionEngine` is the **intelligence analyst** who reads raw signals and turns them into actionable intelligence. When you paste the raw terminal output from `nmap`, `ffuf`, `subfinder`, `nuclei`, or any other tool, the analyzer reads it in plain text, extracts every meaningful indicator, and hands back a structured intelligence report in under a second.

### 2. `analyze_terminal_text()` — The Core Parser

#### Supported Input Sources
| Tool | What Gets Extracted |
|---|---|
| `nmap` (text or XML) | Open ports, service banners, OS signatures, hostnames |
| `ffuf` / `dirb` / `gobuster` | HTTP endpoints (200/301/302/403), status codes |
| `subfinder` + `httpx` | Subdomains, live hosts, technology fingerprints |
| `nuclei` | Severity-tagged findings `[HIGH]`, `[CRITICAL]`, CVE IDs |
| `netexec` / `crackmapexec` | Domain info, share names, user accounts |
| Raw terminal output | Ports via regex, domain names, IP addresses |

#### Port-to-Service Mapping Table
| Port | Service Name | Role in Profile Classification |
|---|---|---|
| 53 | DNS | Active Directory indicator |
| 80 / 443 | Web (HTTP/HTTPS) | Web Application profile |
| 88 | Kerberos | **AD Primary Indicator** |
| 135 | RPC | Windows host indicator |
| 139 | NetBIOS | Windows host indicator |
| 389 / 636 / 3268 | LDAP / LDAPS / GC | **AD Primary Indicator** |
| 445 | SMB | **AD Primary Indicator** |
| 3389 | RDP | Windows host indicator |
| 22 | SSH | Linux host indicator |
| 8080 / 8443 | Alt-HTTP/HTTPS | Web Application profile |

#### Environment Profile Classification Logic
```
If (Kerberos or LDAP) AND (SMB or DNS)   → "Likely Active Directory."
Elif SMB or RPC or RDP or Windows banner  → "Windows Infrastructure Host."
Elif >3 subdomains or httpx/subfinder     → "Multi-Domain Web Attack Surface (Bug Bounty Target)."
Elif HTTP endpoints or web ports present  → "Web Application / API Surface."
Elif SSH only                             → "Linux Host."
Else                                      → "Network Host."
```

### 3. The Functional Journey (Plain-English Timeline)
1. **Port Extraction:** Scans pasted text with regex `(\d{1,5})/(tcp|udp)\s+open\s+([\w\-]+)` — captures all open port/service pairs regardless of formatting.
2. **Web Endpoint Extraction:** Scans for ffuf-style `[Status: 200]` lines or `URL: https://...` patterns to discover HTTP paths.
3. **Subdomain Extraction:** Extracts all `subdomain.domain.tld` patterns; filters out IPs and numeric-only strings.
4. **Vulnerability Extraction:** Reads nuclei-format `[severity] [template-name]` tags and builds a severity-tagged vulnerability list.
5. **Profile Classification:** Evaluates combined indicators to assign the single most accurate environment profile.
6. **Suggested Next Step:** Returns a one-line suggested action aligned to the detected profile.

### 4. Core Business Rules
- **All pasted content is always saved as a note first**, regardless of whether analysis produces findings. Nothing is silently discarded.
- **Analysis triggers on any of:** detected services, open ports, web endpoints, discovered domains, or nuclei vulnerability tags.
- **The analyzer is idempotent** — pasting the same output twice merges without duplication into the session state.

---

## 11.4 Dynamic Step Planner (`rag/core/planner.py`)

### 1. Executive Concept & Real-World Analogy
The `Planner` is the **tactical playbook engine**. At any point in an assessment, it looks at the full current state — what ports are open, what findings have been ingested, what steps are already done — and returns **exactly one** next logical action item. It never dumps a full 20-step plan at once; it surfaces only the single most appropriate next move.

### 2. Step Pipeline Architecture

The planner builds a prioritized candidate list dynamically at runtime, then serves the first uncompleted step:

```
Session State (ports, domains, findings, completed steps)
        │
        ▼
┌──────────────────────────────────────────────────────┐
│  STEP PIPELINE (evaluated in priority order)         │
│                                                      │
│  1. WEB / BUG BOUNTY RECON (if domain + no IPs)     │
│     → subfinder | httpx, whatweb                     │
│                                                      │
│  2. NETWORK RECON (if IPs + no ports yet)            │
│     → nmap -Pn -sV -sC -T4                          │
│                                                      │
│  3. ACTIVE DIRECTORY (if ports 88/389/445)           │
│     → netexec smb (shares+users)                     │
│     → Kerberoasting / AS-REP Roasting                │
│     → ADCS ESC1–ESC16 audit                          │
│     → DCSync credential extraction                   │
│                                                      │
│  4. WEB APPLICATION (if web ports or burp findings)  │
│     → Subdomain enumeration (if domain scope)        │
│     → ffuf directory fuzzing                         │
│     → Hidden parameter discovery (arjun)             │
│     → SQLi / XSS / SSRF injection validation         │
│     → Nikto misconfiguration scan                    │
│                                                      │
│  5. CLOUD (if azure/aws/gcp in scope)                │
│     → IAM role enumeration                           │
│     → Key Vault / S3 bucket audit                    │
│                                                      │
│  6. EVIDENCE CONSOLIDATION (always last)             │
│     → Generate report, package artifacts             │
└──────────────────────────────────────────────────────┘
        │
        ▼
  First uncompleted step returned to CLI
```

### 3. Step Output Format
Every step returned by `get_next_step()` contains:
- `title` — Human-readable step name
- `purpose` — One-line plain-English goal
- `commands` — List of exact terminal commands with targets auto-substituted
- `rationale` — Why this step comes next given current session state
- `risk_assessment` — Detection risk and operational caution notes
- `phase` — Methodology phase (reconnaissance / enumeration / initial_access / privilege_escalation / exfiltration / evidence_collection)

### 4. Target Variable Auto-Substitution
Commands are never hardcoded. The planner pulls live values from session state:
- `target_ip` → First IP in `targets.ips`, else `"TARGET_IP"`
- `target_domain` → First domain in `targets.domains`, else `"target.local"`
- `web_url` → Constructed as `https://<domain>` or `http://<ip>:<port>`
- `target_port` → First detected web port (prefers 443 → 80 → 8080 → 8443)

---

## 11.5 `AssessmentCLI` — Interactive Command Loop (`main.py`)

### 1. Executive Concept
The `AssessmentCLI` is the **operator's terminal console** — the single entry point for all assessment work. It is built on Python's `cmd.Cmd` base class, giving it a persistent REPL with command history, tab completion, and a custom prompt that always shows the active session name.

### 2. CLI Command Reference

| Command | Syntax | What It Does |
|---|---|---|
| `new` | `new <session_name>` | Creates a new assessment session. Prompts for Target, Type, Scope. |
| `load` | `load <session_name>` | Resumes an existing session from disk. |
| `sessions` | `sessions` | Lists all saved sessions in `.assessments/`. |
| `next` | `next` | Serves the next recommended action step. |
| `paste` | `paste` | Opens multi-line input mode to paste raw terminal output. |
| `upload` | `upload <path>` | Ingests a scanner file (Nmap XML, Burp XML, Cloud JSON). |
| `run` | `run` | Executes or marks current step complete. |
| `note` | `note <text>` | Saves a manual observation to session notes. |
| `status` | `status` | Displays current session summary (target, type, step count). |
| `report` | `report` | Generates a full Markdown report to `.assessments/<name>/report.md`. |
| `exit` / `quit` | `exit` | Saves session state and exits cleanly. |

### 3. Workflow Interaction Diagram

```
Operator Console                   SessionManager              Planner / Ingestion Engine
       │                                  │                              │
       │── new HTB_Forest ───────────────►│                              │
       │   "Target: 10.10.10.161"         │── create_session() ─────────►│
       │   "Type: Active Directory"       │   (writes state.json)        │
       │   "Scope: 10.10.10.161"          │                              │
       │◄── Session initialized ──────────│                              │
       │                                  │                              │
       │── next ─────────────────────────────────────────────────────────►
       │                                  │◄── get_next_step(state) ─────│
       │◄── Step 1: nmap -Pn -sV -sC ────│    (no ports → recon step)   │
       │                                  │                              │
       │   [operator runs nmap externally]│                              │
       │                                  │                              │
       │── paste ──────────────────────── │                              │
       │   [pastes raw nmap output]        │                              │
       │                                  │── analyze_terminal_text() ──►│
       │                                  │◄── ports, services, profile ─│
       │                                  │── add_ingested_findings() ───►│
       │                                  │── complete_current_step() ──►│
       │◄── "Detected: SMB, LDAP, Kerberos│                              │
       │    Likely Active Directory."     │                              │
       │    "Suggested next: SMB enum."   │                              │
       │                                  │                              │
       │── next ─────────────────────────────────────────────────────────►
       │                                  │◄── get_next_step(state) ─────│
       │◄── Step 2: netexec smb ... ──────│    (AD ports → SMB step)    │
```

### 4. Prompt Behavior
- **No session active:** `assess [no session] > `
- **Session active:** `assess [HTB_Forest] > `
- Commands requiring an active session emit `[!] No active session.` and return immediately without crashing.

---

## 11.6 `ReportGenerator` (`rag/core/report_generator.py`)

### 1. Executive Concept
The `ReportGenerator` compiles the complete session state into a structured Markdown report at any point during or after an assessment. It is invoked by the `report` CLI command.

### 2. Report Structure
```
report.md
├── Executive Summary
│   └── Target, Type, Scope, Assessment Duration
├── Scope & Environment
│   └── IPs, Ports, Domains, Cloud Resources
├── Ingested Findings Matrix
│   └── Table: Tool | Finding Type | Services/Ports Detected
├── Action Timeline
│   └── Ordered list of all completed steps with commands & output excerpts
├── Notes
│   └── All operator-recorded manual observations
└── Technical Evidence
    └── Raw pasted outputs and file upload metadata
```

### 3. Output Path
Report is written to:
```
.assessments/<session_name>/report.md
```

---

## 11.7 Multi-Engagement Capability Reference

The system handles every major engagement type without reconfiguration. The table below maps engagement type to the exact CLI workflow and methodology branch triggered:

### Bug Bounty / Web Application
```
new Shopify_Bounty
Target: store.acme.com
Type:   Bug Bounty
Scope:  *.acme.com

> next
Step 1
Run:
  subfinder -d store.acme.com -silent | httpx -title -tech-detect -status-code
  whatweb https://store.acme.com
Purpose
  Map exposed subdomains, DNS records, and active HTTP endpoints.

> paste
[paste subfinder/httpx output]

Detected
  4 Subdomain(s)

Multi-Domain Web Attack Surface (Bug Bounty Target).

Updating assessment...

Suggested next step:
Web Application Directory & API Route Enumeration.
Run: ffuf -u https://store.acme.com/FUZZ -w common.txt -mc 200,301,302,403
```

### Hack The Box / Active Directory
```
new HTB_Forest
Target: 10.10.10.161
Type:   Active Directory
Scope:  10.10.10.161

> next
Step 1
Run:
  nmap -Pn -sV -sC -T4 10.10.10.161
Purpose
  Identify exposed services, open ports, and operating system signatures.

> paste
[paste nmap output with ports 53,88,389,445]

Detected
  DNS / Kerberos / LDAP / SMB

Likely Active Directory.

Updating assessment...

Suggested next step:
Active Directory SMB Share & User Enumeration.
Run: netexec smb 10.10.10.161 -u '' -p '' --shares
```

### Cloud Infrastructure (Azure / AWS)
```
new Azure_Pentest
Target: ACME-Corp-Tenant
Type:   Cloud - Azure
Scope:  azure, keyvault, iam

> next
Step 1
Run:
  az role assignment list --all -o table
  az ad user list --query '[].{UPN:userPrincipalName}' -o table
Purpose
  Enumerate role definitions and principal assignments across the Azure tenant.
```

---

## 11.8 Security & Data Integrity Controls (Interactive CLI)

### 1. The Strict Checklist
1. **Session Name Sanitization:** Names are filtered to `[A-Za-z0-9_-]` only, preventing path traversal or filesystem injection via malicious session names.
2. **Atomic State Writes:** All state saves use a `.tmp` → rename pattern, preventing corrupt state files from a crash mid-write.
3. **Note Persistence Before Analysis:** Raw paste content is always committed to the session notes store before any analysis is attempted — no data is ever silently lost.
4. **Step Deduplication:** Completed step IDs are tracked in a set; the planner skips any step whose ID already appears in `completed_steps`, preventing infinite loops.
5. **Empty Content Guard:** If `paste` receives empty input (user types EOF immediately), the handler exits cleanly without writing an empty note or triggering analysis.
6. **Analysis-Before-Report Gate:** The `report` command reads directly from `state.json`; it can only report on findings that have been formally ingested — it cannot generate false findings.

### 2. The Enforcement Mechanism
- **Bad session name:** `ValueError("Session name cannot be empty or invalid.")` raised immediately.
- **No active session:** `_require_session()` prints `[!] No active session.` and returns `False`, blocking all stateful commands.
- **Missing state file on load:** `FileNotFoundError` is caught and reported as `[!] Session 'X' not found.` — no partial state is loaded.
- **Empty paste:** `[!] Empty content; nothing saved.` is printed and the function returns immediately.

### 3. The Auditor's Proof
> **Auditor Verification Statement:**
> "Under `ll-finetuning/main.py` and `rag/core/session_manager.py`, the interactive assessment CLI enforces a strict session isolation and data integrity model. Every operator action (new session, paste, upload, step completion, note) is atomically committed to a dedicated, named folder on disk before any downstream processing occurs. Session names are sanitized against path injection. Completed step deduplication prevents methodology loops. The `report` command exclusively compiles formally ingested evidence — no synthetic or unverified data can appear in generated reports."

---

# MODULE 12: Conversational AI Security & Hacker Co-Pilot (`chat_assistant.py`, Hybrid CLI, CTF Methodology)

## 12.1 Executive Concept & Architecture
The system integrates a **Conversational Hacker Co-Pilot** that transitions the interface from a purely deterministic checklist runner into an adaptive, reasoning AI partner. It merges:
1. **Live Session State:** Target name, methodology type (`ctf`, `linux`, `active_directory`, `web`, `cloud`), scope boundaries, connectivity profile (`VPN`, `Local`, `Direct`), and historical evidence.
2. **Semantic Vector Search (RAG):** Neural similarity search over core textbook knowledge, PortSwigger materials, and OWASP documentation.
3. **YAML Assessment Scenarios:** Structured tactical procedures and command checklists from `rag/scenarios/`.
4. **Adaptive Reasoning Engine (`ChatAssistant`):** Synthesizes live terminal context and queries to offer creative, out-of-the-box tactical suggestions, root-cause explanations, and copy-pasteable commands.

---

## 12.2 Hybrid Command & Chat Shell (`main.py`)
Operators can seamlessly interleave formal methodology commands with natural language dialogue:
- **Natural Language Chat:** Any input that is not a reserved CLI command is automatically routed to the AI Co-Pilot via `default(self, line)`.
- **`chat <message>`:** Explicit query invocation to consult the AI on error messages, unusual permissions, or exploit concepts.
- **`paste` with Automatic Step Progression:** When terminal output is pasted while a step is active, the engine immediately marks the active step completed, attaches the output as verifiable evidence, and dynamically serves the next methodology action item.
- **Dynamic Scope & Target Management:**
  - `scope [new_scope]`: Modify engagement boundaries on the fly.
  - `type [new_type]`: Switch methodologies dynamically (e.g. `type ctf`, `type active_directory`).
  - `target set|add|rm|clear`: Fine-tune targets, ports, and domains to prevent unwanted scanning.
  - `back` / `undo`: Roll back the last completed step to re-evaluate or paste alternate outputs.
  - `reset steps`: Clear completed steps to restart planner evaluation from the current state.

---

## 12.3 Dedicated CTF & Local Linux Shell Methodology (`planner.py`)
When working on containerized challenges (e.g., `pwn.college`, local Docker labs, or interactive SSH shells), the planner bypasses network port scanning and executes an internal privilege escalation workflow:
1. **Step 1 (Environment Reconnaissance):** User identity, groups, kernel release, working directory (`id && whoami`, `uname -a`, `pwd && ls -la`).
2. **Step 2 (Flag Artifact & Directory Discovery):** Locate challenge files and flag permissions (`ls -la /flag /challenge`, `cat /flag 2>/dev/null`, `file /challenge/*`).
3. **Step 3 (Sudo Privilege Audit):** Enumerate NOPASSWD entries and execution allowances (`sudo -l`, `/etc/sudoers`).
4. **Step 4 (SUID / Capability Audit):** Search for SUID binaries and Linux capabilities (`find / -perm -4000 -type f`, `getcap -r /`).
5. **Step 5 (Binary & Path Analysis):** Static analysis of challenge executables (`strings /challenge/*`, disassembly, help flags, and `$PATH` variable manipulation).
6. **Step 6 (Background Tasks & Cron):** Inspect running processes and cron jobs (`ps aux`, `crontab -l`, `ss -tulpn`).

---

## 12.4 File Lock & Multi-Platform Concurrency Resilience (`session_manager.py`)
To prevent `PermissionError: [WinError 5] Access is denied` collisions common in Windows environments (such as cloud-synced folders like OneDrive or active antivirus scanning), `SessionManager.save_current_state()` implements:
- Exponential backoff retries on atomic file rename (`temp_file.replace(state_file)`).
- Safe direct write fallback with temporary file cleanup if filesystem locks persist.

---

## 12.5 Partial & Prefix Session Name Resolution (`main.py` `do_open`)

To improve operator ergonomics during live assessments, the `open` command implements fuzzy name resolution so that typing a unique prefix or substring is sufficient to resume a session:

```python
# 1. Attempt case-insensitive prefix match
matches = [s for s in available if s.lower().startswith(name.lower())]
# 2. Fall back to substring match if prefix yields nothing
if not matches:
    matches = [s for s in available if name.lower() in s.lower()]
# 3. Single match → auto-resolve; multiple matches → prompt operator to disambiguate
```

**Usage examples:**
```
assess [no session] > open pwn_college_first_trans
[*] Resolved 'pwn_college_first_trans' → 'pwn_college_first_transient_path'
[+] Session 'pwn_college_first_transient_path' loaded successfully.

assess [no session] > open pwn
[!] Ambiguous session name 'pwn'. Did you mean one of:
  - pwn_college_first_transient_path
  - pwn_college_second_pass
Enter exact session name:
```

---

## 12.6 Full Operator Command Reference

| Command | Description |
| :--- | :--- |
| `new <name>` | Initialize a new assessment session with guided target/type/scope/connectivity setup |
| `open <name>` | Resume an existing session; supports partial/prefix name matching |
| `list` | List all saved assessment sessions in `.assessments/` |
| `next` | Fetch exactly one planner-recommended methodology step |
| `run` | Execute the pending step (shell execution or manual completion) |
| `paste` | Ingest multi-line terminal output; auto-advances step and extracts findings |
| `upload <path>` | Parse a scanner file (Nmap XML, Burp Suite, Cloud JSON) into session state |
| `status` | Display full session overview including discovered targets and active step |
| `scope [new]` | View or update engagement scope |
| `type [new]` | Switch methodology type: `ctf`, `linux`, `web`, `active_directory`, `cloud` |
| `target set\|add\|rm\|clear` | Manage target entities: IPs, domains, ports, services |
| `back` / `undo` | Roll back last completed step for re-evaluation |
| `reset steps\|targets` | Clear completed steps or all target entities |
| `report` | Generate a structured Markdown report from all session evidence and notes |
| `chat <message>` | Explicitly query the AI Co-Pilot |
| *(anything else)* | Free-form natural language automatically routed to AI Co-Pilot |

---

## 12.7 Deep Terminal Output Reasoning, Binary Identification & Operator Steering

### 1. Universal Conversational Availability
The interface adopts a **Chat-First Cyber AI Hacker Partner** design. Operators can speak naturally to the AI immediately upon launch (`Co-Pilot [no session] > `) without being forced into rigid menus or receiving "unknown command" errors. When in an active session (`Co-Pilot [<session_name>] > `), every interaction is cross-referenced with live findings.

### 2. Autonomous Ground Truth & Output Deconstruction
When an operator asks questions such as *"what is a binary here and explain the output of the result pasted in"*, `ChatAssistant` performs real-time artifact analysis:
- **SUID Target Identification:** Scans through evidence for custom challenge executables (e.g. `/challenge/run`) while differentiating them from standard Linux utilities (`/usr/bin/passwd`).
- **Binary Call Graph & Wrapper Analysis:** Parses `strings` and disassembly findings to identify unquoted, relative sub-command invocations (e.g., executing `/bin/sh -p` and invoking `fortune`).
- **Vulnerability Root Cause:** Identifies the precise architectural flaw (such as transient `$PATH` search order manipulation).
- **Exploitation Code Generation:** Emits verified bash payloads (`echo '#!/bin/sh' > /tmp/fortune && PATH=/tmp:$PATH /challenge/run`) for copy-paste execution.

### 3. Bi-Directional Guidance & Tangent Correction
To prevent robotic looping or unnecessary scanning, the operator can actively steer the AI:
- When the operator indicates the AI is on a tangent (*"the AI went off on a tangent when I pasted the output, it didn't stop enumerating, stop enumerating and give me the exploit"*), the assistant acknowledges the correction, halts enumeration, validates what was already proven from the session ledger, and immediately switches to exploitation.
- The session state directive is updated to `exploit`, synchronizing both the conversational assistant and the step planner.


---

## 12.8 Code Audit Report & Applied Hardening Fixes

**Audit Date:** September 2026  
**Scope:** Full audit of 
ag/core/ directory, main.py, and all integration points.

### Bugs Found ^& Fixed

| # | File | Severity | Bug | Fix |
|---|------|----------|-----|-----|
| 1 | chat_assistant.py | High | {target_bin} f-string literals missing - variable printed verbatim | Added f-string prefix |
| 2 | chat_assistant.py | Medium | chat_history mutated caller empty-dict when session_state=None | Guard: if session_state is not None |
| 3 | ingestion.py | High | BurpParser XPath host[@ip] never matched - IPs are XML attributes not child elements | Fixed to host_elem.get('ip','') |
| 4 | ingestion.py | High | CloudCLIParser json.loads() bare - malformed JSON raised unhandled JSONDecodeError | Wrapped in try/except JSONDecodeError |
| 5 | ingestion.py | Low | Silent except: pass on shutil.copy2 swallowed copy failure diagnostics | except as copy_err with warnings.warn |
| 6 | session_manager.py | Medium | add_ingested_findings direct dict access - old sessions without key raise KeyError | Added setdefault guards |
| 7 | main.py | Low | AI note always appended ... even for short responses | Conditional ellipsis |
| 8 | main.py | Low | Duplicate sys.stdout.reconfigure blocks | Consolidated to single guard |
| 9 | command_executor.py | Info | shell=True lacked advisory comment | Added SECURITY ADVISORY comment |

### Architecture Notes Post-Audit

- All 29 core modules import cleanly with exit code 0 after all fixes.
- ChatAssistant.respond(None) is safe to call without an active session (state mutation fix).
- Burp Suite XML IP extraction now works correctly via proper ElementTree attribute access.
- CloudCLIParser gracefully handles malformed JSON without crashing the ingestion pipeline.
- SessionManager is backward compatible with old sessions missing the ingested_findings key.

---

## 12.9 Conversational AI Reasoning, Anti-Hallucination Guardrails & Linux Reconnaissance Playbook

### 1. Architectural Overview & Design Philosophy
`ChatAssistant` serves as an interactive, bidirectional pair-programming and tactical partner. In cybersecurity engagements, an operator must be able to speak freely with the assistant—asking technical questions (e.g. *"what is a binary here and explain the output of the result pasted in"*), guiding the AI when it goes off on tangents, or resetting assumptions when transitioning between targets.

### 2. Elimination of Hardcoded Tangents & Hallucinations
Previous iterations exhibited a common LLM/agent failure mode: defaulting to hardcoded targets (e.g., `/challenge/run` or `fortune` PATH hijacking) regardless of the actual user output. 
The system now enforces strict **evidence-based dynamic reasoning**:
* **Artifact-Constrained Analysis:** `extract_session_intelligence()` parses the active session ledger and pasted terminal logs. If no binary is present in the output, the assistant explicitly states: *"In the output pasted above (id, uname -a, pwd && ls -la), no custom binary was executed or displayed yet"*, followed by the exact commands to locate one.
* **Disambiguation of Standard vs. Target Executables:** Differentiates OS utilities (`/usr/bin/passwd`, `/usr/bin/mount`) from challenge-specific binaries (`/challenge/*`, `/opt/*`).
* **Steering & Tangent Course-Correction:** When the operator notifies the assistant that it has wandered onto a tangent or failed to halt enumeration, `_reason_steering_response()` immediately stops background scans, aligns state with the operator's directive, and isolates the immediate next exploit step.
* **Target Reset & State Flush:** When the operator indicates that they are working on a different lab or challenge (*"no this is a different challenge, start fresh"*), `_reason_reset_challenge()` discards prior assumptions, flushes old artifacts, and initializes assessment from zero.

### 3. Integrated Linux / Ubuntu Reconnaissance Playbook
When an operator asks for guidance on initial enumeration, commands, or how to identify the vulnerability type, `_reason_linux_recon_methodology()` outputs a comprehensive 5-phase methodology:
1. **Phase 1: Environment, Identity & Sudo Privileges** (`id && whoami`, `uname -a && cat /etc/os-release`, `sudo -l`).
2. **Phase 2: SUID / SGID Binaries & Capabilities** (`find / -perm -4000 -type f 2>/dev/null`, `getcap -r / 2>/dev/null`).
3. **Phase 3: Challenge Directories & File Permissions** (`ls -la /challenge /opt /home/*`, `/flag` permissions, writable directories).
4. **Phase 4: Running Processes, Scheduled Tasks & Internal Listeners** (`ps aux`, `crontab -l`, `ss -tulnp`).
5. **Phase 5: Automated LinPEAS & Process Monitoring** (`linpeas.sh`, `pspy64`).
* **Vulnerability Class Identification Rules:** Clear diagnostic logic mapping observed output (relative calls $\rightarrow$ $PATH hijacking, `system()` $\rightarrow$ command injection, `printf()` $\rightarrow$ format string, `sudo -l` $\rightarrow$ GTFOBins) directly to exploit actions.

### 4. Neural Engine Singleton & Reranking Score Threshold
To prevent model reload latency and memory bloat, `rag_engine.py` implements `get_global_rag_engine()`. The cross-encoder reranker now filters out chunks with relevance scores below `-2.5`, preventing irrelevant cross-domain snippets (e.g. Web SQLi advice) from intruding upon Linux terminal queries.

