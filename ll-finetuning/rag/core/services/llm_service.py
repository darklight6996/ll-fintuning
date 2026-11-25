"""
core/services/llm_service.py

Singleton service that owns ModelRouter instantiation.

Auto-detects GPU/CPU, selects the appropriate model from config, and
exposes a clean generate() interface. All consumers call get_llm_service()
instead of constructing ModelRouter directly — eliminating duplicate
model weight loading across the process.

Usage:
    from core.services.llm_service import get_llm_service
    svc = get_llm_service()
    if svc:
        response = svc.generate("Explain SQL injection.")
"""

import os
from typing import Optional, Any, Dict

# M2: keyed by role ("default", "planner", ...) instead of a single instance, so an
# optional planner-specific model (PLANNER_MODEL_NAME) can be loaded and cached
# alongside the default one without reloading either on every call. When no
# override is configured, every role resolves to the SAME cached "default"
# instance -- see get_llm_service() below -- so this adds no memory cost for
# anyone who hasn't opted in.
_INSTANCES: Dict[str, "LLMService"] = {}
_LOAD_ATTEMPTED: Dict[str, bool] = {}


class LLMService:
    """
    Thin facade over ModelRouter.

    Responsibilities
    ----------------
    * Select GPU or CPU model based on CUDA availability.
    * Load the model once via ModelRouter.
    * Expose generate() and tokenizer for prompt building.
    * Expose mode, model_name for diagnostics.
    """

    def __init__(
        self,
        verbose: bool = True,
        model_name_override: Optional[str] = None,
        max_input_tokens_override: Optional[int] = None,
        max_new_tokens_override: Optional[int] = None,
    ):
        self._router = None
        self.mode: str = "cpu"
        self.model_name: str = ""
        self.tokenizer: Optional[Any] = None
        self._load(
            verbose=verbose,
            model_name_override=model_name_override,
            max_input_tokens_override=max_input_tokens_override,
            max_new_tokens_override=max_new_tokens_override,
        )

    def _load(
        self,
        verbose: bool = True,
        model_name_override: Optional[str] = None,
        max_input_tokens_override: Optional[int] = None,
        max_new_tokens_override: Optional[int] = None,
    ) -> None:
        try:
            # ------------------------------------------------------------------
            # OLLAMA BACKEND (2026-11-19). If OLLAMA_BASE_URL is set, route all
            # generation to an Ollama server over HTTP instead of loading weights
            # in-process. This is how the hunter uses a bigger model than the
            # 3060 Ti can hold (e.g. Qwen3.6-35B-A3B on the 64GB main PC: Ollama
            # does the GPU/CPU split, the MoE keeps it fast). Unset = the original
            # in-process transformers/PEFT path below, byte-identical to before.
            #   OLLAMA_BASE_URL   e.g. http://192.168.1.50:11434  (from WSL, the
            #                     Windows host IP; Ollama must bind 0.0.0.0, see
            #                     OLLAMA_HOST). Required to enable this backend.
            #   OLLAMA_MODEL      e.g. qwen3.6:35b-a3b-q4_K_M      (the pulled tag)
            #   OLLAMA_NUM_CTX    context window to request (default 16384)
            #   OLLAMA_TIMEOUT    per-request seconds (default 300; first call
            #                     also pays Ollama's model-load)
            _ollama_url = os.environ.get("OLLAMA_BASE_URL")
            if _ollama_url:
                self._backend = "ollama"
                self.mode = "ollama"
                self.model_name = (model_name_override or os.environ.get("OLLAMA_MODEL")
                                   or "qwen3.6:35b-a3b")
                self._ollama_url = _ollama_url.rstrip("/")
                self._ollama_num_ctx = int(os.environ.get("OLLAMA_NUM_CTX", "16384"))
                self._ollama_timeout = int(os.environ.get("OLLAMA_TIMEOUT", "300"))
                # Qwen3.6 is a reasoning model; thinking ON emits <think>..</think>
                # blocks that break the planner JSON contract. Default OFF. Set
                # OLLAMA_THINK=1 to experiment with reasoning mode.
                self._ollama_think = os.environ.get("OLLAMA_THINK", "0") == "1"
                # Planner calls pass max_new_tokens=512, too small for a verbose model
                # to finish a proposal that embeds a code field -> truncated JSON. Use
                # a generous ceiling (a cap, not a target: greedy stops at the JSON end).
                self._ollama_num_predict = int(os.environ.get("OLLAMA_NUM_PREDICT", "2048"))
                self._router = None
                self.tokenizer = None
                if verbose:
                    print(f"[+] LLMService ready (ollama @ {self._ollama_url}): {self.model_name}")
                return

            import torch
            from core.model_router import ModelRouter
            from core.config import (
                CPU_MODEL_NAME, GPU_MODEL_NAME,
                CPU_MAX_INPUT_TOKENS, CPU_MAX_NEW_TOKENS,
                GPU_MAX_INPUT_TOKENS, GPU_MAX_NEW_TOKENS,
            )

            if torch.cuda.is_available():
                self.mode = "gpu"
                self.model_name = model_name_override or GPU_MODEL_NAME
                max_in = max_input_tokens_override or GPU_MAX_INPUT_TOKENS
                max_new = max_new_tokens_override or GPU_MAX_NEW_TOKENS
            else:
                self.mode = "cpu"
                self.model_name = model_name_override or CPU_MODEL_NAME
                max_in = max_input_tokens_override or CPU_MAX_INPUT_TOKENS
                max_new = max_new_tokens_override or CPU_MAX_NEW_TOKENS

            self._router = ModelRouter(
                model_name=self.model_name,
                mode=self.mode,
                max_input_tokens=max_in,
                max_new_tokens=max_new,
            )
            self.tokenizer = getattr(self._router, "tokenizer", None)

            if verbose:
                print(f"[+] LLMService ready ({self.mode}): {self.model_name}")
                if self.mode == "cpu":
                    print("[!] WARNING: this is the CPU FALLBACK model (google/flan-t5-large) -- "
                          "far too small for autonomous hunting; any assessment with it is MEANINGLESS. "
                          "The real model (Qwen3.6-35B via Ollama on the GPU PC) is used ONLY when "
                          "OLLAMA_BASE_URL is set. Export OLLAMA_BASE_URL + OLLAMA_MODEL and re-run.")

        except Exception as exc:
            print(f"[!] LLMService load failed ({type(exc).__name__}: {exc}) — LLM unavailable.")
            self._router = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the model loaded successfully (in-process router) or an Ollama backend is configured."""
        return self._router is not None or getattr(self, "_backend", None) == "ollama"

    def generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Generate a response from the loaded LLM.

        Parameters
        ----------
        prompt : str
            The fully-formatted prompt string.
        **kwargs
            Passed through to ModelRouter.generate (e.g. temperature).

        Returns
        -------
        str or None
            Generated text, or None if the LLM is unavailable.
        """
        if getattr(self, "_backend", None) == "ollama":
            return self._ollama_generate(prompt, **kwargs)
        if not self.is_available:
            return None
        return self._router.generate(prompt, **kwargs)

    def _ollama_generate(self, prompt: str, max_new_tokens: int = 512,
                         system_prompt: Optional[str] = None, do_sample: bool = False,
                         **kwargs) -> Optional[str]:
        """
        POST one chat completion to the Ollama server and return the assistant
        text. Greedy by default (do_sample=False -> temperature 0), matching the
        in-process generate_raw() contract the AIPlanner relies on. action_id_choices
        and min_new_tokens are transformers-only knobs and are ignored here (Ollama
        has no logit-bias equivalent; Gate 1 registry validation remains authoritative).
        """
        import json as _json
        import urllib.request
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        body = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "think": getattr(self, "_ollama_think", False),
            "options": {
                "temperature": 0.7 if do_sample else 0.0,
                "num_predict": max(int(max_new_tokens), getattr(self, "_ollama_num_predict", 2048)),
                "num_ctx": self._ollama_num_ctx,
            },
        }
        data = _json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            self._ollama_url + "/api/chat", data=data,
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self._ollama_timeout) as resp:
                obj = _json.loads(resp.read().decode("utf-8"))
            content = ((obj.get("message") or {}).get("content") or "")
            # Strip any <think>..</think> that leaks into content so the planner's
            # JSON extractor sees clean text (newer Ollama also returns thinking in a
            # separate message.thinking field, which we intentionally ignore here).
            import re as _re
            content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()
            return content
        except Exception as exc:
            # Surface the real reason (connection refused = Ollama unreachable /
            # not bound to 0.0.0.0; timeout = model still loading or too slow).
            print(f"[!] Ollama request to {self._ollama_url} failed: {type(exc).__name__}: {exc}")
            raise

    @property
    def router(self):
        """Direct access to the underlying ModelRouter (for advanced use)."""
        return self._router


def get_llm_service(verbose: bool = True, role: str = "default") -> Optional[LLMService]:
    """
    Return the process-wide LLMService singleton for the given role.
    Returns None if model loading failed.

    Parameters
    ----------
    verbose : bool
        Print load status on first initialization.
    role : str
        "default" for chat / scope-compliance / report synthesis (unchanged
        behavior). "planner" for the autonomous AIPlanner's decide_next() calls
        (M2) -- if core.config.PLANNER_MODEL_NAME is unset, this resolves to the
        SAME cached "default" instance (no extra model load, no extra memory);
        if it is set, a second LLMService is loaded and cached under "planner"
        with that model, independent of (and in addition to) the default one.
    """
    global _INSTANCES, _LOAD_ATTEMPTED

    if role != "default":
        try:
            from core.config import (
                PLANNER_MODEL_NAME, PLANNER_MAX_INPUT_TOKENS, PLANNER_MAX_NEW_TOKENS,
            )
        except ImportError:
            PLANNER_MODEL_NAME = None
            PLANNER_MAX_INPUT_TOKENS = None
            PLANNER_MAX_NEW_TOKENS = None
        if role == "planner" and not PLANNER_MODEL_NAME:
            # No override configured -- fall through to the shared default
            # instance below. This is the path every existing caller takes.
            role = "default"

    if role not in _LOAD_ATTEMPTED or not _LOAD_ATTEMPTED[role]:
        _LOAD_ATTEMPTED[role] = True
        if role == "default":
            instance = LLMService(verbose=verbose)
        else:
            instance = LLMService(
                verbose=verbose,
                model_name_override=PLANNER_MODEL_NAME,
                max_input_tokens_override=PLANNER_MAX_INPUT_TOKENS,
                max_new_tokens_override=PLANNER_MAX_NEW_TOKENS,
            )
        _INSTANCES[role] = instance if instance.is_available else None

    return _INSTANCES.get(role)
