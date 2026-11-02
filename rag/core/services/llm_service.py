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

from typing import Optional, Any

_INSTANCE: Optional["LLMService"] = None
_LOAD_ATTEMPTED: bool = False


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

    def __init__(self, verbose: bool = True):
        self._router = None
        self.mode: str = "cpu"
        self.model_name: str = ""
        self.tokenizer: Optional[Any] = None
        self._load(verbose=verbose)

    def _load(self, verbose: bool = True) -> None:
        try:
            import torch
            from core.model_router import ModelRouter
            from core.config import (
                CPU_MODEL_NAME, GPU_MODEL_NAME,
                CPU_MAX_INPUT_TOKENS, CPU_MAX_NEW_TOKENS,
                GPU_MAX_INPUT_TOKENS, GPU_MAX_NEW_TOKENS,
            )

            if torch.cuda.is_available():
                self.mode = "gpu"
                self.model_name = GPU_MODEL_NAME
                max_in, max_new = GPU_MAX_INPUT_TOKENS, GPU_MAX_NEW_TOKENS
            else:
                self.mode = "cpu"
                self.model_name = CPU_MODEL_NAME
                max_in, max_new = CPU_MAX_INPUT_TOKENS, CPU_MAX_NEW_TOKENS

            self._router = ModelRouter(
                model_name=self.model_name,
                mode=self.mode,
                max_input_tokens=max_in,
                max_new_tokens=max_new,
            )
            self.tokenizer = getattr(self._router, "tokenizer", None)

            if verbose:
                print(f"[+] LLMService ready ({self.mode}): {self.model_name}")

        except Exception as exc:
            print(f"[!] LLMService load failed ({type(exc).__name__}: {exc}) — LLM unavailable.")
            self._router = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if the model loaded successfully."""
        return self._router is not None

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
        if not self.is_available:
            return None
        return self._router.generate(prompt, **kwargs)

    @property
    def router(self):
        """Direct access to the underlying ModelRouter (for advanced use)."""
        return self._router


def get_llm_service(verbose: bool = True) -> Optional[LLMService]:
    """
    Return the process-wide LLMService singleton.
    Returns None if model loading failed.

    Parameters
    ----------
    verbose : bool
        Print load status on first initialization.
    """
    global _INSTANCE, _LOAD_ATTEMPTED
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        _INSTANCE = LLMService(verbose=verbose)
        if not _INSTANCE.is_available:
            _INSTANCE = None
    return _INSTANCE
