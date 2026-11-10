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
