"""
core/constrained_decoding.py

M1: action_id-constrained decoding for the autonomous AIPlanner.

SCOPE (deliberately limited -- see AUDIT.md "M1"):
This constrains ONLY the "action_id" field's value inside the planner's JSON output
to strings that are prefixes of a real, registered action_id. It does NOT attempt
full JSON-grammar-constrained decoding of the whole proposal object (arbitrary
nested "parameters", correct quoting/braces throughout, etc.) -- that is a much
larger undertaking (effectively building a JSON-schema-aware grammar sampler) that
was judged too large and too risky to build and verify correctly without GPU access
in this environment. action_id was chosen as the single highest-value field to
constrain because it is the one place a hallucinated value is both common (the model
inventing a plausible-sounding but unregistered action name) and silently costly
(Gate 1 registry validation in command_executor.py rejects it, burning a full
autonomous step and feeding into MED-1's consecutive-failure counter) -- every other
field already goes through Pydantic's extra="forbid" typed validation at Gate 1
regardless.

This is advisory/best-effort, not a safety gate: Gate 1 registry validation is
completely unchanged and remains the authoritative check. If this processor fails to
find a clean constraint at some step (tokenizer boundary quirks, an unexpected
template), it simply stops restricting for that step rather than risk masking every
token to -inf (which would break generation outright) or ever being the thing a
request's safety depends on.
"""

from __future__ import annotations

import re
from typing import List, Optional

try:
    import torch
    from transformers import LogitsProcessor
except ImportError:  # pragma: no cover - exercised only in environments without torch
    torch = None
    LogitsProcessor = object  # fallback base so the class below can still be defined


_ACTION_ID_TAIL_RE = re.compile(r'"action_id"\s*:\s*"([^"]*)$')


class ActionIdConstrainedLogitsProcessor(LogitsProcessor):
    """
    While the model is generating the value of the JSON "action_id" field, restrict
    the next token to ones consistent with at least one real, registered action_id.
    Outside that field, this processor is a no-op (scores pass through unchanged).
    """

    def __init__(self, tokenizer, valid_action_ids: List[str], prompt_length: int):
        if torch is None:
            raise RuntimeError(
                "ActionIdConstrainedLogitsProcessor requires torch + transformers, "
                "neither of which is available in this environment."
            )
        self.tokenizer = tokenizer
        self.valid_action_ids = list(valid_action_ids)
        self.prompt_length = prompt_length
        # Cache: closing-quote token id(s), used once a candidate is fully matched.
        self._quote_token_ids = self._encode_fragment('"')

    def _encode_fragment(self, text: str) -> List[int]:
        try:
            return self.tokenizer.encode(text, add_special_tokens=False)
        except Exception:
            return []

    def _first_token_for_suffix(self, suffix: str) -> Optional[int]:
        """Best-effort: the first token id produced by encoding `suffix` alone.
        Not guaranteed to byte-for-byte match how `suffix` would tokenize mid-stream
        (BPE merges can shift at a boundary) -- acceptable for an advisory nudge,
        never the sole thing correctness depends on."""
        ids = self._encode_fragment(suffix)
        return ids[0] if ids else None

    def __call__(self, input_ids, scores):
        try:
            generated_ids = input_ids[0][self.prompt_length:]
            if generated_ids.numel() == 0:
                return scores
            decoded = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        except Exception:
            return scores

        match = _ACTION_ID_TAIL_RE.search(decoded)
        if not match:
            return scores  # not currently inside the action_id value -- no-op

        partial = match.group(1)
        candidates = [aid for aid in self.valid_action_ids if aid.startswith(partial)]
        if not candidates:
            # The model has already deviated past anything constrainable (e.g. a typo
            # not matching any known prefix). Don't force-mask everything to -inf --
            # let generation continue naturally; Gate 1 will reject the bad action_id
            # downstream exactly as it always has.
            return scores

        allowed_token_ids = set()
        for candidate in candidates:
            suffix = candidate[len(partial):]
            if suffix == "":
                allowed_token_ids.update(self._quote_token_ids)
            else:
                tok = self._first_token_for_suffix(suffix)
                if tok is not None:
                    allowed_token_ids.add(tok)

        if not allowed_token_ids:
            return scores  # defensive: never produce an all -inf row

        mask = torch.full_like(scores, float("-inf"))
        allowed = torch.tensor(sorted(allowed_token_ids), device=scores.device, dtype=torch.long)
        mask[:, allowed] = scores[:, allowed]
        return mask


def build_action_id_logits_processor(tokenizer, valid_action_ids, prompt_length):
    """
    Convenience factory used by ModelRouter.generate(). Returns None (meaning: no
    constraint applied) if torch/transformers are unavailable or valid_action_ids is
    empty, rather than raising -- this feature must never be able to break generation
    for a caller that doesn't need it.
    """
    if torch is None or not valid_action_ids:
        return None
    try:
        return ActionIdConstrainedLogitsProcessor(
            tokenizer=tokenizer,
            valid_action_ids=valid_action_ids,
            prompt_length=prompt_length,
        )
    except Exception:
        return None
