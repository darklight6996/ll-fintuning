"""
tests/test_m1_constrained_decoding.py

Regression tests for M1: ActionIdConstrainedLogitsProcessor must only restrict
generation while positioned inside the JSON "action_id" field's value, must never
produce an all -inf row (which would break generation), and must pass scores through
unchanged whenever it cannot find a clean constraint (no match, or no candidate
action_id). Uses a trivial char-level fake tokenizer (one token per character) so
these tests exercise the real logic without needing real model weights -- only
torch + transformers themselves, which the autonomous framework already depends on.

Skipped automatically wherever torch/transformers are not installed (e.g. a CI
runner or sandbox with no ML stack) rather than failing.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from core.constrained_decoding import ActionIdConstrainedLogitsProcessor


class _CharTokenizer:
    """One token id per distinct character -- deliberately simple so test
    expectations don't depend on any real BPE vocabulary or model."""

    def __init__(self):
        self._char_to_id = {}
        self._id_to_char = {}
        self._next_id = 0

    def _get_id(self, ch: str) -> int:
        if ch not in self._char_to_id:
            self._char_to_id[ch] = self._next_id
            self._id_to_char[self._next_id] = ch
            self._next_id += 1
        return self._char_to_id[ch]

    def encode(self, text, add_special_tokens=False):
        return [self._get_id(ch) for ch in text]

    def decode(self, ids, skip_special_tokens=True):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return "".join(self._id_to_char.get(int(i), "") for i in ids)

    def vocab_size(self) -> int:
        return self._next_id + 16  # pad some extra ids beyond anything encoded


def _make_input_and_scores(tokenizer, generated_text, prompt_length=0):
    ids = tokenizer.encode(generated_text)
    input_ids = torch.tensor([ids], dtype=torch.long)
    scores = torch.randn(1, tokenizer.vocab_size())
    return input_ids, scores


def test_passthrough_when_not_in_action_id_field():
    tok = _CharTokenizer()
    valid_ids = ["port_scan", "port_scan_deep", "dir_brute"]
    input_ids, scores = _make_input_and_scores(tok, '{"decision": "propose_action"')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)
    assert torch.equal(out, scores)


def test_passthrough_when_partial_matches_no_candidate():
    tok = _CharTokenizer()
    valid_ids = ["port_scan", "dir_brute"]
    input_ids, scores = _make_input_and_scores(tok, '{"action_id": "xyz')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)
    assert torch.equal(out, scores)


def test_restricts_to_shared_next_char_among_prefix_matches():
    tok = _CharTokenizer()
    valid_ids = ["port_scan", "port_scan_deep", "dir_brute"]
    # Pre-encode the full alphabet used so token ids are stable before building scores.
    for s in valid_ids + ['"']:
        tok.encode(s)
    input_ids, scores = _make_input_and_scores(tok, '{"action_id": "po')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)

    assert not torch.equal(out, scores)  # a constraint WAS applied
    allowed_mask = out[0] != float("-inf")
    allowed_ids = set(torch.nonzero(allowed_mask).flatten().tolist())
    r_id = tok.encode("r")[0]
    assert allowed_ids == {r_id}
    # The allowed position must keep its ORIGINAL score, not be rewritten.
    assert out[0, r_id].item() == pytest.approx(scores[0, r_id].item())


def test_allows_closing_quote_when_partial_is_exact_match():
    tok = _CharTokenizer()
    valid_ids = ["port_scan", "port_scan_deep", "dir_brute"]
    for s in valid_ids + ['"']:
        tok.encode(s)
    input_ids, scores = _make_input_and_scores(tok, '{"action_id": "port_scan')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)

    allowed_mask = out[0] != float("-inf")
    allowed_ids = set(torch.nonzero(allowed_mask).flatten().tolist())
    quote_id = tok.encode('"')[0]
    underscore_id = tok.encode("_")[0]
    assert allowed_ids == {quote_id, underscore_id}


def test_never_produces_an_all_negative_inf_row():
    """Defensive guarantee: even in a constrained step, at least one token must
    remain generatable -- this processor must never be able to deadlock generation."""
    tok = _CharTokenizer()
    valid_ids = ["port_scan"]
    for s in valid_ids + ['"']:
        tok.encode(s)
    input_ids, scores = _make_input_and_scores(tok, '{"action_id": "port_scan')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)
    assert bool((out[0] != float("-inf")).any())


def test_forces_delimiter_immediately_after_a_closed_valid_action_id():
    """EXTENSION (2026-11-11): once action_id's value is a complete, real,
    registered action_id and has just been closed with its final quote, the very
    next token must be restricted to a plain JSON delimiter (comma/brace/etc.) --
    this is the fix for the model repeatedly appending an annotation right after
    the closing quote (a parenthetical, a "//" comment, and a REGISTERED ACTIONS-
    style "[risk, network=..., state=...]" tag were all observed live)."""
    tok = _CharTokenizer()
    valid_ids = ["port_scan", "port_scan_deep", "dir_brute"]
    for s in valid_ids + ['"', ",", "}", " ", "\n", "["]:
        tok.encode(s)
    input_ids, scores = _make_input_and_scores(tok, '{"action_id": "port_scan"')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)

    assert not torch.equal(out, scores)  # a constraint WAS applied
    allowed_mask = out[0] != float("-inf")
    allowed_ids = set(torch.nonzero(allowed_mask).flatten().tolist())
    bracket_id = tok.encode("[")[0]
    comma_id = tok.encode(",")[0]
    brace_id = tok.encode("}")[0]
    # The annotation-starting token ("[", and by the same logic "(" or "/") must
    # NOT be among the allowed tokens -- that is the entire point of this fix.
    assert bracket_id not in allowed_ids
    assert allowed_ids <= {comma_id, brace_id}
    assert allowed_ids  # never an all -inf row


def test_closed_action_id_that_is_not_a_real_action_is_not_constrained():
    """If the "closed" value doesn't match any real, registered action_id (the
    model typed a plausible but hallucinated name and closed it), this extension
    must NOT kick in -- Gate 1 registry validation downstream is what's supposed
    to catch that, not this advisory processor."""
    tok = _CharTokenizer()
    valid_ids = ["port_scan", "dir_brute"]
    for s in valid_ids + ['"', ",", "}"]:
        tok.encode(s)
    input_ids, scores = _make_input_and_scores(tok, '{"action_id": "totally_made_up"')
    proc = ActionIdConstrainedLogitsProcessor(tok, valid_ids, prompt_length=0)
    out = proc(input_ids, scores)
    assert torch.equal(out, scores)  # passthrough -- no constraint applied
