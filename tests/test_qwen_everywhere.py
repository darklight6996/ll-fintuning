"""
tests/test_qwen_everywhere.py

Regression coverage for switching GPU_MODEL_NAME to the user's fine-tuned
Qwen2.5-7B-Instruct + qwen-qlora adapter "everywhere" (chat, report synthesis,
scope-compliance, and the planner -- since PLANNER_MODEL_NAME is left unset and
therefore resolves to this same default), and for GPU_FORCE_4BIT making
ModelRouter skip the doomed fp16 attempt on an 8GB card and go straight to 4-bit
NF4, matching how the adapter was actually trained (train_qlora.py, bitsandbytes
NF4 QLoRA, targeted at an RTX 3060 Ti 8GB).
"""
import os
import types
import pytest

# No manual sys.path manipulation here: pytest.ini's `pythonpath = .` plus
# core/__init__.py's own __path__ extension (which merges repo-root core/
# with ll-finetuning/rag/core, via a `rag` symlink/junction at the repo root)
# is what makes `from core...` imports resolve in every other test file --
# inserting a second, competing sys.path entry pointing directly at
# ll-finetuning/rag only risks shadowing the real resolution path and getting
# a second, separately-cached copy of the `core` package instead of the one
# everything else in the suite actually uses.


def test_gpu_model_name_is_qwen():
    from core import config
    assert config.GPU_MODEL_NAME == "Qwen/Qwen2.5-7B-Instruct"


def test_gpu_force_4bit_enabled_by_default():
    from core import config
    assert config.GPU_FORCE_4BIT is True


def test_planner_model_name_unset_falls_through_to_qwen_default():
    # Per llm_service.get_llm_service(): when PLANNER_MODEL_NAME is unset, role
    # "planner" resolves to "default", which loads GPU_MODEL_NAME on CUDA --
    # i.e. the planner automatically uses the same Qwen+LoRA model as everything
    # else, with zero extra code, as long as this env var stays unset.
    assert os.getenv("PLANNER_MODEL_NAME") in (None, "")
    from core import config
    assert config.PLANNER_MODEL_NAME is None


def test_local_qwen_qlora_adapter_is_discoverable():
    # Regression guard for the realpath fix in core/config.py: ROOT_DIR must be
    # computed from config.py's REAL (symlink-resolved) physical location, or
    # LORA_ADAPTER_PATH silently overshoots past the repo root and
    # ModelRouter's LoRA auto-discovery (step 3 in _load_model) never fires --
    # which is exactly what happened before that fix (confirmed via a live
    # pytest run: os.path.isdir() was False here on every attempt).
    from core.config import LORA_ADAPTER_PATH, ROOT_DIR
    assert os.path.isdir(ROOT_DIR), f"ROOT_DIR does not exist: {ROOT_DIR!r}"
    qwen_adapter_dir = os.path.join(LORA_ADAPTER_PATH, "qwen-qlora")
    assert os.path.isdir(qwen_adapter_dir), (
        f"expected local adapter at {qwen_adapter_dir!r} (LORA_ADAPTER_PATH="
        f"{LORA_ADAPTER_PATH!r}, ROOT_DIR={ROOT_DIR!r}); ModelRouter's "
        f"auto-discovery (step 3 in _load_model) depends on this existing"
    )
    assert os.path.isfile(os.path.join(qwen_adapter_dir, "adapter_config.json"))


def test_force_4bit_skips_fp16_attempt(monkeypatch):
    """
    With GPU_FORCE_4BIT=True and a "qwen"/7b-matching model name, ModelRouter's
    GPU branch must go straight to the 4-bit NF4 loader and never attempt the
    fp16 AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.float16)
    fast path at all.
    """
    import torch
    from core import model_router as model_router_module

    calls = {"fp16": 0, "4bit": 0}

    class _FakeModel:
        def eval(self):
            return self

        def to(self, device):
            return self

    def _fake_from_pretrained(base_model_name, **kwargs):
        if kwargs.get("quantization_config") is not None:
            calls["4bit"] += 1
        elif kwargs.get("torch_dtype") is not None:
            calls["fp16"] += 1
        return _FakeModel()

    monkeypatch.setattr(model_router_module, "AutoModelForCausalLM",
                         types.SimpleNamespace(from_pretrained=_fake_from_pretrained))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: False)

    # Force "no PEFT adapter resolved" at the source (PeftConfig.from_pretrained
    # itself) rather than relying on real filesystem/network lookups to fail --
    # on a machine where lora-adapters/qwen-qlora genuinely exists (as it does
    # here), auto-discovery succeeds and the real _load_model() goes on to call
    # PeftModel.from_pretrained(self.model, ...), which needs a real nn.Module
    # (named_modules(), etc.) that _FakeModel deliberately does not implement --
    # this test is only about the fp16-vs-4bit base-model branch, not PEFT
    # attachment, so PEFT discovery is cut off up front instead.
    from peft import PeftConfig
    monkeypatch.setattr(PeftConfig, "from_pretrained",
                         staticmethod(lambda *a, **k: (_ for _ in ()).throw(Exception("no adapter"))))

    class _FakeTokenizer:
        pad_token = "<eos>"
        eos_token = "<eos>"
        chat_template = None

        def __call__(self, *a, **k):
            raise NotImplementedError

    monkeypatch.setattr(
        model_router_module.AutoTokenizer, "from_pretrained",
        lambda *a, **k: _FakeTokenizer()
    )

    # Force "no PEFT adapter resolved" so we exercise only the base-model GPU
    # loading branch in isolation.
    router = object.__new__(model_router_module.ModelRouter)
    router.model_name = "Qwen/Qwen2.5-7B-Instruct"
    router.mode = "gpu"
    router.max_input_tokens = 4096
    router.max_new_tokens = 1024
    router.adapter_path = None
    router.tokenizer = None
    router.model = None
    router.device = "cpu"

    # Run just the GPU-loading portion of _load_model by calling the real
    # method -- PEFT auto-discovery will try real HF/local lookups and fail
    # harmlessly (network blocked / not a real adapter), landing in the same
    # base-model-only path.
    router._load_model()

    assert calls["4bit"] == 1, "expected exactly one 4-bit NF4 load attempt"
    assert calls["fp16"] == 0, "fp16 fast path must be skipped when GPU_FORCE_4BIT=True"
