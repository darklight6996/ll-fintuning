"""
core/model_router.py

Purpose:
This file is responsible for loading the language model (LLM) and tokenizer,
handling CPU vs GPU model loading settings, and exposing a single clean
generate() method that the rest of the RAG pipeline can call.

Why this matters:
- Keeps model-loading logic separate from retrieval logic.
- Lets CPU and GPU pipelines use the SAME interface.
- Makes it easy later to swap models without rewriting the whole pipeline.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from typing import Optional
import os
import torch

_GLOBAL_ROUTER: Optional["ModelRouter"] = None


class ModelRouter:
    """
    ModelRouter is the abstraction layer between your pipeline and the actual LLM.

    Instead of your CPU/GPU scripts directly doing:
        tokenizer = AutoTokenizer.from_pretrained(...)
        model = AutoModelForSeq2SeqLM.from_pretrained(...)
        ...
        outputs = model.generate(...)

    they will call ModelRouter, and ModelRouter handles all of that.

    Main responsibilities:
    1. Load the correct tokenizer
    2. Load the correct model (with 4-bit quantization on GPU where appropriate)
    3. Attach PEFT LoRA adapters from Hugging Face or local path via PeftModel.from_pretrained()
    4. Put the model on the right device (CPU or GPU)
    5. Expose a clean generate(prompt) function
    """

    def __init__(
        self,
        model_name: str,
        mode: str = "cpu",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 256,
        adapter_path: Optional[str] = None,
        adapter_repo: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_name : str
            Hugging Face model name or local model path.
        mode : str
            Either "cpu" or "gpu".
        max_input_tokens : int
            Maximum number of input tokens the prompt can use.
        max_new_tokens : int
            Maximum number of tokens the model is allowed to generate.
        adapter_path : Optional[str]
            Hugging Face repo ID or local path to LoRA adapter weights.
        """
        self.model_name = model_name
        self.mode = mode.lower()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens
        self.adapter_path = adapter_path or adapter_repo

        self.tokenizer = None
        self.model = None
        self.device = "cpu"

        self._load_model()

    # ============================================================
    # INTERNAL MODEL LOADER
    # ============================================================

    def _load_model(self):
        """
        Loads tokenizer + model depending on CPU or GPU mode,
        with seamless support for loading PEFT LoRA adapters (TinyLlama/Qwen)
        via PeftModel.from_pretrained() and 4-bit quantization on GPU for 7B models.
        """
        is_peft = False
        base_model_name = self.model_name
        adapter_to_load = self.adapter_path

        # 1. Check if model_name itself is a PEFT adapter
        try:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(self.model_name)
            is_peft = True
            base_model_name = peft_config.base_model_name_or_path or base_model_name
            adapter_to_load = self.model_name
            print(f"[*] Detected PEFT LoRA adapter: {self.model_name} (Base: {base_model_name})")
        except Exception:
            pass

        # 2. Check if explicit adapter_path was provided
        if not is_peft and adapter_to_load:
            try:
                from peft import PeftConfig
                peft_config = PeftConfig.from_pretrained(adapter_to_load)
                is_peft = True
                if peft_config.base_model_name_or_path:
                    base_model_name = peft_config.base_model_name_or_path
                print(f"[*] Configured PEFT LoRA adapter: {adapter_to_load} (Base: {base_model_name})")
            except Exception:
                is_peft = True
                print(f"[*] Using PEFT LoRA adapter: {adapter_to_load}")

        # 3. Check known HF repositories or local lora-adapters if not yet resolved
        if not is_peft:
            from core.config import LORA_ADAPTER_PATH
            lower_name = self.model_name.lower()
            candidate_hf = None
            if "tinyllama" in lower_name:
                candidate_hf = "DarkLight6996/CyberSec-Model-V2"
            elif "qwen" in lower_name:
                candidate_hf = "DarkLight6996/CyberSec-Qwen-V1"

            candidate_local = None
            if os.path.exists(LORA_ADAPTER_PATH):
                if "qwen" in lower_name and os.path.exists(os.path.join(LORA_ADAPTER_PATH, "qwen-qlora")):
                    candidate_local = os.path.join(LORA_ADAPTER_PATH, "qwen-qlora")
                elif os.path.exists(os.path.join(LORA_ADAPTER_PATH, "adapter_config.json")):
                    candidate_local = LORA_ADAPTER_PATH

            chosen_adapter = candidate_local or candidate_hf
            if chosen_adapter:
                try:
                    from peft import PeftConfig
                    peft_config = PeftConfig.from_pretrained(chosen_adapter)
                    is_peft = True
                    adapter_to_load = chosen_adapter
                    if peft_config.base_model_name_or_path:
                        base_model_name = peft_config.base_model_name_or_path
                    print(f"[*] Auto-discovered fine-tuned LoRA adapter: {chosen_adapter} (Base: {base_model_name})")
                except Exception:
                    pass

        print(f"Loading tokenizer for: {base_model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)

        if self.tokenizer.pad_token is None and self.tokenizer.eos_token is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            print(f"pad token set to eos token for {base_model_name}")

        lower_base = base_model_name.lower()
        if "t5" in lower_base or "flan" in lower_base:
            model_family = "seq2seq"
        else:
            model_family = "causal"

        print(f"Detected model family: {model_family}")

        if self.mode == "cpu":
            self.device = "cpu"
            print("Loading model in CPU mode...")
            if model_family == "seq2seq":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(base_model_name)
            if is_peft and adapter_to_load:
                from peft import PeftModel
                print(f"[*] Attaching PEFT LoRA adapter from {adapter_to_load} via PeftModel.from_pretrained()...")
                self.model = PeftModel.from_pretrained(self.model, adapter_to_load)
            self.model.to(self.device)

        elif self.mode == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested, but CUDA is not available.")

            self.device = "cuda"
            print("Loading model in GPU mode...")

            use_large_model = (
                "7b" in lower_base
                or "qwen" in lower_base
                or "mistral" in lower_base
                or is_peft
            )
            if use_large_model and model_family == "causal":
                try:
                    from core.config import GPU_FORCE_4BIT
                except ImportError:
                    GPU_FORCE_4BIT = False

                def _load_4bit():
                    print("[*] Loading in 4-bit NF4 (bitsandbytes QLoRA)...")
                    from transformers import BitsAndBytesConfig
                    bnb_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                    )
                    return AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        quantization_config=bnb_config,
                        device_map="auto",
                        attn_implementation="sdpa"
                    )

                if GPU_FORCE_4BIT:
                    # Skip the fp16 attempt entirely -- for models this config has
                    # flagged (e.g. the qwen-qlora fine-tune, trained in 4-bit NF4 for
                    # an 8GB card), fp16 is known to OOM, so trying it first just
                    # wastes a doomed allocation and risks leaving CUDA in a dirty
                    # state after the partial OOM. Go straight to 4-bit NF4.
                    self.model = _load_4bit()
                    print("[+] 4-bit NF4 load successful (forced).")
                else:
                    try:
                        print("[*] Loading in float16 with device_map=auto (fast path)...")
                        self.model = AutoModelForCausalLM.from_pretrained(
                            base_model_name,
                            torch_dtype=torch.float16,
                            device_map="auto",
                        )
                        print("[+] float16 load successful.")
                    except Exception as oom_err:
                        print(f"[!] float16 load failed ({oom_err}), falling back to 4-bit NF4...")
                        self.model = _load_4bit()
            else:
                if model_family == "seq2seq":
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16
                    )
                    self.model.to(self.device)
                else:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        base_model_name,
                        torch_dtype=torch.float16,
                        device_map="auto"
                    )

            if is_peft and adapter_to_load:
                from peft import PeftModel
                print(f"[*] Attaching PEFT LoRA adapter from {adapter_to_load} via PeftModel.from_pretrained()...")
                self.model = PeftModel.from_pretrained(self.model, adapter_to_load)

        else:
            raise ValueError("mode must be either 'cpu' or 'gpu'")

        self.model.eval()
        print("Model loaded successfully.")

    # ============================================================
    # PROMPT TOKENIZATION
    # ============================================================

    def _prepare_inputs(self, prompt: str):
        """Tokenize the input prompt into tensors."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        )
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)
        return inputs

    # ============================================================
    # GENERATION
    # ============================================================

    def _format_prompt(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """
        Apply chat template or Alpaca framing for causal instruction models.
        """
        chat_tokens = ("<|im_start|>", "<|start_header_id|>", "[INST]", "<start_of_turn>", "### Instruction:")
        if any(tok in prompt for tok in chat_tokens):
            return prompt

        lower = self.model_name.lower()
        is_instruct = "instruct" in lower or "chat" in lower or "qwen" in lower or "llama" in lower
        has_template = hasattr(self.tokenizer, "chat_template") and self.tokenizer.chat_template is not None

        if is_instruct and has_template:
            sys_msg = (
                system_prompt
                or "You are an expert penetration-testing mentor and cybersecurity assistant."
            )
            messages = [
                {"role": "system", "content": sys_msg},
                {"role": "user", "content": prompt},
            ]
            try:
                return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            except Exception:
                pass

        if "tinyllama" in lower or "alpaca" in lower:
            sys_prefix = f"{system_prompt}\n\n" if system_prompt else ""
            return f"### Instruction:\n{sys_prefix}{prompt}\n\n### Response:\n"

        # BUG FIX (found via live smoke test, 2026-11-11): every branch above only
        # covers models with a chat template, or TinyLlama/Alpaca. Any OTHER model --
        # critically, google/flan-t5-large, the actual default CPU_MODEL_NAME -- fell
        # through to `return prompt`, silently DROPPING system_prompt entirely. That
        # meant the planner's _PLANNER_SYSTEM_PROMPT (which contains the ONLY copy of
        # "you must return exactly one JSON object" plus the full output format and
        # decision-verb documentation) never reached the model at all on flan-t5 --
        # it only ever saw the raw context dump with no task framing. Confirmed as the
        # root cause of a real run that failed 5 consecutive planner JSON-parse
        # attempts (MED-1's abort fired correctly, but the actual generation was never
        # going to produce valid JSON in the first place). T5-style encoder-decoder
        # models have no chat-role concept at all -- they expect the instruction simply
        # prepended as plain text in the single input sequence (this is exactly what
        # FLAN-T5 was instruction-tuned on), so that is the fallback here.
        if system_prompt:
            return f"{system_prompt}\n\n{prompt}"

        return prompt

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        min_new_tokens: Optional[int] = None,
        action_id_choices: Optional[list] = None,
        **kwargs
    ) -> str:
        """
        Generate a response from the loaded model.

        action_id_choices (M1): when the caller is AIPlanner requesting a structured
        proposal, this is the registry's list of valid action_id strings. If provided
        (and non-empty), generation is nudged -- via a best-effort LogitsProcessor,
        never a hard constraint -- toward producing a real action_id in the JSON
        "action_id" field. See core/constrained_decoding.py for the exact scope and
        why this is limited to just that one field. Any other caller (chat, report
        synthesis, scope-compliance checks) simply doesn't pass this and is completely
        unaffected.
        """
        formatted_prompt = self._format_prompt(prompt, system_prompt=system_prompt)
        inputs = self._prepare_inputs(formatted_prompt)
        input_length = inputs["input_ids"].shape[1]

        eos_ids = [self.tokenizer.eos_token_id] if self.tokenizer.eos_token_id else []
        im_end = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        if im_end and im_end not in eos_ids:
            eos_ids.append(im_end)

        # H2 fix: honor a per-call max_new_tokens override. Previously this parameter
        # was accepted by every caller (e.g. AIPlanner requesting 512 tokens for a JSON
        # proposal) but silently dropped here, so generation was always capped at the
        # constructor's self.max_new_tokens — truncating structured JSON mid-object.
        effective_max_new_tokens = max_new_tokens if max_new_tokens is not None else self.max_new_tokens

        gen_kwargs = dict(
            max_new_tokens=effective_max_new_tokens,
            do_sample=do_sample,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=eos_ids if eos_ids else None,
        )
        # BUG FIX (found via live GPU smoke test with Qwen2.5-7B-Instruct +
        # qwen-qlora, 2026-11-11): structured planning calls (AIPlanner via
        # generate_raw) got valid JSON up through the "action_id" field --
        # confirming M1's constrained decoding worked correctly -- but then the
        # model emitted <|im_end|>/eos on the very next (unconstrained) step,
        # stopping before "parameters"/"rationale"/"confidence" were ever
        # written. This is the model choosing to end its turn early (plausibly
        # the qwen-qlora fine-tune's conversational ChatML training data making
        # it eager to treat a short fragment as a complete turn), not a
        # tokenization or masking bug. min_new_tokens is the standard mitigation:
        # it makes eos_token_id ineligible until that many tokens have been
        # generated, forcing the model past a premature stop it would otherwise
        # take. Only applied when the caller passes it (generate_raw's JSON
        # callers do; nothing else is affected).
        if min_new_tokens is not None:
            gen_kwargs["min_new_tokens"] = min_new_tokens
        # H4 fix: temperature/top_p only apply under sampling; recent `transformers`
        # versions warn (and some configs error) if they are passed with do_sample=False.
        # Structured planning calls pass do_sample=False for deterministic, greedy decoding.
        if do_sample:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        if action_id_choices:
            try:
                from core.constrained_decoding import build_action_id_logits_processor
                from transformers import LogitsProcessorList

                processor = build_action_id_logits_processor(
                    tokenizer=self.tokenizer,
                    valid_action_ids=action_id_choices,
                    prompt_length=input_length,
                )
                if processor is not None:
                    gen_kwargs["logits_processor"] = LogitsProcessorList([processor])
            except Exception:
                # M1 is advisory -- never let a constrained-decoding setup failure
                # block ordinary unconstrained generation.
                pass

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        if "t5" in self.model_name.lower() or "flan" in self.model_name.lower():
            decoded = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            generated_tokens = outputs[0][input_length:]
            decoded = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return decoded.strip()

    # ============================================================
    # OPTIONAL DEBUG METHOD
    # ============================================================

    def info(self) -> dict:
        """Return basic information about the loaded model."""
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "device": self.device,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens,
            "adapter_path": self.adapter_path
        }


# ============================================================
# MODULE-LEVEL SINGLETON GETTER
# ============================================================

def get_global_model_router(
    model_name: Optional[str] = None,
    mode: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
    adapter_path: Optional[str] = None,
) -> Optional["ModelRouter"]:
    """
    Return the process-wide ModelRouter singleton.
    Creates it on first call using config defaults; subsequent calls return
    the cached instance regardless of arguments.
    """
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        try:
            from core.config import (
                CPU_MODEL_NAME, GPU_MODEL_NAME,
                CPU_MAX_INPUT_TOKENS, CPU_MAX_NEW_TOKENS,
                GPU_MAX_INPUT_TOKENS, GPU_MAX_NEW_TOKENS,
            )

            if mode is None:
                mode = "gpu" if torch.cuda.is_available() else "cpu"
            if model_name is None:
                model_name = GPU_MODEL_NAME if mode == "gpu" else CPU_MODEL_NAME
            if max_input_tokens is None:
                max_input_tokens = GPU_MAX_INPUT_TOKENS if mode == "gpu" else CPU_MAX_INPUT_TOKENS
            if max_new_tokens is None:
                max_new_tokens = GPU_MAX_NEW_TOKENS if mode == "gpu" else CPU_MAX_NEW_TOKENS

            _GLOBAL_ROUTER = ModelRouter(
                model_name=model_name,
                mode=mode,
                max_input_tokens=max_input_tokens,
                max_new_tokens=max_new_tokens,
                adapter_path=adapter_path
            )
        except Exception as exc:
            print(f"[!] get_global_model_router failed ({type(exc).__name__}: {exc})")
            _GLOBAL_ROUTER = None
    return _GLOBAL_ROUTER
