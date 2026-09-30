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
    2. Load the correct model
    3. Put the model on the right device (CPU or GPU)
    4. Expose a clean generate(prompt) function
    """

    def __init__(
        self,
        model_name: str,
        mode: str = "cpu",
        max_input_tokens: int = 2048,
        max_new_tokens: int = 256
    ):
        """
        Parameters
        ----------
        model_name : str
            Hugging Face model name or local model path.

            Example CPU model:
                "google/flan-t5-base"

            Example GPU model:
                "mistralai/Mistral-7B-Instruct-v0.2"

        mode : str
            Either "cpu" or "gpu".
            This controls how the model is loaded.

        max_input_tokens : int
            Maximum number of input tokens the prompt can use.
            If prompt is too long, tokenizer will truncate.

        max_new_tokens : int
            Maximum number of tokens the model is allowed to generate.
        """

        self.model_name = model_name
        self.mode = mode.lower()
        self.max_input_tokens = max_input_tokens
        self.max_new_tokens = max_new_tokens

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
        with seamless support for PEFT LoRA adapters (TinyLlama/Qwen)
        and 4-bit quantization on GPU for 7B models.
        """
        is_peft = False
        base_model_name = self.model_name
        adapter_path = None

        try:
            from peft import PeftConfig
            peft_config = PeftConfig.from_pretrained(self.model_name)
            is_peft = True
            base_model_name = peft_config.base_model_name_or_path
            adapter_path = self.model_name
            print(f"[*] Detected PEFT LoRA adapter: {self.model_name} (Base: {base_model_name})")
        except Exception:
            is_peft = False

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
            if is_peft:
                from peft import PeftModel
                self.model = PeftModel.from_pretrained(self.model, adapter_path)
            self.model.to(self.device)

        elif self.mode == "gpu":
            if not torch.cuda.is_available():
                raise RuntimeError("GPU mode requested, but CUDA is not available.")

            self.device = "cuda"
            print("Loading model in GPU mode...")

            # Broaden 4-bit detection: any large causal model benefits from 4-bit quantization
            # (7B+, Qwen, Mistral, or any PEFT/fine-tuned derivative like CyberSec-Model-V2)
            use_4bit = (
                "7b" in lower_base
                or "qwen" in lower_base
                or "mistral" in lower_base
                or is_peft  # LoRA adapters on 7B+ base models always use 4-bit
            )
            if use_4bit and model_family == "causal":
                from transformers import BitsAndBytesConfig
                print("[*] Applying 4-bit NF4 quantization for GPU loading...")
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    attn_implementation="sdpa"
                )
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

            if is_peft:
                from peft import PeftModel
                print(f"[*] Attaching PEFT LoRA adapter from {adapter_path}...")
                self.model = PeftModel.from_pretrained(self.model, adapter_path)

        else:
            raise ValueError("mode must be either 'cpu' or 'gpu'")

        self.model.eval()
        print("Model loaded successfully.")

    # ============================================================
    # PROMPT TOKENIZATION
    # ============================================================

    def _prepare_inputs(self, prompt: str):
        """
        Tokenize the input prompt into tensors.

        Why this is separated into its own function:
        - keeps generate() cleaner
        - easier to debug prompt/tokenization issues
        - later you can add logging, token counts, prompt inspection, etc.

        Returns
        -------
        dict
            Tokenized tensors ready to be passed into model.generate()
        """

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_tokens
        )

        # Move tokenized tensors to correct device if needed
        for key in inputs:
            inputs[key] = inputs[key].to(self.device)

        return inputs

    # ============================================================
    # GENERATION
    # ============================================================

    def generate(
        self,
        prompt: str,
        temperature: float = 0.2,
        top_p: float = 0.9,
        do_sample: bool = True
    ) -> str:
        """
        Generate a response from the loaded model.
        """

        inputs = self._prepare_inputs(prompt)
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id
            )

        # For causal models, decode only the newly generated tokens
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
        """
        Return basic information about the loaded model.
        Useful for debugging and logging.
        """
        return {
            "model_name": self.model_name,
            "mode": self.mode,
            "device": self.device,
            "max_input_tokens": self.max_input_tokens,
            "max_new_tokens": self.max_new_tokens
        }


# ============================================================
# MODULE-LEVEL SINGLETON GETTER
# ============================================================

def get_global_model_router(
    model_name: Optional[str] = None,
    mode: Optional[str] = None,
    max_input_tokens: Optional[int] = None,
    max_new_tokens: Optional[int] = None,
) -> Optional["ModelRouter"]:
    """
    Return the process-wide ModelRouter singleton.
    Creates it on first call using config defaults; subsequent calls return
    the cached instance regardless of arguments.

    Returns None if model loading fails (CUDA unavailable, weights missing, etc.).
    """
    global _GLOBAL_ROUTER
    if _GLOBAL_ROUTER is None:
        try:
            import torch
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
            )
        except Exception as exc:
            print(f"[!] get_global_model_router failed ({type(exc).__name__}: {exc})")
            _GLOBAL_ROUTER = None  # type: ignore[assignment]
    return _GLOBAL_ROUTER