"""
core/prompts/templates.py

Multi-model token framing and chat template formatting.
Supports:
- Qwen (ChatML tokens: <|im_start|> / <|im_end|>)
- Llama 3 (<|start_header_id|> / <|end_header_id|>)
- Mistral ([INST] / [/INST])
- Gemma (<start_of_turn> / <end_of_turn>)
- DeepSeek (<｜begin of sentence｜> / User / Assistant)
- Standard / Raw fallback
"""

from enum import Enum
from typing import Union, Optional, Any, Dict, List


class ModelFamily(str, Enum):
    QWEN = "qwen"
    LLAMA = "llama"
    MISTRAL = "mistral"
    GEMMA = "gemma"
    DEEPSEEK = "deepseek"
    STANDARD = "standard"


def detect_model_family(model_name: Optional[str]) -> ModelFamily:
    """
    Detect the target ModelFamily from model identifier or path.

    Recognised patterns:
    - Qwen family: qwen
    - Llama family: llama, tinyllama, alpaca, and DarkLight6996 custom adapters
      (CyberSec-Model-V2 is TinyLlama-based; CyberSec-Qwen-V1 is Qwen-based)
    - Mistral family: mistral, zephyr, mixtral
    - Gemma family: gemma
    - DeepSeek family: deepseek
    - Standard / raw: everything else
    """
    if not model_name:
        return ModelFamily.QWEN  # default primary model family

    m = model_name.lower()

    # Known DarkLight6996 custom adapter families
    # CyberSec-Qwen-V1  → Qwen base
    # CyberSec-Model-V2 → TinyLlama base
    if "cybersec-qwen" in m:
        return ModelFamily.QWEN
    if "cybersec-model" in m or ("darklight" in m and "qwen" not in m):
        return ModelFamily.LLAMA

    if "qwen" in m:
        return ModelFamily.QWEN
    elif "llama" in m or "tinyllama" in m or "alpaca" in m:
        return ModelFamily.LLAMA
    elif "mistral" in m or "zephyr" in m or "mixtral" in m:
        return ModelFamily.MISTRAL
    elif "gemma" in m:
        return ModelFamily.GEMMA
    elif "deepseek" in m:
        return ModelFamily.DEEPSEEK
    else:
        return ModelFamily.STANDARD


class PromptTemplate:
    """
    Applies exact token framing for heterogeneous model architectures.
    """

    @staticmethod
    def format(
        system_prompt: str,
        user_content: str,
        model_family: Union[str, ModelFamily] = ModelFamily.QWEN,
        tokenizer: Optional[Any] = None
    ) -> str:
        """
        Format prompt using model-specific framing tokens or tokenizer.apply_chat_template.
        """
        # 1. Native Tokenizer Chat Template override
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template") and getattr(tokenizer, "chat_template", None):
            try:
                messages = [
                    {"role": "system", "content": system_prompt.strip()},
                    {"role": "user", "content": user_content.strip()}
                ]
                formatted = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                return formatted
            except Exception:
                pass  # Fall through to explicit template framing

        # Resolve enum
        if isinstance(model_family, str):
            try:
                fam = ModelFamily(model_family.lower())
            except ValueError:
                fam = detect_model_family(model_family)
        else:
            fam = model_family

        sys_str = system_prompt.strip()
        usr_str = user_content.strip()

        # 2. Qwen (ChatML Framing)
        if fam == ModelFamily.QWEN:
            return (
                f"<|im_start|>system\n{sys_str}<|im_end|>\n"
                f"<|im_start|>user\n{usr_str}<|im_end|>\n"
                f"<|im_start|>assistant\n"
            )

        # 3. Llama 3 Framing
        elif fam == ModelFamily.LLAMA:
            return (
                f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                f"{sys_str}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
                f"{usr_str}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            )

        # 4. Mistral / Zephyr Framing
        elif fam == ModelFamily.MISTRAL:
            return (
                f"<s>[INST] {sys_str}\n\n{usr_str} [/INST]"
            )

        # 5. Gemma Framing
        elif fam == ModelFamily.GEMMA:
            return (
                f"<start_of_turn>user\n{sys_str}\n\n{usr_str}<end_of_turn>\n"
                f"<start_of_turn>model\n"
            )

        # 6. DeepSeek Framing
        elif fam == ModelFamily.DEEPSEEK:
            return (
                f"<｜begin of sentence｜>{sys_str}\n\n"
                f"User: {usr_str}\n\n"
                f"Assistant: <｜thought｜>\n"
            )

        # 7. Standard / Raw Text Fallback
        else:
            return f"{sys_str}\n\n{usr_str}\n\nAssistant:\n"
