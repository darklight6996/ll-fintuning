# scripts/verify_tokenizer.py
import sys
from transformers import AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"

print(f"[*] Loading tokenizer for {MODEL_NAME}...")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sample_messages = [
        {"role": "system", "content": "You are a cybersecurity expert."},
        {"role": "user", "content": "where do i start on ubuntu?"}
    ]
    formatted = tokenizer.apply_chat_template(sample_messages, tokenize=False, add_generation_prompt=True)
    print(f"[*] Formatted output:\n{formatted}")

    if "<|im_start|>" in formatted and "<|im_end|>" in formatted and "<|im_start|>assistant" in formatted:
        print("[+] SUCCESS: Qwen ChatML formatting verified!")
        sys.exit(0)
    else:
        print("[!] ERROR: ChatML tokens missing in formatted prompt.")
        sys.exit(1)
except Exception as e:
    print(f"[!] ERROR loading/verifying tokenizer: {type(e).__name__}: {e}")
    sys.exit(1)
