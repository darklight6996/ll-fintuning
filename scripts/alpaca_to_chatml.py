# scripts/alpaca_to_chatml.py
import json
import argparse
from pathlib import Path
from transformers import AutoTokenizer

def main():
    parser = argparse.ArgumentParser(description="Convert Alpaca JSONL datasets into Qwen ChatML format.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Tokenizer model name or path")
    parser.add_argument("--inputs", nargs="+", default=None, help="List of input Alpaca JSONL files")
    parser.add_argument("--output", type=str, default="data/qwen_chatml.jsonl", help="Path to output ChatML JSONL file")
    args = parser.parse_args()

    # Determine input files
    if args.inputs:
        input_paths = [Path(p) for p in args.inputs]
    else:
        # Default suite of high-quality cybersecurity instruction datasets
        candidates = [
            Path("data/comprehensive_cybersec_finetune.jsonl"),
            Path("ll-finetuning/data/comprehensive_cybersec_finetune.jsonl"),
            Path("ll-finetuning/data/redefining_hacking.jsonl"),
            Path("ll-finetuning/data/structured_data.jsonl"),
            Path("ll-finetuning/data/owasp_alpaca.jsonl"),
            Path("ll-finetuning/data/portswigger_alpaca.jsonl"),
        ]
        input_paths = [p for p in candidates if p.exists()]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[*] Loading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    def alpaca_to_chatml(record):
        # Support various field naming conventions
        instruction = record.get("instruction", "") or record.get("prompt", "") or record.get("question", "")
        input_ctx = record.get("input", "") or record.get("context", "")
        output_txt = record.get("output", "") or record.get("response", "") or record.get("answer", "")

        user_content = instruction.strip()
        if input_ctx and input_ctx.strip():
            user_content += f"\n\nContext:\n{input_ctx.strip()}"

        if not user_content or not output_txt:
            return None

        messages = [
            {"role": "system", "content": "You are a cybersecurity and penetration testing expert inside a live operator console."},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output_txt.strip()}
        ]
        return tokenizer.apply_chat_template(messages, tokenize=False)

    print(f"[*] Converting {len(input_paths)} files into {output_path}:")
    total_count = 0
    seen_hashes = set()

    with output_path.open("w", encoding="utf-8") as fout:
        for in_path in input_paths:
            file_count = 0
            with in_path.open("r", encoding="utf-8") as fin:
                for line_num, line in enumerate(fin, 1):
                    line_str = line.strip()
                    if not line_str:
                        continue
                    try:
                        record = json.loads(line_str)
                        formatted_text = alpaca_to_chatml(record)
                        if not formatted_text:
                            continue
                        
                        # Deduplicate
                        h = hash(formatted_text)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        fout.write(json.dumps({"text": formatted_text}, ensure_ascii=False) + "\n")
                        file_count += 1
                        total_count += 1
                    except Exception as e:
                        pass
            print(f"  - {in_path}: {file_count} records added")

    print(f"[+] Total unique ChatML records written: {total_count} -> {output_path}")

if __name__ == "__main__":
    main()
