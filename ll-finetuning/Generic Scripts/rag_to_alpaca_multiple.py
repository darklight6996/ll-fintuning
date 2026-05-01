#!/usr/bin/env python3
"""
Folder RAG TXT → Alpaca JSONL Converter
Iterates through a directory of RAG .txt files, parses chunks reliably,
and outputs a combined Alpaca-format JSONL file.
"""

import os
import json
import argparse
from pathlib import Path

def parse_rag_file(filepath: Path) -> list:
    """Parses a RAG TXT file using a state-machine approach."""
    chunks = []
    current_chunk = None
    content_lines = []
    in_content = False

    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.rstrip('\n\r')
                
                if "### BEGIN CHUNK ###" in line:
                    current_chunk = {"metadata": {}}
                    content_lines = []
                    in_content = False
                elif "### END CHUNK ###" in line and current_chunk is not None:
                    in_content = False
                    current_chunk["content"] = "\n".join(content_lines).strip()
                    if current_chunk["content"]:
                        chunks.append(current_chunk)
                    current_chunk = None
                elif current_chunk is not None:
                    if line.startswith("# "):
                        parts = line[2:].split(":", 1)
                        if len(parts) == 2:
                            current_chunk["metadata"][parts[0].strip()] = parts[1].strip()
                    elif "### CONTENT ###" in line:
                        in_content = True
                    elif in_content:
                        content_lines.append(line)
    except Exception as e:
        print(f"⚠️ Error reading {filepath.name}: {e}")
        
    return chunks

def generate_alpaca_record(chunk: dict, min_chars: int) -> dict | None:
    """Converts a parsed chunk into Alpaca JSONL format."""
    content = chunk.get("content", "")
    if len(content) < min_chars:
        return None

    meta = chunk.get("metadata", {})
    source = meta.get("SOURCE", meta.get("source", "Unknown Document"))
    section = meta.get("SECTION", meta.get("section", "General"))
    
    # Heuristic instruction generation
    text_lower = content.lower()
    if any(kw in text_lower for kw in ["step", "procedure", "how to", "first", "next"]):
        instr = "Describe the steps or process outlined in the following text:"
    elif any(kw in text_lower for kw in ["command", "code", "function", "script", "api"]):
        instr = "Explain the purpose and usage of the code/command in the following text:"
    elif any(kw in text_lower for kw in ["definition", "concept", "means", "refers to"]):
        instr = "Define and explain the key concept presented in the following text:"
    else:
        instr = "Provide a clear, comprehensive explanation of the following content:"

    return {
        "instruction": f"{instr} (Context: {section} from {source})",
        "input": f"Source: {source}\nSection: {section}",
        "output": content
    }

def main():
    parser = argparse.ArgumentParser(description="Convert folder of RAG TXT files to Alpaca JSONL")
    parser.add_argument("input_dir", help="Directory containing RAG .txt files")
    parser.add_argument("-o", "--output", required=True, help="Output JSONL file")
    parser.add_argument("--min-chars", type=int, default=50, help="Minimum content length to keep")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"❌ Error: '{input_dir}' is not a valid directory.")
        return

    txt_files = sorted(input_dir.glob("*.txt"))
    if not txt_files:
        print("⚠️ No .txt files found in the directory.")
        return

    print(f"📂 Found {len(txt_files)} RAG text files. Processing...")
    total_records = 0

    with open(args.output, "w", encoding="utf-8") as out_f:
        for txt_file in txt_files:
            print(f"📄 Processing: {txt_file.name}")
            chunks = parse_rag_file(txt_file)
            
            for chunk in chunks:
                record = generate_alpaca_record(chunk, args.min_chars)
                if record:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    total_records += 1

    print(f"✅ Successfully converted {total_records} chunks → {args.output}")

if __name__ == "__main__":
    main()