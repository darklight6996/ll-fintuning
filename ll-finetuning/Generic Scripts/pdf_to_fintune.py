#!/usr/bin/env python3
"""
PDF → Fine-Tuning Corpus (JSONL)
Outputs Stanford Alpaca format with auto-generated instructions.
"""
import os, re, sys, json, hashlib, argparse
import fitz
import tiktoken
from tqdm import tqdm

# Instruction templates based on content heuristics
INSTRUCT_TEMPLATES = {
    "technical": "Explain the technical concept and its practical application in cybersecurity/GRC:",
    "scripting": "Provide a clear explanation of this PowerShell/scripting technique and how it's used in practice:",
    "framework": "Summarize this governance, risk, or compliance framework and its key control objectives:",
    "procedural": "Break down this security procedure step-by-step and explain its purpose:",
    "general": "Provide a concise, accurate technical summary of the following content:"
}

def clean_text(text: str) -> str:
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r' +', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def detect_content_type(text: str) -> str:
    t = text.lower()
    if re.search(r'(powershell|cmdlet|script|ps1|automation|module)', t): return "scripting"
    if re.search(r'(framework|control|audit|compliance|nist|iso|cobit)', t): return "framework"
    if re.search(r'(step|procedure|process|workflow|configure|implement)', t): return "procedural"
    if re.search(r'(firewall|network|encryption|vpn|protocol|architecture)', t): return "technical"
    return "general"

def extract_with_structure(pdf_path: str):
    doc = fitz.open(pdf_path)
    pages = []
    for i in range(len(doc)):
        text = clean_text(doc[i].get_text("text"))
        if text: pages.append({"page": i+1, "text": text})
    doc.close()
    return pages

def chunk_text(text: str, max_tokens: int, overlap: int, tokenizer):
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', text) if p.strip()]
    chunks = []
    current, curr_len = [], 0
    for para in paragraphs:
        pl = len(tokenizer.encode(para))
        if curr_len + pl > max_tokens and current:
            chunks.append("\n\n".join(current))
            if overlap > 0:
                ov, ol = [], 0
                for p in reversed(current):
                    pl2 = len(tokenizer.encode(p))
                    if ol + pl2 > overlap: break
                    ov.insert(0, p); ol += pl2
                current, curr_len = ov, ol
            else: current, curr_len = [], 0
        current.append(para); curr_len += pl
    if current: chunks.append("\n\n".join(current))
    return chunks

def generate_instruction(text: str) -> str:
    style = detect_content_type(text)
    return INSTRUCT_TEMPLATES[style]

def process_pdf(pdf_path: str, out_file: str, max_tokens: int, overlap: int):
    pages = extract_with_structure(pdf_path)
    if not pages: return 0
    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    source = os.path.basename(pdf_path)
    total = 0
    
    with open(out_file, "a", encoding="utf-8") as f:
        for page in pages:
            for chunk in chunk_text(page["text"], max_tokens, overlap, tokenizer):
                instr = generate_instruction(chunk)
                record = {
                    "instruction": instr,
                    "input": "",
                    "output": chunk
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                total += 1
    return total

def main():
    parser = argparse.ArgumentParser(description="PDF → Fine-Tuning JSONL (Alpaca)")
    parser.add_argument("input", nargs="+", help="PDF files or directories")
    parser.add_argument("-o", "--output", required=True, help="Output .jsonl file")
    parser.add_argument("--chunk-size", type=int, default=512, help="Max tokens per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap tokens")
    parser.add_argument("--clean", action="store_true", help="Overwrite existing output")
    args = parser.parse_args()

    if args.clean and os.path.exists(args.output):
        os.remove(args.output)
        
    input_paths = []
    for p in args.input:
        p = os.path.abspath(p)
        if os.path.isdir(p):
            input_paths.extend([os.path.join(r,f) for r,_,fs in os.walk(p) for f in fs if f.endswith('.pdf')])
        elif os.path.isfile(p) and p.endswith('.pdf'):
            input_paths.append(p)
            
    if not input_paths:
        print("❌ No valid PDFs found.")
        sys.exit(1)

    total = 0
    for pdf in tqdm(input_paths, desc="Processing PDFs"):
        total += process_pdf(pdf, args.output, args.chunk_size, args.overlap)
    print(f"✅ Complete: {total} instruction pairs written to {args.output}")

if __name__ == "__main__":
    main()