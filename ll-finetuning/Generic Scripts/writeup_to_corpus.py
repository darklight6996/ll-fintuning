#!/usr/bin/env python3
import os
import sys
import argparse
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(description="Convert HTB/Pentest writeups to AI Corpus")
    parser.add_argument("input_dir", type=str, help="Directory containing .txt writeup files")
    parser.add_argument("--rag-dir", type=str, required=True, help="Output directory for RAG chunks")
    parser.add_argument("--chunk-size", type=int, default=512, help="Target words per chunk")
    parser.add_argument("--overlap", type=int, default=64, help="Overlap words between chunks")
    return parser.parse_args()

def chunk_text(text, chunk_size, overlap):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end >= len(words):
            break
        start = end - overlap
    return chunks

def process_file(file_path, output_dir, chunk_size, overlap):
    logging.info(f"📄 Processing: {file_path.name}")
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read().strip()
    except Exception as e:
        logging.error(f"❌ Failed to read {file_path.name}: {e}")
        return 0

    if not content:
        logging.warning(f"⚠️ Empty or whitespace-only file: {file_path.name}")
        return 0

    title = file_path.stem.replace('_', ' ').title()
    chunks = chunk_text(content, chunk_size, overlap)
    
    if not chunks:
        return 0

    out_file = Path(output_dir) / f"{file_path.stem}_rag.txt"
    count = 0
    with open(out_file, 'w', encoding='utf-8') as f:
        for i, chunk in enumerate(chunks):
            f.write(f"### BEGIN CHUNK ###\n")
            f.write(f"# ID: {file_path.stem}_{i:03d}\n")
            f.write(f"# SOURCE: {title}\n")
            f.write(f"# SECTION: Writeup\n")
            f.write(f"### CONTENT ###\n{chunk}\n")
            f.write(f"### END CHUNK ###\n\n")
            count += 1

    logging.info(f"  ✅ Wrote {count} chunks → {out_file.name}")
    return count

def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.rag_dir)

    if not input_dir.is_dir():
        logging.error(f"❌ Input directory does not exist: {input_dir.absolute()}")
        sys.exit(1)

    txt_files = list(input_dir.glob("*.txt"))
    if not txt_files:
        logging.warning(f"⚠️ No .txt files found in {input_dir.absolute()}")
        logging.info("💡 Tip: Ensure your scraped writeups are saved with a .txt extension.")
        sys.exit(0)

    logging.info(f"📂 Found {len(txt_files)} .txt files to process.")
    output_dir.mkdir(parents=True, exist_ok=True)

    total_chunks = 0
    for txt_file in txt_files:
        total_chunks += process_file(txt_file, output_dir, args.chunk_size, args.overlap)

    logging.info(f"🎉 Complete! Total chunks created: {total_chunks}")
    logging.info(f"📁 Output directory: {output_dir.absolute()}")

if __name__ == "__main__":
    main()