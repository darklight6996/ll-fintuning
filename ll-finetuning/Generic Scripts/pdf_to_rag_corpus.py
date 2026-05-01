#!/usr/bin/env python3
import os, re, sys, fitz, argparse, logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(message)s")

class RAGTextBuilder:
    def __init__(self, chunk_size=600, overlap=90, min_chars=50, extract_mode="auto"):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chars = min_chars
        self.extract_mode = extract_mode  # "text", "blocks", "auto"

    def extract_text(self, pdf_path):
        pages = []
        doc = fitz.open(pdf_path)
        for i in range(len(doc)):
            page = doc[i]
            if self.extract_mode == "auto":
                txt = page.get_text("text").strip()
                if len(txt) < 20:
                    txt = page.get_text("blocks")
                    txt = "\n".join([b[4] for b in txt]).strip() if txt else ""
            elif self.extract_mode == "blocks":
                txt = page.get_text("blocks")
                txt = "\n".join([b[4] for b in txt]).strip() if txt else ""
            else:
                txt = page.get_text("text").strip()
                
            # Clean
            txt = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', txt)
            txt = re.sub(r' +', ' ', txt)
            txt = re.sub(r'\n{3,}', '\n\n', txt).strip()
            
            logging.info(f"Page {i+1:3d} | Extracted {len(txt):5d} chars")
            pages.append({"page": i+1, "text": txt})
        doc.close()
        return pages

    def chunk_content(self, pages):
        chunks = []
        for page in pages:
            text = page["text"]
            if len(text) < self.min_chars:
                continue
                
            # Split into paragraphs, fallback to sentences if no paragraphs
            paras = [p.strip() for p in re.split(r'\n\s*\n', text) if len(p.strip()) > 10]
            if not paras:
                paras = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 10]
                
            current = []
            curr_len = 0
            for para in paras:
                p_len = len(para)
                if curr_len + p_len > self.chunk_size * 3 and current:  # ~3 chars/token heuristic
                    chunks.append({"page": page["page"], "text": "\n\n".join(current)})
                    # Overlap
                    overlap_parts = []
                    ol_len = 0
                    for p in reversed(current):
                        if ol_len + len(p) > self.overlap * 3: break
                        overlap_parts.insert(0, p)
                        ol_len += len(p)
                    current = overlap_parts if overlap_parts else []
                    curr_len = ol_len
                current.append(para)
                curr_len += p_len
            if current:
                chunks.append({"page": page["page"], "text": "\n\n".join(current)})
        return chunks

    def format_chunk(self, chunk, idx, source_name):
        return (
            f"### BEGIN CHUNK ###\n"
            f"# ID: {source_name}_p{chunk['page']}_c{idx:03d}\n"
            f"# SOURCE: {source_name}\n"
            f"# PAGE: {chunk['page']}\n"
            f"# TOKENS: ~{len(chunk['text'])//4}\n"
            f"### CONTENT ###\n"
            f"{chunk['text']}\n"
            f"### END CHUNK ###\n\n"
        )

    def process_pdf(self, pdf_path, out_file):
        pages = self.extract_text(pdf_path)
        if not pages:
            logging.error("❌ No text extracted.")
            return 0
            
        chunks = self.chunk_content(pages)
        if not chunks:
            logging.warning("⚠️ Text extracted but chunking filtered everything out. Lower --min-chars or check paragraph breaks.")
            return 0

        with open(out_file, "w", encoding="utf-8") as f:
            for i, ch in enumerate(chunks):
                f.write(self.format_chunk(ch, i, Path(pdf_path).stem))
                
        logging.info(f"✅ Written {len(chunks)} chunks to {out_file}")
        return len(chunks)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="PDF file path")
    parser.add_argument("-o", "--output", required=True)
    parser.add_argument("--chunk-size", type=int, default=600)
    parser.add_argument("--overlap", type=int, default=90)
    parser.add_argument("--min-chars", type=int, default=50)
    parser.add_argument("--extract-mode", choices=["text", "blocks", "auto"], default="auto")
    args = parser.parse_args()

    builder = RAGTextBuilder(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        min_chars=args.min_chars,
        extract_mode=args.extract_mode
    )
    builder.process_pdf(args.input, args.output)

if __name__ == "__main__":
    main()