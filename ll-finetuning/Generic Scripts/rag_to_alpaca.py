#!/usr/bin/env python3
"""
RAG TXT → Alpaca JSONL Converter
Converts structured RAG text files into Alpaca instruction-tuning JSONL format.
Designed to be generic, memory-efficient, and robust for any txt file following the delimiter pattern.
"""

import json
import argparse
import logging
from pathlib import Path
from typing import Generator, Dict, List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

class AlpacaConverter:
    def __init__(self, min_len: int = 50, max_len: int = 4000, add_meta: bool = False):
        self.min_len = min_len
        self.max_len = max_len
        self.add_meta = add_meta

    def parse_txt(self, filepath: Path) -> Generator[Dict, None, None]:
        """State-machine parser that reads line-by-line for memory efficiency."""
        logger.info(f"Parsing: {filepath}")
        
        in_chunk = False
        meta_lines = []
        content_lines = []

        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if "### BEGIN CHUNK ###" in line:
                    in_chunk = True
                    meta_lines, content_lines = [], []
                    continue
                elif "### END CHUNK ###" in line:
                    if in_chunk and content_lines:
                        yield {
                            "metadata": {k: v for k, v in (m.split(":", 1) for m in meta_lines if ":" in m)},
                            "content": "\n".join(content_lines).strip()
                        }
                    in_chunk = False
                    meta_lines, content_lines = [], []
                    continue
                
                if in_chunk:
                    if line.startswith("# "):
                        meta_lines.append(line)
                    elif line == "### CONTENT ###":
                        continue
                    else:
                        content_lines.append(line)

    @staticmethod
    def _generate_instruction(content: str, metadata: Dict) -> str:
        """Context-aware instruction generator."""
        text_lower = content.lower()
        
        # Heuristic classification
        if any(kw in text_lower for kw in ['step', 'procedure', 'how to', 'first', 'next', 'then']):
            base = "Describe the steps or process outlined in the following text:"
        elif any(kw in text_lower for kw in ['definition', 'is defined as', 'refers to', 'concept', 'means']):
            base = "Define and explain the key concept presented in the following text:"
        elif any(kw in text_lower for kw in ['command', 'code', 'function', 'script', 'syntax', 'parameter']):
            base = "Explain the purpose, syntax, and usage of the code/command in the following text:"
        elif any(kw in text_lower for kw in ['risk', 'threat', 'vulnerability', 'attack', 'mitigation', 'control']):
            base = "Analyze the security context, risks, and mitigations described in the following text:"
        else:
            base = "Provide a clear, comprehensive explanation of the following content:"

        # Append contextual hint
        section = metadata.get("SECTION", "").strip()
        source = Path(metadata.get("SOURCE", "Document")).stem.strip()
        context = " ".join(filter(None, [section, f"from {source}"]))
        if context:
            return f"{base} (Context: {context})"
        return base

    def convert_to_alpaca(self, chunks: Generator) -> Generator[Dict, None, None]:
        """Stream chunks into Alpaca format."""
        for chunk in chunks:
            content = chunk["content"]
            if not content or len(content) < self.min_len or len(content) > self.max_len:
                continue

            record = {
                "instruction": self._generate_instruction(content, chunk["metadata"]),
                "input": (
                    f"Source: {chunk['metadata'].get('SOURCE', 'Unknown')}\n"
                    f"Section: {chunk['metadata'].get('SECTION', 'General')}\n"
                    f"Pages: {chunk['metadata'].get('PAGES', 'N/A')}"
                ),
                "output": content
            }
            if self.add_meta:
                record["_meta"] = chunk["metadata"]
            yield record

    def run(self, input_txt: Path, output_jsonl: Path):
        """Main conversion pipeline."""
        chunks = self.parse_txt(input_txt)
        count = 0
        
        with open(output_jsonl, "w", encoding="utf-8") as f:
            for record in self.convert_to_alpaca(chunks):
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                count += 1
                
        logger.info(f"✅ Successfully converted {count} chunks → {output_jsonl}")

def main():
    parser = argparse.ArgumentParser(description="Convert RAG TXT to Alpaca JSONL")
    parser.add_argument("input", type=Path, help="Input .txt file path")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output .jsonl file path")
    parser.add_argument("--min-len", type=int, default=50, help="Minimum content length to keep")
    parser.add_argument("--max-len", type=int, default=4000, help="Maximum content length to keep")
    parser.add_argument("--add-meta", action="store_true", help="Include raw metadata in output JSONL")
    args = parser.parse_args()

    if not args.input.exists():
        logger.error(f"❌ Input file not found: {args.input}")
        return

    converter = AlpacaConverter(
        min_len=args.min_len,
        max_len=args.max_len,
        add_meta=args.add_meta
    )
    converter.run(args.input, args.output)

if __name__ == "__main__":
    main()