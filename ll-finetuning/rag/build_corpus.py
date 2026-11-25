"""
rag/build_corpus.py

Convert the project's JSONL knowledge datasets into a retrievable text corpus
that core/build_index.py can embed into the FAISS doc index.

WHY THIS EXISTS
---------------
build_index.py reads .txt/.md files from CORPUS_DIR (rag/corpus/) -- but all of
the actual technique knowledge lives in JSONL training datasets (htb.jsonl,
portswigger_alpaca.jsonl, owasp_alpaca.jsonl, comprehensive_cybersec_finetune
.jsonl, qwen_chatml.jsonl, ...), which build_index.py does NOT read. Without
this converter, rag/corpus/ is empty and the RAG technique-advisor (wired into
every agent's planner via core/technique_advisor.py) has nothing to retrieve.

This streams each dataset, normalizes every record to plain text regardless of
its schema (alpaca instruction/input/output, ChatML {"text"}, or anything else
-> JSON fallback), splits it into embedding-sized chunks (the MiniLM embedder
truncates at ~256 tokens, so one giant doc would lose almost all its content),
and writes each chunk as a .txt doc under CORPUS_DIR. It writes to the exact
CORPUS_DIR that config.py resolves, so build_index.py reads precisely what this
writes -- no path drift across the rag symlink/junction.

USAGE
-----
    python3 rag/build_corpus.py                 # sensible defaults
    python3 rag/build_corpus.py --max-records-per-file 1500 --chunk-chars 1200
    python3 rag/build_corpus.py --data-dir /path/to/extra/jsonl

Then build the index:
    python3 rag/core/build_index.py
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

# --- path bootstrap --------------------------------------------------------
# rag/core/__init__.py does `from rag.core.config import *`, which requires the
# REPO ROOT (the directory that contains the `rag` package) to be importable --
# NOT the rag/ directory itself. The framework is normally launched from the
# repo root, so this just happened to work; running `python3 rag/build_corpus.py`
# from inside rag/ broke it with `ModuleNotFoundError: No module named 'rag'`.
# Add both the rag-parent and its parent (the repo's double-nesting: an outer
# repo root holding `core/` + the `rag` junction, and an inner ll-finetuning/
# holding the physical rag/) so `import core` and `import rag` both resolve no
# matter which directory this is run from.
_RAG_DIR = os.path.dirname(os.path.realpath(__file__))              # .../rag
for _cand in (os.path.dirname(_RAG_DIR), os.path.dirname(os.path.dirname(_RAG_DIR))):
    if _cand and _cand not in sys.path:
        sys.path.insert(0, _cand)  # outer candidate ends up first -> canonical `core`
from core.config import CORPUS_DIR, ROOT_DIR  # noqa: E402

# Datasets whose names indicate they are NOT technique knowledge (run logs,
# tiny samples) -- skipped so the corpus stays signal, not noise.
_SKIP_SUBSTRINGS = ("trajector",)


def _candidate_data_dirs(extra):
    """Every plausible location the JSONL datasets live, deduped by realpath.
    Handles the repo's double-nesting (repo-root /data and ll-finetuning/data)
    and the rag symlink without hardcoding either."""
    cands = [
        os.path.join(ROOT_DIR, "data"),
        os.path.join(ROOT_DIR, "..", "data"),
        os.path.join(ROOT_DIR, "ll-finetuning", "data"),
    ]
    if extra:
        cands.insert(0, extra)
    seen, out = set(), []
    for c in cands:
        if c and os.path.isdir(c):
            rp = os.path.realpath(c)
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
    return out


def _record_to_text(rec):
    """Normalize one JSONL record of ANY schema to a single text blob."""
    if not isinstance(rec, dict):
        return str(rec).strip()

    # ChatML / raw-text style
    if "text" in rec and isinstance(rec["text"], str):
        t = rec["text"]
        # Strip ChatML control tokens for cleaner embeddings; keep the content.
        for tok in ("<|im_start|>", "<|im_end|>"):
            t = t.replace(tok, "")
        return t.strip()

    # Alpaca instruction/input/output style
    instr = rec.get("instruction") or rec.get("prompt") or rec.get("question")
    out = rec.get("output") or rec.get("completion") or rec.get("answer") or rec.get("response")
    ctx = rec.get("input") or rec.get("context")
    if instr or out:
        parts = []
        if instr:
            parts.append(f"TECHNIQUE / QUESTION:\n{str(instr).strip()}")
        if ctx:
            parts.append(f"CONTEXT:\n{str(ctx).strip()}")
        if out:
            parts.append(f"METHOD / ANSWER:\n{str(out).strip()}")
        meta = rec.get("metadata")
        if isinstance(meta, dict) and meta.get("source_url"):
            parts.append(f"SOURCE: {meta['source_url']}")
        return "\n\n".join(parts).strip()

    # Unknown schema -> readable JSON so no knowledge is silently dropped.
    try:
        return json.dumps(rec, ensure_ascii=False, indent=2)
    except Exception:
        return str(rec).strip()


def _chunk(text, chunk_chars):
    """Split on blank lines, packing paragraphs up to chunk_chars so each doc
    embeds meaningfully instead of being truncated by the embedder."""
    text = (text or "").strip()
    if not text:
        return []
    if len(text) <= chunk_chars:
        return [text]
    chunks, buf = [], ""
    for para in text.split("\n\n"):
        if len(buf) + len(para) + 2 > chunk_chars and buf:
            chunks.append(buf.strip())
            buf = ""
        if len(para) > chunk_chars:
            # A single oversized paragraph: hard-split it.
            for i in range(0, len(para), chunk_chars):
                chunks.append(para[i:i + chunk_chars].strip())
            continue
        buf += ("\n\n" if buf else "") + para
    if buf.strip():
        chunks.append(buf.strip())
    return [c for c in chunks if c]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-records-per-file", type=int, default=3000,
                    help="Cap records read per dataset (htb.jsonl is huge); default 3000.")
    ap.add_argument("--chunk-chars", type=int, default=1200,
                    help="Target characters per corpus doc; default 1200 (~embedder window).")
    ap.add_argument("--data-dir", default=None, help="Extra directory of .jsonl datasets to include.")
    ap.add_argument("--clean", action="store_true", default=True,
                    help="Clear rag/corpus/ before writing (default on).")
    args = ap.parse_args()

    data_dirs = _candidate_data_dirs(args.data_dir)
    if not data_dirs:
        print("[!] No data directory with .jsonl datasets found. Nothing to do.")
        return

    jsonl_files = []
    seen_rp = set()
    for d in data_dirs:
        for name in sorted(os.listdir(d)):
            if not name.endswith(".jsonl"):
                continue
            if any(s in name.lower() for s in _SKIP_SUBSTRINGS):
                continue
            fp = os.path.realpath(os.path.join(d, name))
            if fp not in seen_rp:
                seen_rp.add(fp)
                jsonl_files.append(fp)

    if not jsonl_files:
        print(f"[!] No .jsonl datasets in: {data_dirs}")
        return

    corpus = Path(CORPUS_DIR)
    if args.clean and corpus.exists():
        for p in corpus.rglob("*.txt"):
            try:
                p.unlink()
            except Exception:
                pass
    corpus.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Building RAG corpus from JSONL knowledge datasets")
    print(f"Corpus output: {CORPUS_DIR}")
    print(f"Datasets ({len(jsonl_files)}):")
    for f in jsonl_files:
        print(f"  - {f}")
    print("=" * 60)

    seen_hashes = set()
    total_docs = 0
    for fp in jsonl_files:
        label = os.path.splitext(os.path.basename(fp))[0]
        dataset_dir = corpus / label
        dataset_dir.mkdir(parents=True, exist_ok=True)
        n_rec = n_doc = 0
        with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                if n_rec >= args.max_records_per_file:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                n_rec += 1
                text = _record_to_text(rec)
                for ci, chunk in enumerate(_chunk(text, args.chunk_chars)):
                    h = hashlib.sha1(chunk.encode("utf-8", "ignore")).hexdigest()
                    if h in seen_hashes:
                        continue  # dedup identical chunks across datasets
                    seen_hashes.add(h)
                    (dataset_dir / f"{n_rec:05d}_{ci:02d}.txt").write_text(chunk, encoding="utf-8")
                    n_doc += 1
        total_docs += n_doc
        print(f"  [{label}] {n_rec} record(s) -> {n_doc} corpus doc(s)")

    print("=" * 60)
    print(f"Done. {total_docs} corpus doc(s) written to {CORPUS_DIR}")
    print("Next: python3 rag/core/build_index.py   (builds the FAISS index)")
    print("=" * 60)


if __name__ == "__main__":
    main()
