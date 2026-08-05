"""
core/yaml_indexer.py

Purpose
-------
Build a semantic FAISS index from every YAML assessment template.

Responsibilities
----------------
1. Load all YAML templates
2. Convert each template into searchable text
3. Generate embeddings
4. Build a FAISS index
5. Save the index and metadata
"""

import os
import sys
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Resolve system paths
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.yaml_loader import YAMLLoader
from core.config import SCENARIOS_DIR, YAML_INDEX_DIR, EMBED_MODEL


class YAMLIndexer:

    def __init__(
        self,
        yaml_root=SCENARIOS_DIR,
        output_dir=YAML_INDEX_DIR,
        embedding_model=EMBED_MODEL
    ):
        target_root = Path(yaml_root)
        if not target_root.exists() and (Path(RAG_DIR) / yaml_root).exists():
            target_root = Path(RAG_DIR) / yaml_root

        self.yaml_root = target_root
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = Path(RAG_DIR) / output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.embedder = SentenceTransformer(embedding_model)
        self.loader = YAMLLoader(yaml_root=str(self.yaml_root), recursive=True)
        self.registry = None

    # ======================================================
    # BUILD SEARCHABLE DOCUMENT
    # ======================================================

    def _build_document(self, template):
        fields = []

        simple_fields = [
            "id",
            "name",
            "category",
            "subcategory",
            "description",
            "difficulty"
        ]

        for field in simple_fields:
            value = template.get(field)
            if value:
                fields.append(str(value))

        for item in template.get("learning_objectives", []):
            fields.append(str(item))

        tags = template.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        fields.extend([str(t) for t in tags])

        platforms = template.get("platforms", [])
        if isinstance(platforms, str):
            platforms = [platforms]
        fields.extend([str(p) for p in platforms])

        mitre = template.get("mitre_attack", [])
        if isinstance(mitre, str):
            mitre = [mitre]
        fields.extend([str(m) for m in mitre])

        for control in template.get("compliance_controls", []):
            if isinstance(control, dict):
                standard = control.get("standard")
                if standard:
                    fields.append(str(standard))
                cid = control.get("control")
                if cid:
                    fields.append(str(cid))
                desc = control.get("description")
                if desc:
                    fields.append(str(desc))

        classification = template.get("classification", {})
        if isinstance(classification, dict):
            for value in classification.values():
                if isinstance(value, list):
                    fields.extend([str(v) for v in value])
                else:
                    fields.append(str(value))

        return "\n".join(fields)

    # ======================================================
    # BUILD INDEX
    # ======================================================

    def build(self):
        print("=" * 60)
        print("Building YAML Semantic Index")
        print(f"Source Scenarios: {self.yaml_root}")
        print(f"Output Directory: {self.output_dir}")
        print("=" * 60)

        self.registry = self.loader.load()

        documents = []
        metadata = []

        for template in self.registry.all():
            documents.append(self._build_document(template))
            metadata.append(template)

        if not documents:
            print("[WARNING] No YAML templates found to index.")
            return None, None

        print(f"Generating embeddings for {len(documents)} templates...")
        embeddings = self.embedder.encode(
            documents,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        embeddings = embeddings.astype(np.float32)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        index.add(embeddings)

        index_file = self.output_dir / "yaml.index"
        meta_file = self.output_dir / "yaml_meta.pkl"

        faiss.write_index(index, str(index_file))

        with open(meta_file, "wb") as f:
            pickle.dump(metadata, f)

        print("\nDone.\n")
        print(f"Templates : {len(metadata)}")
        print(f"Dimension : {dimension}")
        print(f"Index     : {index_file}")
        print(f"Metadata  : {meta_file}")

        return index_file, meta_file


# ==========================================================
# STANDALONE BUILD
# ==========================================================

if __name__ == "__main__":
    indexer = YAMLIndexer()
    indexer.build()