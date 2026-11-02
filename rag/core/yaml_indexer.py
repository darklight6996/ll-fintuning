"""
core/yaml_indexer.py

Purpose
-------
Build, persist, and maintain an incremental FAISS vector index for all YAML assessment templates.

Responsibilities
----------------
1. Scan YAML scenario templates with SHA-256 hash-based change detection.
2. Persist `yaml.index` (FAISS) and `yaml_meta.pkl` (metadata + file hashes + embeddings cache).
3. Zero re-embedding on startup when files are unchanged.
4. Incremental indexing: re-embed ONLY added or modified YAML files, preserving embeddings of unchanged templates.
5. Provide high-performance vector lookup for YAMLSearch and AssessmentRouter.
"""

import os
import sys
import hashlib
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import faiss
import numpy as np
import yaml

# Resolve system paths
CORE_DIR = os.path.dirname(os.path.abspath(__file__))
RAG_DIR = os.path.dirname(CORE_DIR)
if RAG_DIR not in sys.path:
    sys.path.insert(0, RAG_DIR)

from core.yaml_loader import YAMLLoader
from core.yaml_registry import YAMLRegistry
from core.config import SCENARIOS_DIR, YAML_INDEX_DIR, YAML_INDEX_PATH, YAML_META_PATH, EMBED_MODEL
from core.services.embedding_service import get_embedding_service


def compute_file_hash(filepath: Path) -> str:
    """Compute SHA-256 hash of a file."""
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


class YAMLIndexer:
    """
    Manages persistent FAISS indexing and incremental synchronization for YAML scenarios.
    """

    def __init__(
        self,
        yaml_root: str = SCENARIOS_DIR,
        output_dir: str = YAML_INDEX_DIR,
        embedding_model: str = EMBED_MODEL,
        verbose: bool = False
    ):
        target_root = Path(yaml_root)
        if not target_root.exists() and (Path(RAG_DIR) / yaml_root).exists():
            target_root = Path(RAG_DIR) / yaml_root

        self.yaml_root = target_root
        self.output_dir = Path(output_dir)
        if not self.output_dir.is_absolute():
            self.output_dir = Path(RAG_DIR) / output_dir

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.output_dir / "yaml.index"
        self.meta_file = self.output_dir / "yaml_meta.pkl"

        self.embedding_model = embedding_model
        self.embedder = get_embedding_service(model_name=embedding_model)
        self.verbose = verbose

    # ======================================================
    # DOCUMENT TEXT BUILDER
    # ======================================================

    def build_document_text(self, template: Dict[str, Any]) -> str:
        """
        Convert a template into a dense, searchable semantic representation.
        Includes: title/name, category, subcategory, description, difficulty,
        learning objectives, tags, platforms, MITRE techniques, compliance, and steps.
        """
        fields = []

        simple_fields = [
            ("Title", template.get("name") or template.get("title")),
            ("Category", template.get("category")),
            ("Subcategory", template.get("subcategory")),
            ("Description", template.get("description")),
            ("Difficulty", template.get("difficulty"))
        ]

        for label, val in simple_fields:
            if val:
                fields.append(f"{label}: {val}")

        objectives = template.get("learning_objectives", [])
        if objectives:
            if isinstance(objectives, list):
                fields.append(f"Objectives: {' | '.join(str(o) for o in objectives)}")
            else:
                fields.append(f"Objectives: {objectives}")

        tags = template.get("tags", [])
        if tags:
            if isinstance(tags, list):
                fields.append(f"Tags: {', '.join(str(t) for t in tags)}")
            else:
                fields.append(f"Tags: {tags}")

        platforms = template.get("platforms", [])
        if platforms:
            if isinstance(platforms, list):
                fields.append(f"Platforms: {', '.join(str(p) for p in platforms)}")
            else:
                fields.append(f"Platforms: {platforms}")

        mitre = template.get("mitre_attack", [])
        if mitre:
            if isinstance(mitre, list):
                fields.append(f"MITRE ATT&CK: {', '.join(str(m) for m in mitre)}")
            else:
                fields.append(f"MITRE ATT&CK: {mitre}")

        compliance = template.get("compliance_controls", [])
        if compliance:
            comp_strs = []
            for c in compliance:
                if isinstance(c, dict):
                    std = c.get("standard", "")
                    ctrl = c.get("control", "")
                    desc = c.get("description", "")
                    comp_strs.append(f"{std} {ctrl} {desc}".strip())
                else:
                    comp_strs.append(str(c))
            if comp_strs:
                fields.append(f"Compliance: {' | '.join(comp_strs)}")

        steps = template.get("steps", [])
        if steps:
            step_strs = []
            for step in steps:
                if isinstance(step, dict):
                    action = step.get("action", "")
                    phase = step.get("phase", "")
                    details = f"[{phase}] {action}".strip()
                    step_strs.append(details)
                else:
                    step_strs.append(str(step))
            if step_strs:
                fields.append(f"Steps: {' | '.join(step_strs)}")

        return "\n".join(fields)

    # ======================================================
    # SCAN SCENARIO FILES
    # ======================================================

    def scan_files(self) -> Dict[str, Tuple[Path, str]]:
        """
        Scan yaml_root for all .yaml/.yml files.
        Returns mapping: {relative_path_str: (full_path, sha256_hash)}
        """
        scanned = {}
        if not self.yaml_root.exists():
            return scanned

        for root, _, files in os.walk(self.yaml_root):
            for filename in files:
                if filename.endswith(".yaml") or filename.endswith(".yml"):
                    full_path = Path(root) / filename
                    rel_path = str(full_path.relative_to(self.yaml_root)).replace("\\", "/")
                    try:
                        fhash = compute_file_hash(full_path)
                        scanned[rel_path] = (full_path, fhash)
                    except Exception as e:
                        if self.verbose:
                            print(f"[!] Could not hash {full_path}: {e}")
        return scanned

    def _read_yaml_file(self, full_path: Path) -> Optional[Dict[str, Any]]:
        """Read and parse a YAML scenario file."""
        try:
            with open(full_path, "r", encoding="utf-8") as f:
                content = yaml.safe_load(f)
            if isinstance(content, dict):
                if "id" not in content:
                    content["id"] = full_path.stem
                content["_file_path"] = str(full_path)
                return content
        except Exception as e:
            if self.verbose:
                print(f"[!] Error loading YAML {full_path}: {e}")
        return None

    # ======================================================
    # PERSISTENT SYNC & BUILD
    # ======================================================

    def sync_index(self, force_rebuild: bool = False) -> Tuple[faiss.Index, Dict[str, Any]]:
        """
        Synchronize the persistent index with disk scenarios.
        - If persistent index exists and all file hashes match: instantaneous load.
        - If files were added, modified, or removed: computes embeddings incrementally.
        - If no index exists or force_rebuild=True: full build.

        Returns
        -------
        Tuple[faiss.Index, dict]:
            - faiss_index: Loaded FAISS IndexFlatIP.
            - meta_store: Dict containing templates, template_ids, file_hashes, etc.
        """
        current_files = self.scan_files()

        # Check if existing index and metadata can be loaded
        if not force_rebuild and self.index_file.exists() and self.meta_file.exists():
            try:
                with open(self.meta_file, "rb") as f:
                    meta_store = pickle.load(f)

                stored_hashes = meta_store.get("file_hashes", {})
                current_hash_map = {rel: info[1] for rel, info in current_files.items()}

                # Check if current files match stored hashes exactly
                if stored_hashes == current_hash_map and "embeddings" in meta_store:
                    faiss_index = faiss.read_index(str(self.index_file))
                    if self.verbose:
                        print(f"[+] Loaded persistent YAML index ({len(meta_store.get('templates', []))} scenarios, 0 re-embeddings needed).")
                    return faiss_index, meta_store

                # If hashes differ, perform incremental update
                return self._incremental_sync(current_files, meta_store)

            except Exception as e:
                if self.verbose:
                    print(f"[!] Failed loading persistent index ({e}), triggering full rebuild.")

        return self._full_build(current_files)

    def _incremental_sync(
        self,
        current_files: Dict[str, Tuple[Path, str]],
        meta_store: Dict[str, Any]
    ) -> Tuple[faiss.Index, Dict[str, Any]]:
        """
        Incrementally re-embed only modified and newly added templates.
        """
        stored_hashes = meta_store.get("file_hashes", {})
        stored_templates = meta_store.get("templates_by_file", {})
        stored_embeddings = meta_store.get("embeddings_by_file", {})

        new_templates_by_file = {}
        new_embeddings_by_file = {}
        new_hashes = {}

        to_embed_files = []
        to_embed_texts = []
        to_embed_templates = []

        for rel_path, (full_path, fhash) in current_files.items():
            new_hashes[rel_path] = fhash
            if rel_path in stored_hashes and stored_hashes[rel_path] == fhash and rel_path in stored_embeddings:
                # File is unchanged; reuse cached template & embedding
                new_templates_by_file[rel_path] = stored_templates[rel_path]
                new_embeddings_by_file[rel_path] = stored_embeddings[rel_path]
            else:
                # File is new or modified; parse and queue for embedding
                tmpl = self._read_yaml_file(full_path)
                if tmpl:
                    to_embed_files.append(rel_path)
                    to_embed_templates.append(tmpl)
                    to_embed_texts.append(self.build_document_text(tmpl))

        if self.verbose:
            print(f"[*] Incremental YAML sync: {len(to_embed_files)} scenario(s) changed/added, {len(new_templates_by_file)} unchanged.")

        # Embed newly added / modified templates in batch
        if to_embed_texts:
            fresh_embeddings = self.embedder.embed_many(to_embed_texts, normalize=True)
            for rel_path, tmpl, vec in zip(to_embed_files, to_embed_templates, fresh_embeddings):
                new_templates_by_file[rel_path] = tmpl
                new_embeddings_by_file[rel_path] = vec

        # Assemble final ordered lists
        ordered_rel_paths = sorted(new_templates_by_file.keys())
        all_templates = [new_templates_by_file[r] for r in ordered_rel_paths]
        all_template_ids = [tmpl.get("id") for tmpl in all_templates]

        if ordered_rel_paths:
            all_embeddings = np.array([new_embeddings_by_file[r] for r in ordered_rel_paths], dtype=np.float32)
            dimension = all_embeddings.shape[1]
            index = faiss.IndexFlatIP(dimension)
            index.add(all_embeddings)
        else:
            dimension = self.embedder.dimension
            index = faiss.IndexFlatIP(dimension)
            all_embeddings = np.empty((0, dimension), dtype=np.float32)

        meta_store = {
            "version": "1.0",
            "file_hashes": new_hashes,
            "templates": all_templates,
            "template_ids": all_template_ids,
            "templates_by_file": new_templates_by_file,
            "embeddings_by_file": new_embeddings_by_file,
            "embeddings": all_embeddings
        }

        self._save_index_and_meta(index, meta_store)
        return index, meta_store

    def _full_build(self, current_files: Dict[str, Tuple[Path, str]]) -> Tuple[faiss.Index, Dict[str, Any]]:
        """
        Perform a complete build of the FAISS index and metadata store.
        """
        if self.verbose:
            print("=" * 60)
            print("Building Full YAML Semantic Index")
            print(f"Source Scenarios : {self.yaml_root}")
            print(f"Output Directory : {self.output_dir}")
            print("=" * 60)

        templates_by_file = {}
        file_hashes = {}
        documents = []
        ordered_files = []

        for rel_path in sorted(current_files.keys()):
            full_path, fhash = current_files[rel_path]
            tmpl = self._read_yaml_file(full_path)
            if tmpl:
                templates_by_file[rel_path] = tmpl
                file_hashes[rel_path] = fhash
                ordered_files.append(rel_path)
                documents.append(self.build_document_text(tmpl))

        if not documents:
            dimension = self.embedder.dimension
            index = faiss.IndexFlatIP(dimension)
            meta_store = {
                "version": "1.0",
                "file_hashes": {},
                "templates": [],
                "template_ids": [],
                "templates_by_file": {},
                "embeddings_by_file": {},
                "embeddings": np.empty((0, dimension), dtype=np.float32)
            }
            self._save_index_and_meta(index, meta_store)
            return index, meta_store

        if self.verbose:
            print(f"Generating embeddings for {len(documents)} templates...")

        embeddings = self.embedder.embed_many(documents, normalize=True, show_progress_bar=self.verbose)
        embeddings = np.array(embeddings, dtype=np.float32)
        dimension = embeddings.shape[1]

        index = faiss.IndexFlatIP(dimension)
        index.add(embeddings)

        embeddings_by_file = {
            rel_path: embeddings[i]
            for i, rel_path in enumerate(ordered_files)
        }

        all_templates = [templates_by_file[r] for r in ordered_files]
        all_template_ids = [tmpl.get("id") for tmpl in all_templates]

        meta_store = {
            "version": "1.0",
            "file_hashes": file_hashes,
            "templates": all_templates,
            "template_ids": all_template_ids,
            "templates_by_file": templates_by_file,
            "embeddings_by_file": embeddings_by_file,
            "embeddings": embeddings
        }

        self._save_index_and_meta(index, meta_store)

        if self.verbose:
            print(f"\n[+] YAML Index Build Complete.")
            print(f"Templates : {len(all_templates)}")
            print(f"Dimension : {dimension}")
            print(f"Index     : {self.index_file}")
            print(f"Metadata  : {self.meta_file}\n")

        return index, meta_store

    def _save_index_and_meta(self, index: faiss.Index, meta_store: Dict[str, Any]) -> None:
        """Save FAISS index and metadata pickle file to disk."""
        faiss.write_index(index, str(self.index_file))
        with open(self.meta_file, "wb") as f:
            pickle.dump(meta_store, f)

    def build(self) -> Tuple[Path, Path]:
        """Backward-compatible build trigger."""
        self.sync_index(force_rebuild=True)
        return self.index_file, self.meta_file


if __name__ == "__main__":
    indexer = YAMLIndexer(verbose=True)
    indexer.build()