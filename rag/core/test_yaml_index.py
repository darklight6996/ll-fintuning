"""
core/test_yaml_index.py

Unit tests for Phase 3: Persistent YAML Vector Index with Change Detection and Incremental Updates.
Validates:
1. Persistent index creation (yaml.index + yaml_meta.pkl).
2. Fast startup / zero re-embedding when files are unchanged.
3. Modification detection and incremental update (only changed templates are re-embedded).
4. Deletion detection and index synchronization.
5. Semantic search accuracy and filtering (category/tag).
"""

import os
import sys
import shutil
import tempfile
import unittest
from pathlib import Path
import yaml
import numpy as np

# Resolve system paths to include rag/
RAG_DIR = Path(__file__).resolve().parent.parent
if str(RAG_DIR) not in sys.path:
    sys.path.insert(0, str(RAG_DIR))

from core.yaml_indexer import YAMLIndexer
from core.yaml_search import YAMLSearch
from core.services.yaml_service import get_yaml_service


class TestYAMLIndex(unittest.TestCase):

    def setUp(self):
        """Create a temporary scenario library and index directory for isolated testing."""
        self.temp_dir = tempfile.mkdtemp(prefix="test_yaml_index_")
        self.scenario_dir = Path(self.temp_dir) / "scenarios"
        self.index_dir = Path(self.temp_dir) / "yaml_index"
        self.scenario_dir.mkdir(parents=True)
        self.index_dir.mkdir(parents=True)

        # Create two sample scenario YAMLs
        self.tmpl1 = {
            "id": "TEST-AZ-01",
            "name": "Azure AD Privilege Escalation Assessment",
            "category": "cloud",
            "subcategory": "azure_active_directory",
            "description": "Assess role assignments and PIM privilege escalation pathways in Azure AD.",
            "tags": ["azure", "privilege_escalation", "cloud", "iam"],
            "difficulty": "intermediate",
            "steps": [
                {"action": "Enumerate Directory Roles", "phase": "reconnaissance"},
                {"action": "Check PIM assignment bypasses", "phase": "exploitation"}
            ]
        }

        self.tmpl2 = {
            "id": "TEST-LIN-01",
            "name": "Linux SUID Binary Exploitation",
            "category": "linux",
            "subcategory": "privilege_escalation",
            "description": "Identify and exploit misconfigured SUID binaries on Linux systems.",
            "tags": ["linux", "suid", "privesc", "gtfobins"],
            "difficulty": "beginner",
            "steps": [
                {"action": "Find SUID binaries with find / -perm -4000", "phase": "reconnaissance"},
                {"action": "Exploit vulnerable binary via GTFOBins", "phase": "exploitation"}
            ]
        }

        with open(self.scenario_dir / "azure_pim.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.tmpl1, f)

        with open(self.scenario_dir / "linux_suid.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.tmpl2, f)

    def tearDown(self):
        """Clean up temporary directory."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_01_initial_index_creation(self):
        """Verify that YAMLIndexer generates yaml.index and yaml_meta.pkl correctly."""
        indexer = YAMLIndexer(
            yaml_root=str(self.scenario_dir),
            output_dir=str(self.index_dir),
            verbose=False
        )

        index, meta = indexer.sync_index()

        self.assertTrue((self.index_dir / "yaml.index").exists())
        self.assertTrue((self.index_dir / "yaml_meta.pkl").exists())
        self.assertEqual(len(meta["templates"]), 2)
        self.assertIn("TEST-AZ-01", meta["template_ids"])
        self.assertIn("TEST-LIN-01", meta["template_ids"])
        self.assertEqual(index.ntotal, 2)

    def test_02_zero_reembedding_on_unchanged_files(self):
        """Verify that loading an existing index without file changes requires 0 re-embeddings."""
        indexer = YAMLIndexer(
            yaml_root=str(self.scenario_dir),
            output_dir=str(self.index_dir),
            verbose=False
        )
        _, meta1 = indexer.sync_index()

        # Mock the embedder to track if embed_many is called
        call_count = [0]
        original_embed_many = indexer.embedder.embed_many

        def counting_embed_many(*args, **kwargs):
            call_count[0] += 1
            return original_embed_many(*args, **kwargs)

        indexer.embedder.embed_many = counting_embed_many

        # Re-sync without touching any files
        index2, meta2 = indexer.sync_index()

        self.assertEqual(call_count[0], 0, "embed_many should NOT be called when files are unchanged!")
        self.assertEqual(len(meta2["templates"]), 2)

    def test_03_incremental_update_on_file_edit(self):
        """Verify that modifying one YAML only re-embeds that single modified file."""
        indexer = YAMLIndexer(
            yaml_root=str(self.scenario_dir),
            output_dir=str(self.index_dir),
            verbose=False
        )
        indexer.sync_index()

        # Modify only the Azure scenario
        self.tmpl1["description"] = "UPDATED: Advanced Azure PIM role elevation and Entra ID attack paths."
        with open(self.scenario_dir / "azure_pim.yaml", "w", encoding="utf-8") as f:
            yaml.dump(self.tmpl1, f)

        embedded_batches = []
        original_embed_many = indexer.embedder.embed_many

        def tracking_embed_many(texts, *args, **kwargs):
            embedded_batches.append(texts)
            return original_embed_many(texts, *args, **kwargs)

        indexer.embedder.embed_many = tracking_embed_many

        index_updated, meta_updated = indexer.sync_index()

        # Exactly 1 template text should have been embedded, not all 2
        self.assertEqual(len(embedded_batches), 1)
        self.assertEqual(len(embedded_batches[0]), 1, "Only the 1 modified scenario should be re-embedded!")
        self.assertIn("UPDATED", embedded_batches[0][0])
        self.assertEqual(index_updated.ntotal, 2)

    def test_04_incremental_addition_and_deletion(self):
        """Verify adding a new scenario and deleting an existing one."""
        indexer = YAMLIndexer(
            yaml_root=str(self.scenario_dir),
            output_dir=str(self.index_dir),
            verbose=False
        )
        indexer.sync_index()

        # Add a 3rd scenario
        tmpl3 = {
            "id": "TEST-WEB-01",
            "name": "SQL Injection in Login Portal",
            "category": "web_application",
            "subcategory": "sql_injection",
            "description": "Exploit auth bypass via error-based and boolean SQL injection.",
            "tags": ["web", "sqli", "owasp_top_10"],
            "difficulty": "beginner"
        }
        with open(self.scenario_dir / "web_sqli.yaml", "w", encoding="utf-8") as f:
            yaml.dump(tmpl3, f)

        # Delete the Linux scenario
        (self.scenario_dir / "linux_suid.yaml").unlink()

        index3, meta3 = indexer.sync_index()

        self.assertEqual(index3.ntotal, 2)
        self.assertEqual(len(meta3["templates"]), 2)
        self.assertIn("TEST-AZ-01", meta3["template_ids"])
        self.assertIn("TEST-WEB-01", meta3["template_ids"])
        self.assertNotIn("TEST-LIN-01", meta3["template_ids"])

    def test_05_yaml_search_semantic_queries(self):
        """Test YAMLSearch semantic retrieval, score ranking, and filtering."""
        searcher = YAMLSearch(
            yaml_root=str(self.scenario_dir),
            output_dir=str(self.index_dir),
            auto_sync=True,
            verbose=False
        )

        # Query for Azure
        azure_results = searcher.search("How do I escalate privileges in Azure Active Directory?", top_k=2)
        self.assertGreater(len(azure_results), 0)
        self.assertEqual(azure_results[0]["id"], "TEST-AZ-01")
        self.assertGreater(azure_results[0]["_score"], 0.3)

        # Query for Linux
        linux_results = searcher.search("Find SUID binaries and GTFOBins on Ubuntu", top_k=2)
        self.assertGreater(len(linux_results), 0)
        self.assertEqual(linux_results[0]["id"], "TEST-LIN-01")

        # Test category filter
        filtered = searcher.search("privilege escalation", top_k=5, category="cloud")
        for res in filtered:
            self.assertEqual(res["category"], "cloud")

        # Test direct get by ID
        item = searcher.get("TEST-AZ-01")
        self.assertIsNotNone(item)
        self.assertEqual(item["name"], "Azure AD Privilege Escalation Assessment")


if __name__ == "__main__":
    unittest.main()
