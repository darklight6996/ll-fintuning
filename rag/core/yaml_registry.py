"""
core/yaml_registry.py

Purpose
-------
Central registry for all YAML assessment templates.

Responsibilities
----------------
1. Discover YAML files (recursively or flat)
2. Load YAML safely
3. Validate required fields
4. Index metadata for fast searching
5. Provide helper search methods

This module does NOT perform:
- FAISS indexing
- Embedding generation
- LLM interaction
- Command execution

It is simply the source-of-truth for every YAML template.
"""

from pathlib import Path
from collections import defaultdict
import yaml


class YAMLRegistry:

    def __init__(
        self,
        yaml_root: str = "yaml",
        recursive: bool = False
    ):
        """
        Initialize the YAML registry.

        Args:
            yaml_root: Path to directory containing YAML files
            recursive: If True, scan subdirectories recursively.
                       If False, only scan the root directory.
        """
        self.yaml_root = Path(yaml_root)
        self.recursive = recursive

        # Main storage
        self.templates = {}

        # Search indexes
        self.category_index = defaultdict(set)
        self.subcategory_index = defaultdict(set)
        self.tag_index = defaultdict(set)
        self.framework_index = defaultdict(set)
        self.platform_index = defaultdict(set)
        self.difficulty_index = defaultdict(set)

        # Track loading status
        self.loaded = False
        self.failed_files = []
        self.loaded_files = []

        # Required fields for validation
        self.required_fields = [
            "id",
            "name",
            "category",
            "description",
            "steps"
        ]

        # Optional fields that should be validated if present
        self.optional_field_types = {
            "difficulty": str,
            "estimated_time": (int, str),
            "tools": list,
            "prerequisites": list,
            "compliance_controls": list,
            "detection": dict,
            "remediation": dict,
            "variables": dict,
            "assessment_questions": list,
            "references": list
        }

    # ==========================================================
    # LOAD EVERYTHING
    # ==========================================================

    def load(self) -> int:
        """
        Load all YAML files from the root directory.

        Returns:
            Number of successfully loaded templates.
        """
        if not self.yaml_root.exists():
            raise FileNotFoundError(
                f"YAML directory not found: {self.yaml_root}"
            )

        # Reset state
        self.templates = {}
        self.failed_files = []
        self.loaded_files = []
        self._clear_indexes()

        # Determine which files to load
        if self.recursive:
            files = list(self.yaml_root.rglob("*"))
        else:
            files = list(self.yaml_root.glob("*"))

        count = 0

        for file in files:
            if file.suffix.lower() not in (".yaml", ".yml"):
                continue

            success = self._load_file(file)
            if success:
                count += 1

        self.loaded = True
        print(f"Loaded {count} YAML templates.")

        return count

    def _clear_indexes(self):
        """Clear all search indexes."""
        self.category_index.clear()
        self.subcategory_index.clear()
        self.tag_index.clear()
        self.framework_index.clear()
        self.platform_index.clear()
        self.difficulty_index.clear()

    # ==========================================================
    # LOAD ONE FILE
    # ==========================================================

    def _load_file(self, path: Path) -> bool:
        """
        Load a single YAML file.

        Returns:
            True if loaded successfully, False otherwise.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)

        except yaml.YAMLError as e:
            print(f"[ERROR] YAML syntax error in {path.name}: {e}")
            self.failed_files.append(str(path))
            return False

        except Exception as e:
            print(f"[ERROR] Could not read {path.name}: {e}")
            self.failed_files.append(str(path))
            return False

        if not isinstance(data, dict):
            print(f"[WARNING] Invalid YAML structure (not a dict): {path.name}")
            self.failed_files.append(str(path))
            return False

        template_id = data.get("id")
        if not template_id:
            print(f"[WARNING] Missing 'id' field: {path.name}")
            self.failed_files.append(str(path))
            return False

        # Validate required fields
        validation_errors = self._validate(data, path)
        if validation_errors:
            print(f"[WARNING] {path.name} missing fields: {', '.join(validation_errors)}")
            # We still load it, but track that it's incomplete
            data["__validation_errors"] = validation_errors

        # Store original path
        data["__file"] = str(path)

        self.templates[template_id] = data
        self.loaded_files.append(str(path))
        self._index_metadata(data)

        return True

    # ==========================================================
    # VALIDATION
    # ==========================================================

    def _validate(self, data: dict, path: Path) -> list:
        """
        Validate required fields.

        Returns:
            List of missing required field names.
        """
        missing = []
        for field in self.required_fields:
            if field not in data:
                missing.append(field)
        return missing

    def validate_all(self) -> dict:
        """
        Validate all loaded templates.

        Returns:
            Dict mapping template_id to list of validation errors.
        """
        errors = {}

        for tid, data in self.templates.items():
            field_errors = []

            # Check required fields
            for field in self.required_fields:
                if field not in data:
                    field_errors.append(f"Missing required field: {field}")

            # Check steps structure if present
            steps = data.get("steps", [])
            if not isinstance(steps, list):
                field_errors.append("Steps must be a list")
            else:
                for i, step in enumerate(steps):
                    if not isinstance(step, dict):
                        field_errors.append(f"Step {i+1} is not a dict")
                    elif "phase" not in step:
                        field_errors.append(f"Step {i+1} missing 'phase' field")

            if field_errors:
                errors[tid] = field_errors

        return errors

    # ==========================================================
    # BUILD SEARCH INDEXES
    # ==========================================================

    def _index_metadata(self, data: dict):
        """Index template metadata for fast searching."""
        tid = data["id"]

        category = data.get("category")
        if category:
            self.category_index[category.lower()].add(tid)

        subcategory = data.get("subcategory")
        if subcategory:
            self.subcategory_index[subcategory.lower()].add(tid)

        difficulty = data.get("difficulty")
        if difficulty:
            self.difficulty_index[difficulty.lower()].add(tid)

        # Tags
        tags = data.get("tags", [])
        if isinstance(tags, str):
            tags = [tags]
        for tag in tags:
            self.tag_index[tag.lower()].add(tid)

        # Platforms
        platforms = data.get("platforms", [])
        if isinstance(platforms, str):
            platforms = [platforms]
        for platform in platforms:
            self.platform_index[platform.lower()].add(tid)

        # Compliance frameworks
        controls = data.get("compliance_controls", [])
        for control in controls:
            if isinstance(control, dict):
                framework = control.get("standard")
                if framework:
                    self.framework_index[framework.lower()].add(tid)

    # ==========================================================
    # BASIC LOOKUPS
    # ==========================================================

    def get(self, template_id: str):
        """Get a template by ID, or None if not found."""
        return self.templates.get(template_id)

    def all(self) -> list:
        """Return all loaded templates."""
        return list(self.templates.values())

    def exists(self, template_id: str) -> bool:
        """Check if a template with the given ID exists."""
        return template_id in self.templates

    # ==========================================================
    # SEARCH METHODS
    # ==========================================================

    def search_by_category(self, category: str) -> list:
        """Search templates by category."""
        ids = self.category_index.get(category.lower(), set())
        return [self.templates[i] for i in ids if i in self.templates]

    def search_by_subcategory(self, subcategory: str) -> list:
        """Search templates by subcategory."""
        ids = self.subcategory_index.get(subcategory.lower(), set())
        return [self.templates[i] for i in ids if i in self.templates]

    def search_by_tag(self, tag: str) -> list:
        """Search templates by tag."""
        ids = self.tag_index.get(tag.lower(), set())
        return [self.templates[i] for i in ids if i in self.templates]

    def search_by_framework(self, framework: str) -> list:
        """Search templates by compliance framework."""
        ids = self.framework_index.get(framework.lower(), set())
        return [self.templates[i] for i in ids if i in self.templates]

    def search_by_platform(self, platform: str) -> list:
        """Search templates by platform."""
        ids = self.platform_index.get(platform.lower(), set())
        return [self.templates[i] for i in ids if i in self.templates]

    def search_by_difficulty(self, difficulty: str) -> list:
        """Search templates by difficulty level."""
        ids = self.difficulty_index.get(difficulty.lower(), set())
        return [self.templates[i] for i in ids if i in self.templates]

    # ==========================================================
    # FILTERS
    # ==========================================================

    def filter(self, **kwargs) -> list:
        """
        Filter templates by any field value.

        Example:
            registry.filter(category="web_application", difficulty="intermediate")
        """
        results = self.all()

        for key, value in kwargs.items():
            value = str(value).lower()
            results = [
                t for t in results
                if str(t.get(key, "")).lower() == value
            ]

        return results

    # ==========================================================
    # STATISTICS
    # ==========================================================

    def stats(self) -> dict:
        """Return statistics about the loaded templates."""
        return {
            "loaded": self.loaded,
            "template_count": len(self.templates),
            "categories": len(self.category_index),
            "subcategories": len(self.subcategory_index),
            "frameworks": len(self.framework_index),
            "platforms": len(self.platform_index),
            "tags": len(self.tag_index),
            "difficulty_levels": len(self.difficulty_index),
            "failed_files": len(self.failed_files)
        }

    def list_ids(self) -> list:
        """Return sorted list of all template IDs."""
        return sorted(self.templates.keys())

    # ==========================================================
    # SUMMARY
    # ==========================================================

    def summary(self):
        """Print a summary of the registry contents."""
        stats = self.stats()

        print("\n" + "=" * 60)
        print("YAML Registry Summary")
        print("=" * 60)

        if not self.loaded:
            print("  Status: NOT LOADED")
            return

        print(f"  Total Templates       : {stats['template_count']}")
        print(f"  Categories            : {stats['categories']}")
        print(f"  Subcategories         : {stats['subcategories']}")
        print(f"  Compliance Frameworks : {stats['frameworks']}")
        print(f"  Platforms             : {stats['platforms']}")
        print(f"  Difficulty Levels     : {stats['difficulty_levels']}")
        print(f"  Failed to Load        : {stats['failed_files']}")

        if self.failed_files:
            print("\n  Failed Files:")
            for f in self.failed_files[:5]:
                print(f"    - {f}")

        print("=" * 60 + "\n")