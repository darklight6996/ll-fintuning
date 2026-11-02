"""
core/yaml_loader.py

Purpose
-------
Thin wrapper around YAMLRegistry.

Responsibilities
----------------
1. Initialize the YAML registry
2. Load all YAML assessment templates
3. Validate loading
4. Provide a simple interface for the rest of the system

This module intentionally contains very little logic.
All indexing, validation, and searching belong in YAMLRegistry.
"""

from pathlib import Path
from core.yaml_registry import YAMLRegistry

class YAMLLoader:
    """
    Loads every assessment template into a YAMLRegistry.

    Example
    -------
    loader = YAMLLoader("yaml")
    loader.load()  # <-- REQUIRED to actually load files

    registry = loader.registry
    """

    def __init__(
        self,
        yaml_root: str = "yaml",
        recursive: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the YAML loader.

        Args:
            yaml_root: Path to directory containing YAML files
            recursive: If True, scan subdirectories recursively
            verbose: If True, print status messages
        """
        self.yaml_root = Path(yaml_root)
        self.recursive = recursive
        self.verbose = verbose

        # Registry will be created and populated on load()
        self.registry = None
        self.loaded = False

        """
core/yaml_loader.py

Purpose
-------
Thin wrapper around YAMLRegistry.

Responsibilities
----------------
1. Initialize the YAML registry
2. Load all YAML assessment templates
3. Validate loading
4. Provide a simple interface for the rest of the system

This module intentionally contains very little logic.
All indexing, validation, and searching belong in YAMLRegistry.
"""

from pathlib import Path

from core.yaml_registry import YAMLRegistry


class YAMLLoader:
    """
    Loads every assessment template into a YAMLRegistry.

    Example
    -------
    loader = YAMLLoader("yaml")
    loader.load()  # <-- REQUIRED to actually load files

    registry = loader.registry
    """

    def __init__(
        self,
        yaml_root: str = "yaml",
        recursive: bool = False,
        verbose: bool = True
    ):
        """
        Initialize the YAML loader.

        Args:
            yaml_root: Path to directory containing YAML files
            recursive: If True, scan subdirectories recursively
            verbose: If True, print status messages
        """
        self.yaml_root = Path(yaml_root)
        self.recursive = recursive
        self.verbose = verbose

        # Registry will be created and populated on load()
        self.registry = None
        self.loaded = False

    # ==========================================================
    # PRIVATE
    # ==========================================================

    def _log(self, message: str):
        """Print log message if verbose is enabled."""
        if self.verbose:
            print(message)

    # ==========================================================
    # LOAD ALL YAML FILES
    # ==========================================================

    def load(self) -> YAMLRegistry:
        """
        Load all YAML templates from the root directory.

        This method:
        1. Creates the registry
        2. Triggers the actual file loading
        3. Validates the loaded templates

        Returns:
            The populated registry instance.
        """
        self._log("=" * 60)
        self._log("Loading YAML Assessment Library")
        self._log(f"  Root: {self.yaml_root}")
        self._log(f"  Recursive: {self.recursive}")
        self._log("=" * 60)

        # Create the registry with the correct parameters
        self.registry = YAMLRegistry(
            yaml_root=self.yaml_root,
            recursive=self.recursive
        )

        # ACTUALLY LOAD THE FILES
        count = self.registry.load()

        self.loaded = True

        self._log(f"\nLoaded {count} assessment templates.")

        # Run additional validation
        validation_errors = self.registry.validate_all()
        if validation_errors:
            self._log(f"WARNING: {len(validation_errors)} templates have validation issues.")
            if self.verbose:
                for tid, errors in validation_errors.items():
                    self._log(f"  - {tid}: {', '.join(errors)}")

        self.registry.summary()

        return self.registry

    # ==========================================================
    # RELOAD
    # ==========================================================

    def reload(self) -> YAMLRegistry:
        """
        Reload all YAML templates from disk.

        This completely resets the registry and reloads all files.
        Useful when templates are updated.
        """
        self._log("\nReloading YAML library...")

        # Reset state
        self.loaded = False
        if hasattr(self, 'registry'):
            del self.registry

        return self.load()

    # ==========================================================
    # STATUS
    # ==========================================================

    def is_loaded(self) -> bool:
        """Return True if templates are loaded."""
        return self.loaded

    # ==========================================================
    # SHORTCUTS (Delegated to Registry)
    # ==========================================================

    def get(self, template_id: str):
        """Get a template by ID, or None if not found."""
        if not self.is_loaded():
            raise RuntimeError("Loader not loaded. Call load() first.")
        return self.registry.get(template_id)

    def require(self, template_id: str):
        """Get a template by ID, raising KeyError if not found."""
        result = self.get(template_id)
        if result is None:
            raise KeyError(f"Template ID '{template_id}' not found.")
        return result

    def all(self) -> list:
        """Return all loaded templates."""
        if not self.is_loaded():
            raise RuntimeError("Loader not loaded. Call load() first.")
        return self.registry.all()

    def stats(self) -> dict:
        """Return statistics about the loaded templates."""
        if not self.is_loaded():
            return {"loaded": False, "message": "Loader not loaded. Call load() first."}
        return self.registry.stats()

    def summary(self):
        """Print a summary of the loaded templates."""
        if not self.is_loaded():
            print("Loader not loaded. Call load() first.")
        else:
            self.registry.summary()

    # ==========================================================
    # SEARCH SHORTCUTS
    # ==========================================================

    def search_by_category(self, category: str) -> list:
        return self.registry.search_by_category(category)

    def search_by_subcategory(self, subcategory: str) -> list:
        return self.registry.search_by_subcategory(subcategory)

    def search_by_tag(self, tag: str) -> list:
        return self.registry.search_by_tag(tag)

    def search_by_platform(self, platform: str) -> list:
        return self.registry.search_by_platform(platform)

    def search_by_framework(self, framework: str) -> list:
        return self.registry.search_by_framework(framework)

    def search_by_difficulty(self, difficulty: str) -> list:
        return self.registry.search_by_difficulty(difficulty)

    def filter(self, **kwargs) -> list:
        """
        Filter templates by any field.

        Supported filters:
            - category (str)
            - subcategory (str)
            - difficulty (str)
            - platform (str)
            - framework (str)
            - tags (list)

        Example:
            loader.filter(category="web_application", difficulty="intermediate")
        """
        return self.registry.filter(**kwargs)

    def validate(self) -> dict:
        """
        Validate all loaded templates.

        Returns:
            Dict mapping template_id to list of validation errors.
        """
        if not self.is_loaded():
            raise RuntimeError("Loader not loaded. Call load() first.")
        return self.registry.validate_all()