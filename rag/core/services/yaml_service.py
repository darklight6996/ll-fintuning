"""
core/services/yaml_service.py

Singleton service that owns the YAML scenario stack:
    YAMLIndexer → YAMLSearch (backed by persistent FAISS index)

Loaded once per process. All consumers call get_yaml_service() instead of
constructing these objects themselves, eliminating duplication and
ensuring sub-10ms startup times with zero re-embeddings when scenarios are unchanged.

Usage:
    from core.services.yaml_service import get_yaml_service
    svc = get_yaml_service()
    if svc:
        results = svc.search("privilege escalation", top_k=3)
"""

from typing import Optional, List, Dict, Any

_INSTANCE: Optional["YAMLService"] = None
_LOAD_ATTEMPTED: bool = False


class YAMLService:
    """
    Facade over persistent YAML indexing & search.

    Responsibilities
    ----------------
    * Fast startup via persistent vector index.
    * Automatic change detection on disk.
    * Expose search() and get() for semantic scenario lookup.
    """

    def __init__(self, verbose: bool = False):
        self._search = None
        self._registry = None
        self._load(verbose=verbose)

    def _load(self, verbose: bool = False) -> None:
        try:
            from pathlib import Path
            from core.config import SCENARIOS_DIR, YAML_INDEX_DIR
            from core.yaml_search import YAMLSearch

            scenarios_path = Path(SCENARIOS_DIR)
            if not scenarios_path.exists():
                if verbose:
                    print(f"[!] YAMLService: SCENARIOS_DIR not found ({SCENARIOS_DIR})")
                return

            self._search = YAMLSearch(
                yaml_root=str(scenarios_path),
                output_dir=YAML_INDEX_DIR,
                auto_sync=True,
                verbose=verbose
            )

            if verbose:
                count = len(self._search)
                print(f"[+] YAMLService ready: {count} scenario(s) indexed.")

        except Exception as exc:
            print(f"[!] YAMLService load failed ({type(exc).__name__}: {exc})")
            self._search = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def is_available(self) -> bool:
        """True if scenarios loaded and search index is ready."""
        return self._search is not None and self._search.indexed

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: Optional[float] = None,
        category: Optional[str] = None,
        tag: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for matching YAML scenarios.

        Parameters
        ----------
        query : str
            The operator's query or intent.
        top_k : int
            Maximum number of results to return.
        min_score : float, optional
            Cosine similarity threshold.
        category : str, optional
            Filter by category.
        tag : str, optional
            Filter by tag.

        Returns
        -------
        list[dict]
            Matching scenario template dicts (may be empty).
        """
        if not self.is_available:
            return []
        try:
            return self._search.search(
                query=query,
                top_k=top_k,
                min_score=min_score,
                category=category,
                tag=tag
            )
        except Exception:
            return []

    def get(self, template_id: str) -> Optional[Dict[str, Any]]:
        """Get template dictionary by ID."""
        if not self.is_available:
            return None
        return self._search.get(template_id)

    def sync(self, force_rebuild: bool = False) -> None:
        """Sync persistent index with disk scenarios."""
        if self._search:
            self._search.sync(force_rebuild=force_rebuild)

    @property
    def registry(self):
        """Direct access to YAMLRegistry (loaded on-demand for backward compatibility)."""
        if self._registry is None and self._search:
            try:
                from core.yaml_loader import YAMLLoader
                loader = YAMLLoader(str(self._search.indexer.yaml_root), verbose=False)
                self._registry = loader.load()
            except Exception:
                pass
        return self._registry

    @property
    def search_engine(self):
        """Direct access to YAMLSearch instance."""
        return self._search

    def __len__(self) -> int:
        return len(self._search) if self._search else 0


def get_yaml_service(verbose: bool = False) -> Optional[YAMLService]:
    """
    Return the process-wide YAMLService singleton.
    Returns None if the scenarios directory is missing or load failed.

    Parameters
    ----------
    verbose : bool
        Print load status on first initialization.
    """
    global _INSTANCE, _LOAD_ATTEMPTED
    if not _LOAD_ATTEMPTED:
        _LOAD_ATTEMPTED = True
        _INSTANCE = YAMLService(verbose=verbose)
        if not _INSTANCE.is_available:
            _INSTANCE = None
    return _INSTANCE
