"""
core/session_manager.py

Purpose
-------
Manages assessment session lifecycle, state persistence, evidence collection,
and target tracking in dedicated local folder structures (.assessments/<session_name>/).

Responsibilities
----------------
1. Maintain session state including scope, targets, notes, evidence, and planner steps.
2. Persist state to `.assessments/<session_name>/state.json`.
3. Provide isolated session workspace with an `artifacts/` subfolder.
4. Support clean session creation, loading, updating, and inspection.
"""

import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Optional


class SessionManager:
    """
    Manages local assessment state, storage directories, and session metadata.
    """

    DEFAULT_ASSESSMENTS_DIR = ".assessments"

    def __init__(self, base_dir: Optional[str] = None):
        """
        Initialize SessionManager.

        Args:
            base_dir: Optional override for the root assessments folder.
                      Defaults to '.assessments' in the working directory or project root.
        """
        if base_dir:
            self.base_dir = Path(base_dir).resolve()
        else:
            # Anchor relative to current working directory or search upward
            cwd = Path.cwd().resolve()
            self.base_dir = cwd / self.DEFAULT_ASSESSMENTS_DIR

        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.active_session_name: Optional[str] = None
        self.session_dir: Optional[Path] = None
        self.artifacts_dir: Optional[Path] = None
        self.state_file: Optional[Path] = None
        self.state: Dict[str, Any] = {}

    def _get_timestamp(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def list_sessions(self) -> List[str]:
        """
        List all available assessment sessions in the base directory.
        """
        if not self.base_dir.exists():
            return []
        sessions = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and (item / "state.json").exists():
                sessions.append(item.name)
        return sorted(sessions)

    def create_session(
        self,
        name: str,
        scope: str = "",
        target: str = "",
        target_type: str = "",
        mode: str = "",
    ) -> Dict[str, Any]:
        """
        Initialize a new assessment session folder and state.

        Args:
            name: Human-readable name/slug for the session.
            scope: Engagement scope (IPs, domains, cloud accounts, applications).
            target: Primary target IP, CIDR, or hostname.
            target_type: Assessment category (e.g., 'Active Directory', 'Web', 'Cloud').

        Returns:
            Initial session state dictionary.
        """
        clean_name = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in name.strip())
        if not clean_name:
            raise ValueError("Session name cannot be empty or invalid.")

        self.active_session_name = clean_name
        self.session_dir = self.base_dir / clean_name
        self.artifacts_dir = self.session_dir / "artifacts"
        self.state_file = self.session_dir / "state.json"

        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        now = self._get_timestamp()
        self.state = {
            "name": clean_name,
            "target": target.strip(),
            "type": target_type.strip(),
            "scope": scope.strip(),
            # Related --resume fix: persist the run mode ("manual"/"automated") too --
            # previously unset entirely, so main.py's --resume path always read back
            # state.get("mode", "manual") and silently resumed an autonomous session
            # as a manual one, calling agent.run() instead of agent.run_autonomous().
            "mode": mode.strip().lower() if mode else "manual",
            "status": "in_progress",
            "created_at": now,
            "updated_at": now,
            "current_step": None,
            "completed_steps": [],
            "notes": [],
            "evidence": [],
            "ingested_findings": {
                "nmap": [],
                "burp": [],
                "cloud": [],
                "generic": []
            },
            "targets": {
                "ips": [],
                "ports": [],
                "domains": [],
                "services": [],
                "cloud_resources": [],
                "vulnerabilities": []
            },
            "history": []
        }

        # Auto-extract any initial targets from target and scope
        if target:
            self._extract_targets_from_scope(target)
        if scope:
            self._extract_targets_from_scope(scope)

        self.save_current_state()
        return self.state

    def load_session(self, name: str) -> Dict[str, Any]:
        """
        Load an existing assessment session from disk.

        Args:
            name: Name of the session to load.

        Returns:
            Loaded state dictionary.
        """
        clean_name = name.strip()
        target_dir = self.base_dir / clean_name
        target_file = target_dir / "state.json"

        if not target_file.exists():
            raise FileNotFoundError(f"Assessment session '{clean_name}' does not exist at {target_dir}.")

        self.active_session_name = clean_name
        self.session_dir = target_dir
        self.artifacts_dir = self.session_dir / "artifacts"
        self.state_file = target_file
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        with open(target_file, "r", encoding="utf-8") as f:
            self.state = json.load(f)

        return self.state

    def save_scope(self, scope: Any) -> None:
        """
        --resume fix: persist the FULL Scope object (every authorization flag, not
        just the target) into session state, so resuming a session restores the exact
        engagement boundaries and authorizations it was created with.

        Before this, only a stringified `in_scope` list went into state["scope"] (set
        once, at session creation, from `str(self.scope.in_scope)` -- see BaseAgent.run()
        / run_autonomous()). Every other Scope field -- out_of_scope, prohibited_tests,
        prohibited_tools, time_window, authorization_ref, and critically
        active_testing_allowed / active_network_allowed / state_changes_allowed /
        recon_only_authorized (the C3 independent authorization flags) -- was never
        written anywhere, so main.py's --resume path had no way to recover them and
        silently reconstructed a bare, fully-unauthorized `Scope(target, in_scope)`.
        A resumed autonomous run would then have every active-testing action rejected
        at Gate 3, even if the original session had explicitly authorized them.
        """
        import dataclasses

        if dataclasses.is_dataclass(scope):
            self.state["scope_full"] = dataclasses.asdict(scope)
        else:
            # Defensive fallback for a non-dataclass Scope-like object (e.g. a test
            # double): persist whatever plain attributes we can find.
            self.state["scope_full"] = {
                k: v for k, v in vars(scope).items()
                if isinstance(v, (str, bool, int, float, list, dict, type(None)))
            }
        self.save_current_state()

    def load_scope(self) -> Optional[Dict[str, Any]]:
        """
        Return the full persisted Scope dict saved by save_scope(), or None if this
        session predates that field (older sessions saved before this fix) -- callers
        must fall back to a bare Scope(target, in_scope) in that case, exactly as
        main.py's --resume path already did.
        """
        return self.state.get("scope_full")

    def save_current_state(self) -> None:
        """
        Persist current session state to state.json atomically.
        """
        if not self.state_file or not self.session_dir:
            raise RuntimeError("No active session to save. Create or open a session first.")

        self.state["updated_at"] = self._get_timestamp()
        temp_file = self.session_dir / "state.json.tmp"

        # DEFENSIVE FALLBACK (2026-11-13): a non-JSON-serializable value (e.g. raw
        # bytes from a subprocess.TimeoutExpired.stdout, before the dedicated fix in
        # command_executor.py) must never crash the whole autonomous run by failing
        # to persist state. default=_json_fallback degrades anything json can't
        # natively encode to a string instead of raising, trading perfect fidelity
        # for "the run keeps going and the history is still inspectable."
        def _json_fallback(obj):
            if isinstance(obj, bytes):
                return obj.decode("utf-8", errors="replace")
            return repr(obj)

        import time
        for attempt in range(5):
            try:
                with open(temp_file, "w", encoding="utf-8") as f:
                    json.dump(self.state, f, indent=2, ensure_ascii=False, default=_json_fallback)
                temp_file.replace(self.state_file)
                return
            except (PermissionError, OSError):
                time.sleep(0.05 * (attempt + 1))

        # Direct write fallback if atomic replace is blocked by OneDrive sync lock
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2, ensure_ascii=False, default=_json_fallback)
        if temp_file.exists():
            try:
                temp_file.unlink(missing_ok=True)
            except OSError:
                pass

    def _extract_targets_from_scope(self, scope_text: str) -> None:
        """Extract IPs, domains, and ports from scope text."""
        import re
        if not scope_text:
            return

        ip_pattern = r"\b(?:\d{1,3}\.){3}\d{1,3}(?:/\d{1,2})?\b"
        ips = re.findall(ip_pattern, scope_text)
        for ip in ips:
            if ip not in self.state["targets"]["ips"]:
                self.state["targets"]["ips"].append(ip)

        domain_pattern = r"\b[a-zA-Z0-9][-a-zA-Z0-9]*\.[a-zA-Z]{2,}(?:\.[a-zA-Z]{2,})?\b"
        domains = re.findall(domain_pattern, scope_text)
        for domain in domains:
            # Filter out non-domain matches like decimals
            if not domain.replace(".", "").isdigit() and domain not in self.state["targets"]["domains"]:
                self.state["targets"]["domains"].append(domain)

    def update_scope(self, new_scope: str) -> str:
        """Update engagement scope and re-extract targets."""
        self._require_active_session()
        self.state["scope"] = new_scope.strip()
        self._extract_targets_from_scope(new_scope)
        self.save_current_state()
        return self.state["scope"]

    def update_type(self, new_type: str) -> str:
        """Update assessment engagement type (e.g., 'ctf', 'linux', 'web', 'ad')."""
        self._require_active_session()
        self.state["type"] = new_type.strip()
        self.save_current_state()
        return self.state["type"]

    def update_target(self, new_target: str) -> str:
        """Update primary target identifier."""
        self._require_active_session()
        self.state["target"] = new_target.strip()
        self._extract_targets_from_scope(new_target)
        self.save_current_state()
        return self.state["target"]

    def add_target_entity(self, category: str, value: Any) -> bool:
        """Add a specific target entity (ips, domains, ports, cloud_resources, services)."""
        self._require_active_session()
        cat = category.lower().strip()
        if cat in ("ip", "ips"):
            key = "ips"
            val = str(value).strip()
        elif cat in ("domain", "domains", "host", "hosts"):
            key = "domains"
            val = str(value).strip()
        elif cat in ("port", "ports"):
            key = "ports"
            try:
                val = int(value)
            except ValueError:
                return False
        elif cat in ("cloud", "cloud_resources"):
            key = "cloud_resources"
            val = str(value).strip()
        elif cat in ("service", "services"):
            key = "services"
            val = str(value).strip()
        else:
            key = cat
            val = str(value).strip()

        targets = self.state.setdefault("targets", {})
        lst = targets.setdefault(key, [])
        if val not in lst:
            lst.append(val)
            self.save_current_state()
            return True
        return False

    def remove_target_entity(self, category: str, value: Any) -> bool:
        """Remove a specific target entity from the session."""
        self._require_active_session()
        cat = category.lower().strip()
        if cat in ("ip", "ips"):
            key = "ips"
            val = str(value).strip()
        elif cat in ("domain", "domains", "host", "hosts"):
            key = "domains"
            val = str(value).strip()
        elif cat in ("port", "ports"):
            key = "ports"
            try:
                val = int(value)
            except ValueError:
                return False
        else:
            key = cat
            val = str(value).strip()

        targets = self.state.get("targets", {})
        lst = targets.get(key, [])
        if val in lst:
            lst.remove(val)
            self.save_current_state()
            return True
        return False

    def clear_target_category(self, category: Optional[str] = None) -> None:
        """Clear entities in a target category, or all targets if None."""
        self._require_active_session()
        targets = self.state.setdefault("targets", {})
        if category:
            cat = category.lower().strip()
            if cat in ("ip", "ips"):
                targets["ips"] = []
            elif cat in ("domain", "domains"):
                targets["domains"] = []
            elif cat in ("port", "ports"):
                targets["ports"] = []
            elif cat in targets:
                targets[cat] = []
        else:
            for k in targets:
                targets[k] = []
        self.save_current_state()

    def undo_last_step(self) -> Optional[Dict[str, Any]]:
        """Roll back the most recently completed step, making it the active step again."""
        self._require_active_session()
        completed = self.state.get("completed_steps", [])
        if not completed:
            return None
        last_step = completed.pop()
        self.state["current_step"] = last_step
        self.save_current_state()
        return last_step

    def reset_steps(self) -> int:
        """Clear all completed steps and the active step so planner starts fresh."""
        self._require_active_session()
        count = len(self.state.get("completed_steps", []))
        self.state["completed_steps"] = []
        self.state["current_step"] = None
        self.save_current_state()
        return count

    def add_note(self, content: str, author: str = "operator") -> Dict[str, Any]:
        """
        Add a manual note or observation to the session.
        """
        self._require_active_session()
        note = {
            "id": f"note_{len(self.state.get('notes', [])) + 1}",
            "content": content.strip(),
            "author": author,
            "timestamp": self._get_timestamp()
        }
        self.state.setdefault("notes", []).append(note)
        self.save_current_state()
        return note

    def add_evidence(
        self,
        command: str,
        output: str,
        step_id: str = "manual",
        status: str = "success",
        artifacts: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Add an evidence record to the session.
        """
        self._require_active_session()
        record = {
            "id": f"ev_{len(self.state.get('evidence', [])) + 1}",
            "step_id": step_id,
            "command": command,
            "output": output,
            "status": status,
            "timestamp": self._get_timestamp(),
            "artifacts": artifacts or []
        }
        self.state.setdefault("evidence", []).append(record)
        self.save_current_state()
        return record

    def add_ingested_findings(
        self,
        finding_type: str,
        findings: Dict[str, Any],
        raw_source: Optional[str] = None
    ) -> None:
        """
        Merge findings from the ingestion pipeline into session state.

        Args:
            finding_type: Type of finding ('nmap', 'burp', 'cloud', 'generic')
            findings: Structured dictionary with parsed data and target entities
            raw_source: Original path of the ingested file
        """
        self._require_active_session()
        
        now = self._get_timestamp()
        entry = {
            "source_file": raw_source or "upload",
            "ingested_at": now,
            "data": findings
        }
        
        # Defensive: ensure the ingested_findings key and all sub-categories exist
        # (old sessions loaded from disk may predate these keys)
        ingest_store = self.state.setdefault("ingested_findings", {})
        for cat in ("nmap", "burp", "cloud", "generic"):
            ingest_store.setdefault(cat, [])
        ingest_store.setdefault(finding_type, []).append(entry)

        # Merge extracted target entities into global session targets
        entities = findings.get("entities", {})
        for ip in entities.get("ips", []):
            if ip not in self.state["targets"]["ips"]:
                self.state["targets"]["ips"].append(ip)

        for port in entities.get("ports", []):
            if port not in self.state["targets"]["ports"]:
                self.state["targets"]["ports"].append(port)

        for domain in entities.get("domains", []):
            if domain not in self.state["targets"]["domains"]:
                self.state["targets"]["domains"].append(domain)

        for service in entities.get("services", []):
            if service not in self.state["targets"]["services"]:
                self.state["targets"]["services"].append(service)

        for resource in entities.get("cloud_resources", []):
            if resource not in self.state["targets"]["cloud_resources"]:
                self.state["targets"]["cloud_resources"].append(resource)

        for vuln in entities.get("vulnerabilities", []):
            if vuln not in self.state["targets"]["vulnerabilities"]:
                self.state["targets"]["vulnerabilities"].append(vuln)

        # Append timeline entry
        self.state["history"].append({
            "action": f"ingest_{finding_type}",
            "summary": findings.get("summary", f"Ingested {finding_type} artifact"),
            "timestamp": now
        })

        self.save_current_state()

    def set_current_step(self, step_data: Dict[str, Any]) -> None:
        """
        Set the active planner recommended step.
        """
        self._require_active_session()
        step_data["assigned_at"] = self._get_timestamp()
        self.state["current_step"] = step_data
        self.save_current_state()

    def complete_current_step(
        self,
        execution_result: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Mark current step as completed, record its execution proof, and clear current step.
        """
        self._require_active_session()
        current = self.state.get("current_step")
        if not current:
            return None

        now = self._get_timestamp()
        completed_record = dict(current)
        completed_record["completed_at"] = now
        completed_record["execution"] = execution_result or {"status": "manual_confirmation"}

        self.state.setdefault("completed_steps", []).append(completed_record)
        self.state["history"].append({
            "action": "step_completed",
            "step_number": current.get("step_number"),
            "title": current.get("title") or current.get("action"),
            "phase": current.get("phase"),
            "timestamp": now
        })

        # Record evidence if execution result provided
        if execution_result:
            cmd = execution_result.get("command", "")
            out = execution_result.get("stdout", "") or execution_result.get("output", "")
            status = execution_result.get("status", "completed")
            step_id = str(current.get("step_number", "step"))
            self.add_evidence(command=cmd, output=out, step_id=step_id, status=status)

        self.state["current_step"] = None
        self.save_current_state()
        return completed_record

    def get_summary(self) -> Dict[str, Any]:
        """
        Return high-level summary of active session.
        """
        if not self.active_session_name:
            return {"active": False}

        return {
            "active": True,
            "name": self.state.get("name"),
            "scope": self.state.get("scope"),
            "status": self.state.get("status"),
            "created_at": self.state.get("created_at"),
            "completed_steps_count": len(self.state.get("completed_steps", [])),
            "current_step": self.state.get("current_step"),
            "notes_count": len(self.state.get("notes", [])),
            "evidence_count": len(self.state.get("evidence", [])),
            "targets": self.state.get("targets", {}),
            "ingested_counts": {k: len(v) for k, v in self.state.get("ingested_findings", {}).items()}
        }


    def save_investigation_store(self, evidence_store) -> None:
        """
        Persist the EvidenceStore (observation/hypothesis/validation/finding chain)
        to '.assessments/<session>/investigation.json'.

        Args:
            evidence_store: EvidenceStore instance from core.investigation.
        """
        self._require_active_session()
        inv_file = self.session_dir / "investigation.json"
        data = evidence_store.to_dict()
        tmp_file = inv_file.with_suffix(".tmp")
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            tmp_file.replace(inv_file)
        except Exception:
            if tmp_file.exists():
                tmp_file.unlink(missing_ok=True)
            raise

    def load_investigation_store(self):
        """
        Load a previously persisted EvidenceStore.

        Returns:
            EvidenceStore instance if investigation.json exists, else None.
        """
        self._require_active_session()
        from core.investigation import EvidenceStore
        inv_file = self.session_dir / "investigation.json"
        if not inv_file.exists():
            return None
        try:
            with open(inv_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            return EvidenceStore.from_dict(data)
        except Exception:
            return None

    def append_autonomous_history(self, entry: dict) -> None:
        """
        Append one autonomous execution history entry to session state.
        Keeps the in-memory state synchronized with the entry and persists.

        Args:
            entry: Dict containing action_id, status, rejection_reason, results, etc.
        """
        self._require_active_session()
        history = self.state.setdefault("autonomous_execution_history", [])
        history.append(entry)
        # Also record in the audit evidence trail if the action succeeded
        if entry.get("status") == "success" and entry.get("results"):
            results = entry["results"]
            if isinstance(results, list):
                for res in results:
                    argv = res.get("argv", [])
                    cmd = " ".join(argv) if isinstance(argv, list) else str(entry.get("action_id", ""))
                    out = str(res.get("stdout", ""))[:2000]
                    self.state.setdefault("evidence", []).append({
                        "command": cmd,
                        "output": out,
                        "step_id": entry.get("action_id", "autonomous"),
                        "status": "completed",
                        "timestamp": entry.get("timestamp", ""),
                    })
        self.save_current_state()

    def _require_active_session(self) -> None:
        if not self.active_session_name or not self.state:
            raise RuntimeError("No active assessment session. Create or open one first.")
