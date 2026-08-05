"""
core/evidence_manager.py

Purpose
-------
System memory for assessment proof and execution records, distinct from conversational memory.

Responsibilities
----------------
1. Store proof of every completed assessment step (command, output, timestamp, status).
2. Organize evidence by assessment ID -> YAML ID -> Step ID.
3. Provide querying and export facilities for downstream report generation.
"""

from typing import Dict, Any, List, Optional
import time
import json
from pathlib import Path


class EvidenceRecord:
    """
    Individual evidence record.
    """

    def __init__(
        self,
        assessment_id: str,
        yaml_id: str,
        step_id: str,
        command: str,
        output: str,
        status: str = "success",
        timestamp: Optional[float] = None,
        artifacts: Optional[List[str]] = None
    ):
        self.assessment_id = assessment_id
        self.yaml_id = yaml_id
        self.step_id = step_id
        self.command = command
        self.output = output
        self.status = status
        self.timestamp = timestamp or time.time()
        self.artifacts = artifacts or []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "yaml_id": self.yaml_id,
            "step_id": self.step_id,
            "command": self.command,
            "output": self.output,
            "status": self.status,
            "timestamp": self.timestamp,
            "artifacts": self.artifacts
        }


class EvidenceManager:
    """
    Manager for storing, querying, and exporting assessment evidence.
    """

    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = Path(storage_path) if storage_path else None
        self.records: List[EvidenceRecord] = []

        if self.storage_path and self.storage_path.exists():
            self._load_from_disk()

    def add_evidence(
        self,
        assessment_id: str,
        yaml_id: str,
        step_id: str,
        command: str,
        output: str,
        status: str = "success",
        artifacts: Optional[List[str]] = None
    ) -> EvidenceRecord:
        """
        Record a new piece of assessment evidence.
        """
        record = EvidenceRecord(
            assessment_id=assessment_id,
            yaml_id=yaml_id,
            step_id=step_id,
            command=command,
            output=output,
            status=status,
            artifacts=artifacts
        )
        self.records.append(record)

        if self.storage_path:
            self._save_to_disk()

        return record

    def get_evidence_for_assessment(self, assessment_id: str) -> List[EvidenceRecord]:
        """
        Retrieve all evidence records associated with a specific assessment.
        """
        return [r for r in self.records if r.assessment_id == assessment_id]

    def get_evidence_by_yaml(self, yaml_id: str) -> List[EvidenceRecord]:
        """
        Retrieve evidence records for a specific YAML procedure across assessments.
        """
        return [r for r in self.records if r.yaml_id == yaml_id]

    def export_summary(self, assessment_id: str) -> Dict[str, Any]:
        """
        Export a structured evidence summary suitable for report generation.
        """
        assessment_records = self.get_evidence_for_assessment(assessment_id)
        
        steps_summary = []
        for r in assessment_records:
            steps_summary.append({
                "yaml_id": r.yaml_id,
                "step_id": r.step_id,
                "command": r.command,
                "output_preview": r.output[:200] + ("..." if len(r.output) > 200 else ""),
                "status": r.status,
                "timestamp": r.timestamp,
                "artifacts": r.artifacts
            })

        return {
            "assessment_id": assessment_id,
            "total_evidence_count": len(assessment_records),
            "evidence_steps": steps_summary
        }

    def _save_to_disk(self):
        """Persist evidence to disk as JSON."""
        if not self.storage_path:
            return
        data = [r.to_dict() for r in self.records]
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load_from_disk(self):
        """Load evidence from disk JSON."""
        if not self.storage_path or not self.storage_path.exists():
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                self.records = [
                    EvidenceRecord(
                        assessment_id=d["assessment_id"],
                        yaml_id=d["yaml_id"],
                        step_id=d["step_id"],
                        command=d["command"],
                        output=d["output"],
                        status=d.get("status", "success"),
                        timestamp=d.get("timestamp"),
                        artifacts=d.get("artifacts", [])
                    )
                    for d in data
                ]
        except Exception as e:
            print(f"[EvidenceManager] Error loading evidence from disk: {e}")
