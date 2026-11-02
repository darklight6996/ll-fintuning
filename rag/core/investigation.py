"""
core/investigation.py

Formal models and store for Observation, Hypothesis, Validation, Finding, and EvidenceGraph.
Enforces the mandatory progression:
  OBSERVATION -> HYPOTHESIS -> VALIDATION -> FINDING -> IMPACT

Phase 4 Deliverable:
- Evidence graph (Entity -> Relationship -> Entity)
- Cross-observation correlation and multi-hop attack path reasoning
- Preserves strict validation invariants before any Finding can be recorded
"""

from __future__ import annotations

import time
import uuid
from typing import Any, Dict, List, Literal, Optional, Tuple, Set
from pydantic import BaseModel, Field


class Observation(BaseModel):
    """
    A tool produced output. This is a FACT, not a conclusion.
    """
    obs_id: str = Field(default_factory=lambda: f"obs_{uuid.uuid4().hex[:8]}")
    action_id: str
    argv: List[str]
    raw_stdout: str
    raw_stderr: str = ""
    exit_code: int = 0
    observed_at: float = Field(default_factory=time.time)
    parsed_facts: Dict[str, Any] = Field(default_factory=dict)


class Hypothesis(BaseModel):
    """
    A candidate explanation of observations. NOT yet a finding.
    """
    hyp_id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:8]}")
    observation_ids: List[str] = Field(default_factory=list)
    description: str
    validation_test: str
    confidence: Literal["speculative", "plausible", "likely"] = "plausible"
    status: Literal["open", "validating", "confirmed", "refuted", "inconclusive"] = "open"
    created_at: float = Field(default_factory=time.time)


class Validation(BaseModel):
    """
    Evidence collected to confirm or refute a hypothesis.
    """
    val_id: str = Field(default_factory=lambda: f"val_{uuid.uuid4().hex[:8]}")
    hypothesis_id: str
    action_id: str
    observation: Observation
    verdict: Literal["supports", "refutes", "inconclusive"]
    reasoning: str


class Finding(BaseModel):
    """
    A validated vulnerability backed by an unbroken evidence chain.
    """
    finding_id: str = Field(default_factory=lambda: f"fnd_{uuid.uuid4().hex[:8]}")
    hypothesis_id: str
    observation_ids: List[str]
    validation_ids: List[str]
    title: str
    description: str
    severity: Literal["info", "low", "medium", "high", "critical"] = "medium"
    affected_target: str
    evidence_summary: str
    reproducible_steps: List[str] = Field(default_factory=list)
    impact: str = "pending further validation"
    confidence: float = 1.0


class Entity(BaseModel):
    """
    An identified security entity in the target environment (host, port, user, credential, etc.).
    """
    entity_id: str = Field(default_factory=lambda: f"ent_{uuid.uuid4().hex[:8]}")
    entity_type: Literal[
        "host", "port", "service", "user", "credential",
        "vulnerability", "endpoint", "file", "token", "domain"
    ]
    name: str
    value: str
    properties: Dict[str, Any] = Field(default_factory=dict)
    source_observation_ids: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class Relationship(BaseModel):
    """
    A directed relationship connecting two entities in the evidence graph.
    """
    rel_id: str = Field(default_factory=lambda: f"rel_{uuid.uuid4().hex[:8]}")
    source_id: str
    relation_type: Literal[
        "runs_on", "authenticates_as", "member_of", "vulnerable_to",
        "leads_to", "exposes", "contains", "trusts", "compromises"
    ]
    target_id: str
    confidence: float = 1.0
    evidence_ids: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)


class EvidenceGraph(BaseModel):
    """
    Multi-step entity-relationship graph connecting discovered artifacts.
    Enables multi-hop attack path reasoning and cross-observation correlation.
    """
    entities: Dict[str, Entity] = Field(default_factory=dict)
    relationships: Dict[str, Relationship] = Field(default_factory=dict)

    def add_entity(self, entity: Entity) -> str:
        self.entities[entity.entity_id] = entity
        return entity.entity_id

    def add_relationship(self, rel: Relationship) -> str:
        self.relationships[rel.rel_id] = rel
        return rel.rel_id

    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)

    def find_entities_by_type(self, entity_type: str) -> List[Entity]:
        return [e for e in self.entities.values() if e.entity_type == entity_type]

    def get_neighbors(
        self,
        entity_id: str,
        direction: Literal["outgoing", "incoming", "both"] = "both"
    ) -> List[Tuple[Relationship, Entity]]:
        """Return connected (relationship, target_or_source_entity) tuples."""
        results = []
        for rel in self.relationships.values():
            if direction in ("outgoing", "both") and rel.source_id == entity_id:
                target = self.entities.get(rel.target_id)
                if target:
                    results.append((rel, target))
            elif direction in ("incoming", "both") and rel.target_id == entity_id:
                source = self.entities.get(rel.source_id)
                if source:
                    results.append((rel, source))
        return results

    def find_paths(
        self,
        source_id: str,
        target_id: str,
        max_depth: int = 5
    ) -> List[List[Relationship]]:
        """
        Find all directed paths of relationships connecting source_id to target_id.
        Useful for tracing privilege escalation and lateral movement chains.
        """
        paths: List[List[Relationship]] = []

        def dfs(curr: str, target: str, visited: Set[str], current_path: List[Relationship], depth: int):
            if depth > max_depth:
                return
            if curr == target and current_path:
                paths.append(list(current_path))
                return
            for rel in self.relationships.values():
                if rel.source_id == curr and rel.target_id not in visited:
                    visited.add(rel.target_id)
                    current_path.append(rel)
                    dfs(rel.target_id, target, visited, current_path, depth + 1)
                    current_path.pop()
                    visited.remove(rel.target_id)

        dfs(source_id, target_id, {source_id}, [], 0)
        return paths

    def correlate_cross_observations(
        self,
        obs_id_a: str,
        obs_id_b: str
    ) -> List[Dict[str, Any]]:
        """
        Identify entities and relationships that connect two distinct observations.
        """
        ents_a = {e.entity_id for e in self.entities.values() if obs_id_a in e.source_observation_ids}
        ents_b = {e.entity_id for e in self.entities.values() if obs_id_b in e.source_observation_ids}

        correlations = []
        for ea in ents_a:
            for eb in ents_b:
                if ea == eb:
                    correlations.append({
                        "type": "shared_entity",
                        "entity_id": ea,
                        "entity": self.entities[ea].model_dump()
                    })
                else:
                    paths = self.find_paths(ea, eb, max_depth=3)
                    for p in paths:
                        correlations.append({
                            "type": "attack_path",
                            "source_entity": self.entities[ea].model_dump(),
                            "target_entity": self.entities[eb].model_dump(),
                            "path": [r.model_dump() for r in p]
                        })
        return correlations

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entities": {k: v.model_dump() for k, v in self.entities.items()},
            "relationships": {k: v.model_dump() for k, v in self.relationships.items()},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceGraph:
        graph = cls()
        for k, v in data.get("entities", {}).items():
            graph.entities[k] = Entity(**v)
        for k, v in data.get("relationships", {}).items():
            graph.relationships[k] = Relationship(**v)
        return graph


class EvidenceStore(BaseModel):
    """
    In-memory and persistent store of all assessment intelligence entities.
    Tracks observations, hypotheses, validations, findings, and the unified evidence graph.
    """
    observations: Dict[str, Observation] = Field(default_factory=dict)
    hypotheses: Dict[str, Hypothesis] = Field(default_factory=dict)
    validations: Dict[str, Validation] = Field(default_factory=dict)
    findings: Dict[str, Finding] = Field(default_factory=dict)
    refuted_hypotheses: Dict[str, Hypothesis] = Field(default_factory=dict)
    graph: EvidenceGraph = Field(default_factory=EvidenceGraph)

    def add_observation(self, obs: Observation) -> str:
        self.observations[obs.obs_id] = obs
        return obs.obs_id

    def add_hypothesis(self, hyp: Hypothesis) -> str:
        self.hypotheses[hyp.hyp_id] = hyp
        return hyp.hyp_id

    def add_validation(self, val: Validation) -> str:
        self.validations[val.val_id] = val
        if val.hypothesis_id in self.hypotheses:
            hyp = self.hypotheses[val.hypothesis_id]
            if val.verdict == "supports":
                hyp.status = "confirmed"
            elif val.verdict == "refutes":
                hyp.status = "refuted"
                self.refuted_hypotheses[hyp.hyp_id] = hyp
            else:
                hyp.status = "inconclusive"
        return val.val_id

    def can_create_finding(self, hypothesis_id: str) -> Tuple[bool, str]:
        """
        Framework-enforced gate: only create findings with a confirmed hypothesis
        and at least one supporting validation test.
        """
        if hypothesis_id not in self.hypotheses:
            return False, f"Hypothesis '{hypothesis_id}' does not exist in evidence store."

        hyp = self.hypotheses[hypothesis_id]
        if hyp.status != "confirmed":
            return False, f"Hypothesis '{hypothesis_id}' status is '{hyp.status}', expected 'confirmed'."

        supporting_validations = [
            v for v in self.validations.values()
            if v.hypothesis_id == hypothesis_id and v.verdict == "supports"
        ]
        if not supporting_validations:
            return False, f"No supporting validation found for hypothesis '{hypothesis_id}'."

        return True, "OK"

    def record_finding(
        self,
        hypothesis_id: str,
        title: str,
        description: str,
        severity: Literal["info", "low", "medium", "high", "critical"],
        affected_target: str,
        evidence_summary: str,
        reproducible_steps: Optional[List[str]] = None,
        impact: str = "pending further validation",
        confidence: float = 1.0,
    ) -> Tuple[Optional[Finding], str]:
        """
        Attempt to create a Finding. Refuses if evidence chain is incomplete.
        """
        can_create, reason = self.can_create_finding(hypothesis_id)
        if not can_create:
            return None, reason

        hyp = self.hypotheses[hypothesis_id]
        supporting_vals = [
            v.val_id for v in self.validations.values()
            if v.hypothesis_id == hypothesis_id and v.verdict == "supports"
        ]

        finding = Finding(
            hypothesis_id=hypothesis_id,
            observation_ids=list(hyp.observation_ids),
            validation_ids=supporting_vals,
            title=title,
            description=description,
            severity=severity,
            affected_target=affected_target,
            evidence_summary=evidence_summary,
            reproducible_steps=reproducible_steps or [],
            impact=impact,
            confidence=confidence,
        )
        self.findings[finding.finding_id] = finding
        return finding, "Finding recorded successfully with verified evidence chain."

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": {k: v.model_dump() for k, v in self.observations.items()},
            "hypotheses": {k: v.model_dump() for k, v in self.hypotheses.items()},
            "validations": {k: v.model_dump() for k, v in self.validations.items()},
            "findings": {k: v.model_dump() for k, v in self.findings.items()},
            "refuted_hypotheses": {k: v.model_dump() for k, v in self.refuted_hypotheses.items()},
            "graph": self.graph.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> EvidenceStore:
        store = cls()
        for k, v in data.get("observations", {}).items():
            store.observations[k] = Observation(**v)
        for k, v in data.get("hypotheses", {}).items():
            store.hypotheses[k] = Hypothesis(**v)
        for k, v in data.get("validations", {}).items():
            store.validations[k] = Validation(**v)
        for k, v in data.get("findings", {}).items():
            store.findings[k] = Finding(**v)
        for k, v in data.get("refuted_hypotheses", {}).items():
            store.refuted_hypotheses[k] = Hypothesis(**v)
        if "graph" in data:
            store.graph = EvidenceGraph.from_dict(data["graph"])
        return store
