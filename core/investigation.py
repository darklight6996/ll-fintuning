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
    # MED-7 fix: tag each Observation with the phase of the action that produced it
    # (e.g. "recon", "enum", "privesc" -- whatever the agent's ActionDefinition.phase
    # says). Previously unset on every Observation, which made it impossible to
    # group or filter evidence by phase later (e.g. a future deterministic
    # Hypothesis-promotion strategy keyed on phase transitions). Optional and
    # defaulted so existing persisted sessions / callers that don't pass it keep
    # working unchanged.
    phase: Optional[str] = None


class Component(BaseModel):
    name: str
    description: str = ""
    endpoints: List[str] = Field(default_factory=list)
    technology: Optional[str] = None


class DataFlow(BaseModel):
    source: str
    sink: str
    data_type: str = ""
    authenticated: bool = False


class Assumption(BaseModel):
    assumption_id: str = Field(default_factory=lambda: f"asm_{uuid.uuid4().hex[:8]}")
    target_or_component: str
    description: str
    fragile: bool = True
    tested: bool = False
    invalidation_test: Optional[str] = None


class AppModel(BaseModel):
    """
    Working memory artifact representing the agent's mental model of the target:
    components, roles, trust boundaries, state/limited resources, and fragile assumptions.
    """
    purpose: str = ""
    components: List[Component] = Field(default_factory=list)
    roles: List[str] = Field(default_factory=list)
    trust_boundaries: List[str] = Field(default_factory=list)
    state_and_resources: List[str] = Field(default_factory=list)
    data_flows: List[DataFlow] = Field(default_factory=list)
    assumptions: List[Assumption] = Field(default_factory=list)

    def update(
        self,
        purpose: Optional[str] = None,
        components: Optional[List[Any]] = None,
        roles: Optional[List[str]] = None,
        trust_boundaries: Optional[List[str]] = None,
        state_and_resources: Optional[List[str]] = None,
        data_flows: Optional[List[Any]] = None,
        assumptions: Optional[List[Any]] = None,
    ) -> None:
        if purpose:
            self.purpose = purpose
        if roles:
            for r in roles:
                if r not in self.roles:
                    self.roles.append(r)
        if trust_boundaries:
            for tb in trust_boundaries:
                if tb not in self.trust_boundaries:
                    self.trust_boundaries.append(tb)
        if state_and_resources:
            for sr in state_and_resources:
                if sr not in self.state_and_resources:
                    self.state_and_resources.append(sr)
        if components:
            # BUG FIX (2026-11-25): roles/trust_boundaries/state_and_resources were
            # already deduplicated above, but components never was -- every
            # update(components=[...]) call unconditionally appended, so the same
            # component (e.g. re-recorded across several update_app_model steps)
            # accumulated once per call. Dedupe by name, same posture as the
            # list-of-str fields above.
            existing_names = {c.name for c in self.components}
            for c in components:
                if isinstance(c, dict):
                    comp = Component(**c)
                elif isinstance(c, Component):
                    comp = c
                elif isinstance(c, str):
                    comp = Component(name=c)
                else:
                    continue
                if comp.name in existing_names:
                    continue
                existing_names.add(comp.name)
                self.components.append(comp)
        if data_flows:
            for df in data_flows:
                if isinstance(df, dict):
                    self.data_flows.append(DataFlow(**df))
                elif isinstance(df, DataFlow):
                    self.data_flows.append(df)
        if assumptions:
            for asm in assumptions:
                if isinstance(asm, dict):
                    self.assumptions.append(Assumption(**asm))
                elif isinstance(asm, Assumption):
                    self.assumptions.append(asm)
                elif isinstance(asm, str):
                    self.assumptions.append(Assumption(target_or_component="target", description=asm))

    def summary(self) -> str:
        lines = []
        if self.purpose:
            lines.append(f"  Purpose: {self.purpose}")
        if self.components:
            comps = [f"{c.name}" + (f" ({', '.join(c.endpoints)})" if c.endpoints else "") for c in self.components]
            lines.append(f"  Components: {', '.join(comps)}")
        if self.roles:
            lines.append(f"  Roles: {', '.join(self.roles)}")
        if self.trust_boundaries:
            lines.append(f"  Trust Boundaries: {', '.join(self.trust_boundaries)}")
        if self.state_and_resources:
            lines.append(f"  State & Limited Resources: {', '.join(self.state_and_resources)}")
        if self.assumptions:
            lines.append("  Developer Assumptions:")
            for a in self.assumptions:
                status = "tested" if a.tested else ("FRAGILE" if a.fragile else "untested")
                lines.append(f"    - [{a.assumption_id}] ({status}) {a.target_or_component}: {a.description}")
        return "\n".join(lines) if lines else "  (no application model built yet)"


class Hypothesis(BaseModel):
    """
    A candidate explanation of observations. NOT yet a finding.
    """
    hyp_id: str = Field(default_factory=lambda: f"hyp_{uuid.uuid4().hex[:8]}")
    observation_ids: List[str] = Field(default_factory=list)
    description: str
    validation_test: str
    # Suspicion gate (2026-11-16): the vuln class this hypothesis is about
    # ("sqli"/"xss"/"ssrf"/"idor"/...), so the planner can unlock the matching
    # specialist action (see core.action_contract.action_is_unlocked). None = a
    # generic suspicion that unlocks only the generic weapons (self_script, etc.).
    vuln_class: Optional[str] = None
    # True only for the deterministic exit-0 'phase milestone' promotion (a benign
    # 'this recon action ran' marker), NOT a real vuln suspicion. The suspicion gate
    # ignores these so coverage never auto-unlocks offensive tools.
    milestone: bool = False
    confidence: Literal["speculative", "plausible", "likely"] = "plausible"
    status: Literal["open", "validating", "confirmed", "refuted", "inconclusive"] = "open"
    created_at: float = Field(default_factory=time.time)
    # Working memory link: optional assumption tested
    assumption_id: Optional[str] = None
    expected_control: Optional[str] = None
    expected_attack: Optional[str] = None
    # Workstream D-1 (#18): the stated, falsifiable PREMISE of this hypothesis
    # (e.g. "productIds 16-20 are hidden / beyond the public range"). Before a
    # MODEL-authored finding built on this premise can be promoted to "proven",
    # the premise is cross-checked against the store's recorded facts; a premise
    # the store CONTRADICTS demotes the finding to "lead". Absence of
    # corroboration is NOT contradiction. Defaults None (optional / best-effort).
    premise: Optional[str] = None


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
    # Confirmation strength (2026-11-20): "proven" = backed by concrete evidence of a
    # real vulnerability; "lead" = suspicious but UNVERIFIED (weak/heuristic oracle) --
    # surfaced for human verification, never counted as a solved finding; "milestone" =
    # a deterministic phase-progress marker, not a vulnerability. Completion counts
    # only "proven". Default "proven" keeps existing callers/tests unchanged.
    strength: Literal["proven", "lead", "milestone"] = "proven"


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
    app_model: AppModel = Field(default_factory=AppModel)
    # --sweep mode (class-sweep hunter, 2026-11-25): the ENUMERATE+CHARACTERIZE
    # surface map (one entry per endpoint/param observed, keyed by
    # "<method> <endpoint>::<param>") and the CLASS-SWEEP coverage ledger
    # (one entry per (vuln_class, endpoint_key) pair actually tested or
    # gated-in, keyed by "<vuln_class>::<endpoint_key>"). Pure bookkeeping --
    # neither ever creates a Finding; a sweep mode "proven" result still only
    # ever comes from the SAME engines/_confirmation_strength path every
    # other mode uses.
    surface_map: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    coverage_ledger: Dict[str, Dict[str, Any]] = Field(default_factory=dict)

    def add_surface_entry(self, endpoint_key: str, **fields: Any) -> Dict[str, Any]:
        """Upsert a surface-map entry (dedup by endpoint_key): later calls
        merge new tags/facts into the existing entry rather than duplicating
        it, so repeated characterization of the same endpoint/param across
        steps accumulates evidence instead of overwriting it."""
        entry = self.surface_map.setdefault(endpoint_key, {"endpoint_key": endpoint_key})
        for k, v in fields.items():
            if v is not None:
                entry[k] = v
        return entry

    def record_coverage(
        self, vuln_class: str, endpoint_key: str, status: str, engine: Optional[str] = None,
        note: Optional[str] = None,
    ) -> str:
        """Upsert a (vuln_class, endpoint) coverage-ledger entry. status is one
        of 'untested' | 'tested' | 'proven' | 'lead' | 'exhausted'
        | 'no_engine' -- the CONFIRM-OR-MOVE-ON bookkeeping the sweep spec
        asks for. Never itself promotes a Finding."""
        key = f"{vuln_class}::{endpoint_key}"
        self.coverage_ledger[key] = {
            "vuln_class": vuln_class, "endpoint_key": endpoint_key, "status": status,
            "engine": engine, "note": note,
        }
        return key

    def coverage_summary_lines(self) -> List[str]:
        """Human/model-readable render of the coverage ledger, stable order."""
        lines = []
        for key in sorted(self.coverage_ledger):
            c = self.coverage_ledger[key]
            tag = f" ({c['engine']})" if c.get("engine") else ""
            note = f" -- {c['note']}" if c.get("note") else ""
            lines.append(f"  [{c['status'].upper()}] {c['vuln_class']} @ {c['endpoint_key']}{tag}{note}")
        return lines

    def update_app_model(self, **kwargs) -> None:
        self.app_model.update(**kwargs)

    def get_app_model_summary(self) -> str:
        return self.app_model.summary()

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

    @staticmethod
    def _strength_rank(s: str) -> int:
        return {"milestone": 0, "lead": 1, "proven": 2}.get(s, 1)

    @staticmethod
    def _claim_key(hyp, title: str):
        """Identity of the CLAIM a finding makes, for dedup. A non-empty vuln_class
        is the claim key; otherwise fall back to the normalized title so two
        DISTINCT untyped findings are not collapsed, while a re-record of the same
        one is."""
        vc = (getattr(hyp, "vuln_class", None) or "").strip().lower() if hyp is not None else ""
        if vc:
            return ("class", vc)
        return ("title", " ".join((title or "").lower().split()))

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
        strength: str = "proven",
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

        # DEDUP AT THE STORE (workstream D-1 #25): no promotion path -- automatic
        # (_promote_evidence_if_warranted) or manual (_handle_record_finding) --
        # may create a duplicate Finding. Two Findings are the SAME claim iff their
        # observation_ids OVERLAP AND they represent the same claim (same
        # vuln_class; or, when neither has a class, the same normalized title).
        # Genuinely DISTINCT vulns evidenced by the same observation are NOT
        # collapsed -- dedup is on (overlapping obs AND same claim), never obs
        # alone -- so proven_finding_count counts one piece of evidence once.
        new_obs = set(hyp.observation_ids)
        new_key = self._claim_key(hyp, title)
        for existing in self.findings.values():
            if not (new_obs & set(existing.observation_ids)):
                continue
            existing_hyp = self.hypotheses.get(existing.hypothesis_id)
            if self._claim_key(existing_hyp, existing.title) != new_key:
                continue  # distinct claim on shared evidence -> both findings stand
            # Same claim + overlapping evidence -> MERGE into the existing Finding
            # instead of creating a second. Union the evidence/validation ids and
            # keep the STRONGER strength (a later 'proven' upgrades a prior 'lead').
            existing.observation_ids = list(dict.fromkeys(
                list(existing.observation_ids) + list(hyp.observation_ids)))
            existing.validation_ids = list(dict.fromkeys(
                list(existing.validation_ids) + supporting_vals))
            if self._strength_rank(strength) > self._strength_rank(existing.strength):
                existing.strength = strength if strength in ("proven", "lead", "milestone") else existing.strength
                existing.severity = severity
            return existing, "Duplicate finding merged into existing (same claim + overlapping evidence)."

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
            strength=strength if strength in ("proven", "lead", "milestone") else "proven",
        )
        self.findings[finding.finding_id] = finding
        return finding, "Finding recorded successfully with verified evidence chain."

    @property
    def proven_finding_count(self) -> int:
        """Count only PROVEN vulnerability findings -- not unverified leads, not
        phase-milestone markers. This is what completion criteria require, so a run
        can never declare success on a weak/heuristic confirmation."""
        return sum(1 for f in self.findings.values()
                   if getattr(f, "strength", "proven") == "proven")

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observations": {k: v.model_dump() for k, v in self.observations.items()},
            "hypotheses": {k: v.model_dump() for k, v in self.hypotheses.items()},
            "validations": {k: v.model_dump() for k, v in self.validations.items()},
            "findings": {k: v.model_dump() for k, v in self.findings.items()},
            "refuted_hypotheses": {k: v.model_dump() for k, v in self.refuted_hypotheses.items()},
            "graph": self.graph.to_dict(),
            "app_model": self.app_model.model_dump(),
            "surface_map": self.surface_map,
            "coverage_ledger": self.coverage_ledger,
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
        if "app_model" in data:
            store.app_model = AppModel(**data["app_model"])
        store.surface_map = dict(data.get("surface_map") or {})
        store.coverage_ledger = dict(data.get("coverage_ledger") or {})
        return store
