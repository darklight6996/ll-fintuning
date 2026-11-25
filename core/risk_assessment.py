"""
core/risk_assessment.py

Deterministic, rule-based risk scoring + dual-perspective (Owner/Hunter)
analysis, re-implemented from ARTEX's methodology (see work.md's 2026-11-23
gap analysis) against OUR EvidenceStore shape instead of ARTEX's Flask
scan_results dict. No cloud AI call of any kind -- the scoring and narrative
here are both plain Python, string templates driven by computed factors, so
the result is reproducible and auditable. This module never creates a
Finding and never touches the suspicion/approval/completion gates; it only
reads EvidenceStore state that those gates already produced.

Inputs:
  - EvidenceStore.findings (PROVEN and unproven/leads)
  - EvidenceStore.hypotheses (open candidate areas, for the Hunter view)
  - EvidenceStore.observations whose action_id == "recon_scan" (the breadth
    sweep's facts bundle -- see core/tools/recon_scan_runner.py +
    agents/common/recon_primitive.py)

Output: build_risk_report(evidence_store) -> a single dict consumed by
ll-finetuning/rag/core/report_generator.py's new report section.
"""

from typing import Any, Dict, List

# Keyword -> (bucket, points) for categorizing discovered subdomain/host
# names, re-implemented from ARTEX's FastSecurityAnalyst._load_security_patterns.
SUBDOMAIN_RISK_KEYWORDS: Dict[str, Dict[str, List[str]]] = {
    "critical": {
        "admin": ["admin", "administrator", "cpanel", "whm", "plesk"],
        "database": ["db", "database", "mysql", "phpmyadmin", "adminer", "mongodb", "redis"],
        "backup": ["backup", "bak", "old", "archive", "dump"],
        "internal": ["internal", "intranet", "private", "corp"],
    },
    "high": {
        "development": ["dev", "devel", "test", "testing", "qa", "staging", "stage"],
        "api": ["api", "rest", "graphql", "service"],
        "config": ["config", "configuration", "setup", "install"],
        "auth": ["auth", "login", "sso", "oauth"],
    },
    "medium": {
        "mail": ["mail", "email", "smtp", "webmail"],
        "monitoring": ["monitor", "grafana", "kibana"],
        "vpn": ["vpn", "remote", "gateway"],
    },
}


def _bucketize_subdomains(subdomains: List[str]) -> Dict[str, Dict[str, List[str]]]:
    buckets: Dict[str, Dict[str, List[str]]] = {level: {cat: [] for cat in cats}
                                                 for level, cats in SUBDOMAIN_RISK_KEYWORDS.items()}
    for name in subdomains or []:
        lowered = name.lower()
        for level, categories in SUBDOMAIN_RISK_KEYWORDS.items():
            for category, keywords in categories.items():
                if any(kw in lowered for kw in keywords):
                    buckets[level][category].append(name)
    return buckets


def collect_recon_facts(evidence_store: Any) -> Dict[str, Any]:
    """Merge every recon_scan Observation's parsed_facts["recon"] bundle into
    one dict. Multiple recon_scan calls are merged additively (lists unioned,
    dicts last-write-wins)."""
    merged: Dict[str, Any] = {"ports": [], "auth_pages": [], "subdomains": [], "tech": []}
    observations = getattr(evidence_store, "observations", {}) or {}
    for obs in observations.values():
        if getattr(obs, "action_id", None) != "recon_scan":
            continue
        recon = (getattr(obs, "parsed_facts", {}) or {}).get("recon")
        if not isinstance(recon, dict):
            continue
        for key in ("ports", "auth_pages", "subdomains", "tech"):
            val = recon.get(key)
            if isinstance(val, list):
                for item in val:
                    if item not in merged[key]:
                        merged[key].append(item)
        for key in ("dns", "tls"):
            val = recon.get(key)
            if isinstance(val, dict):
                merged[key] = val
    return merged


def assess_risk(evidence_store: Any, recon_facts: Dict[str, Any]) -> Dict[str, Any]:
    """Additive 0-100 risk score (re-implemented from ARTEX's
    _assess_security_risk), fed by proven/lead findings plus the recon
    sweep's surface facts."""
    score = 0
    factors: List[str] = []

    findings = getattr(evidence_store, "findings", {}) or {}
    proven = [f for f in findings.values() if getattr(f, "strength", "proven") == "proven"]
    # A single PROVEN critical finding alone should already read as at least
    # High risk (>=50) -- these weights are deliberately steep, not linear
    # with ARTEX's subdomain-pattern weights below (a proven exploit is a much
    # stronger signal than a suspicious hostname).
    severity_points = {"critical": 55, "high": 30, "medium": 15, "low": 5, "info": 1}
    for f in proven:
        pts = severity_points.get(getattr(f, "severity", "medium"), 8)
        score += pts
    if proven:
        factors.append(f"{len(proven)} proven finding(s) ({', '.join(sorted({f.severity for f in proven}))})")

    open_ports = recon_facts.get("ports", []) or []
    risky_open = [p for p in open_ports if isinstance(p, dict) and p.get("risk_level") in ("high", "critical")]
    if risky_open:
        score += 10 * len(risky_open)
        factors.append(f"{len(risky_open)} high/critical-risk open port(s): "
                        f"{', '.join(str(p.get('port')) for p in risky_open)}")
    if len(open_ports) > 10:
        score += 10
        factors.append(f"Large open-port surface: {len(open_ports)} ports")

    tls = recon_facts.get("tls", {}) or {}
    if tls.get("self_signed"):
        score += 10
        factors.append("Self-signed TLS certificate")
    if tls.get("expired"):
        score += 15
        factors.append("Expired TLS certificate")
    if tls.get("hsts") is False:
        score += 8
        factors.append("HSTS not enabled")
    missing_headers = tls.get("missing_security_headers") or []
    if len(missing_headers) >= 3:
        score += 6
        factors.append(f"{len(missing_headers)} recommended security headers missing")

    auth_pages = recon_facts.get("auth_pages", []) or []
    if auth_pages:
        score += 5
        factors.append(f"{len(auth_pages)} authentication surface page(s) discovered")

    subdomain_buckets = _bucketize_subdomains(recon_facts.get("subdomains", []))
    critical_total = sum(len(v) for v in subdomain_buckets.get("critical", {}).values())
    high_total = sum(len(v) for v in subdomain_buckets.get("high", {}).values())
    if critical_total:
        score += critical_total * 15
        factors.append(f"Critical-pattern subdomains exposed: {critical_total}")
    if high_total:
        score += high_total * 10
        factors.append(f"High-risk-pattern subdomains exposed: {high_total}")

    score = min(score, 100)
    if score >= 80:
        level = "Critical"
    elif score >= 50:
        level = "High"
    elif score >= 25:
        level = "Medium"
    else:
        level = "Low"

    return {
        "score": score,
        "level": level,
        "factors": factors,
        "critical_assets": critical_total,
        "total_risk_assets": critical_total + high_total,
        "subdomain_buckets": subdomain_buckets,
    }


def owner_perspective(evidence_store: Any, recon_facts: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    """Defensive summary: what's exposed, how to fix it. Plain rule-based
    text, no cloud call."""
    findings = getattr(evidence_store, "findings", {}) or {}
    proven = [f for f in findings.values() if getattr(f, "strength", "proven") == "proven"]

    key_findings = [f"{f.title} ({f.severity})" for f in sorted(
        proven, key=lambda f: {"critical": 4, "high": 3, "medium": 2, "low": 1, "info": 0}.get(f.severity, 0),
        reverse=True)][:5]

    protection_recommendations = []
    if recon_facts.get("ports"):
        risky = [p for p in recon_facts["ports"] if p.get("risk_level") in ("high", "critical")]
        if risky:
            risky_labels = [f"{p['port']}/{p['service']}" for p in risky]
            protection_recommendations.append(
                f"Close or firewall unneeded high-risk services: {', '.join(risky_labels)}.")
    tls = recon_facts.get("tls", {}) or {}
    if tls.get("hsts") is False:
        protection_recommendations.append("Enable HTTP Strict Transport Security (HSTS).")
    if tls.get("missing_security_headers"):
        protection_recommendations.append(
            f"Add missing security headers: {', '.join(tls['missing_security_headers'])}.")
    if tls.get("self_signed") or tls.get("expired"):
        protection_recommendations.append("Replace the TLS certificate (self-signed or expired).")
    if recon_facts.get("auth_pages"):
        protection_recommendations.append(
            "Review discovered authentication surfaces for rate limiting, lockout, and MFA.")
    if not protection_recommendations:
        protection_recommendations.append("No immediate infrastructure hardening gaps identified by the recon sweep.")

    immediate_actions = [f"Validate and remediate: {f.title}" for f in proven[:3]]
    if not immediate_actions:
        immediate_actions = ["No proven vulnerabilities yet -- continue the assessment before declaring done."]

    summary = (
        f"Overall risk is assessed as {risk['level']} ({risk['score']}/100) based on "
        f"{len(proven)} proven finding(s) and the breadth-first recon sweep's surface map."
    )

    return {
        "summary": summary,
        "key_findings": key_findings,
        "protection_recommendations": protection_recommendations,
        "immediate_actions": immediate_actions,
    }


def hunter_perspective(evidence_store: Any, recon_facts: Dict[str, Any], risk: Dict[str, Any]) -> Dict[str, Any]:
    """Offensive summary: ranked promising targets + why, from the same facts
    plus still-open hypotheses. Plain rule-based text, no cloud call."""
    promising_targets = []

    for p in recon_facts.get("ports", []):
        if p.get("risk_level") in ("high", "critical"):
            promising_targets.append({
                "target": f"port {p['port']} ({p['service']})",
                "reason": p.get("note") or f"{p['risk_level']}-risk service exposed",
            })
    for page in recon_facts.get("auth_pages", [])[:5]:
        promising_targets.append({
            "target": page.get("url", "unknown"),
            "reason": f"discovered {page.get('type', 'auth')} page -- test for brute force / enumeration",
        })

    hypotheses = getattr(evidence_store, "hypotheses", {}) or {}
    open_hyps = [h for h in hypotheses.values() if getattr(h, "status", "open") == "open" and not h.milestone]
    for h in open_hyps[:5]:
        promising_targets.append({
            "target": h.description[:80],
            "reason": f"open hypothesis (vuln_class={h.vuln_class or 'unclassified'}), not yet tested",
        })

    attack_vectors = []
    buckets = risk.get("subdomain_buckets", {})
    for level in ("critical", "high"):
        for category, names in buckets.get(level, {}).items():
            if names:
                attack_vectors.append({"type": f"{level}:{category}", "description": f"{len(names)} target(s)"})

    summary = (
        f"{len(promising_targets)} candidate target(s) surfaced from the recon sweep and open "
        f"hypotheses; {len(attack_vectors)} subdomain attack-vector categor(y/ies) flagged."
    )

    return {
        "summary": summary,
        "promising_targets": promising_targets[:10],
        "attack_vectors": attack_vectors[:10],
    }


def build_risk_report(evidence_store: Any) -> Dict[str, Any]:
    """Top-level entry point: everything report_generator.py needs for the
    new 'Risk Assessment & Dual-Perspective Analysis' section."""
    recon_facts = collect_recon_facts(evidence_store)
    risk = assess_risk(evidence_store, recon_facts)
    return {
        "risk_assessment": risk,
        "recon_facts": recon_facts,
        "owner_analysis": owner_perspective(evidence_store, recon_facts, risk),
        "hunter_analysis": hunter_perspective(evidence_store, recon_facts, risk),
    }
