"""
core/campaign_judgment.py

Standing judgment notes for the AI planner: which findings are rejectable
on their own vs. which chain into something payable, the per-vuln-class
"does the evidence match the claim" checks and devil's-advocate framing
that catch an optimistic over-claim before it's recorded, and the
operational discipline lessons that most often separate a real confirmed
bug from a wasted step. Re-implemented in our own words from the GENERAL
METHODOLOGY seen in the pentest-agents project (github.com/H-mmer/pentest-agents) --
that repo ships no LICENSE file (its pyproject.toml MIT classifier is not
backed by any actual license grant or text anywhere in the tree), so this
is a from-scratch re-authoring of the underlying ideas against OUR data
shapes and vocabulary, not a port of its source. See work.md's 2026-11-25
harvest-plan entry for the full attribution and rationale.

This module renders ONE compact, STATIC prompt section (no retrieval, no
query) -- unlike core/technique_advisor.py's RAG lookup, these notes apply
on every single turn regardless of what's being investigated, the same way
a human reviewer would carry a standing checklist into every finding they
write up. Intentionally static and short: this is judgment calibration,
not a payload library (that lives in the RAG/YAML scenario corpus).

GRADING NOTE: nothing here changes _confirmation_strength() or the
proven/lead boundary in agents/base_agent.py. This is advisory prompt text
that helps the model avoid proposing a weak completion or a low-value
hypothesis in the first place -- the framework's own rigor gate still
makes the real decision regardless of what this text says.
"""

from typing import List

# Patterns that, standing alone with no further chain, are essentially
# never worth promoting to a real finding -- re-authored from the general
# "which bugs are rejectable alone" methodology common across bug-bounty
# triage guidance (not any one source's exact wording/table).
WEAK_ALONE_PATTERNS: List[str] = [
    "A missing security header (CSP/HSTS/X-Frame-Options) with no PoC of an actual exploit built on top of it",
    "Missing SPF/DKIM/DMARC records alone",
    "GraphQL/API schema introspection being enabled, with no auth-bypass or data-exposure demonstrated through it",
    "A software version/banner disclosure with no matching exploit actually run against that version",
    "Clickjacking with no sensitive state-changing action framed and triggered",
    "Self-XSS (the payload only fires against the attacker's own session)",
    "An open redirect with no further chain (e.g. into an OAuth flow) demonstrated",
    "Blind/DNS-only SSRF with no internal data or service actually reached",
    "A permissive CORS policy with no credentialed cross-origin data theft demonstrated",
    "CSRF on a logout action, or on a form with no meaningful state change",
    "A rate limit missing on a non-critical, non-authentication form",
    "An internal IP or stack trace fragment appearing in an error message, with nothing further extracted",
]

# Capability -> what to look for next, so a confirmed low-severity bug is
# evaluated for an escalation path before the investigation calls it done,
# instead of being reported (or dismissed) in isolation. Condensed and
# re-authored from the general "chain confirmed primitives into higher
# impact" methodology; this is a representative subset, not an exhaustive
# table, and intentionally mirrors the vocabulary the AppModel/Hypothesis
# layer already uses (observations, assumptions, confirmed primitives).
CHAIN_HINTS: List[str] = [
    "JS execution in a victim's browser -> check whether the session cookie lacks HttpOnly (session theft), "
    "whether a CSRF token is readable from that context (forge authenticated actions), or whether a "
    "postMessage listener trusts unchecked origins (control over app state).",
    "Arbitrary text/markup injection into a response -> check whether it is later evaluated/executed "
    "elsewhere (template injection -> RCE), rendered again in a DIFFERENT user's context (stored XSS "
    "against other users, not just yourself), or forwarded into a backend API call (injection there too).",
    "Control over a redirect target or URL parameter -> check whether it feeds an OAuth/OIDC "
    "redirect_uri (authorization-code theft -> account takeover), or is reachable from a trusted-looking "
    "link (phishing with a legitimate domain in the address bar).",
    "An IDOR/BAC primitive confirmed on one endpoint -> test the SIBLING endpoints, methods, and fields "
    "(a confirmed object-reference bug on GET is frequently also exploitable on PUT/PATCH/DELETE, or on a "
    "near-identical endpoint that was never individually tested).",
    "SSRF reaching an internal hostname/IP -> check whether it can reach a cloud metadata service or an "
    "internal admin interface with actual data in the response, not just a differing status code.",
    "A leaked or guessable secret/token -> check what it actually authorizes (a read-only API key is a "
    "much smaller finding than one that authorizes write/delete actions) before grading its severity.",
]

# A short, framework-relevant subset of hard-won operational lessons.
# Re-authored, not copied -- and deliberately filtered down to the ones that
# map onto THIS framework's single-sequential-loop architecture (no parallel
# subagent dispatch, no cross-session memory files to index), so this isn't
# generic advice that doesn't apply here.
OPERATIONAL_LESSONS: List[str] = [
    "A gate is not the goal. If satisfying a completion/coverage requirement would mean writing a "
    "marker instead of doing the work that marker is supposed to represent, the marker is invalid -- "
    "do the work or let the gate correctly block completion.",
    "\"Confirmed\" means a working probe against the live target produced a real, distinctive effect -- "
    "not a status-code difference, not a fingerprint match, not an inferred chain. Anything short of "
    "that is a lead, not a finding, no matter how confident the reasoning sounds.",
    "Demonstrate impact with the actual data/effect captured, not hypothetical language. \"This could "
    "allow an attacker to...\" is a sign the probe never actually confirmed anything; \"here is the data "
    "the probe retrieved\" is a finding.",
    "Test the sibling surface before concluding a hypothesis is dead. A vulnerability class refuted on "
    "one endpoint/parameter/method is not refuted on the adjacent ones -- a meaningful share of real "
    "access-control bugs are found on a sibling route that was never individually tested.",
    "A WAF/CDN block (403, challenge page) is not evidence the target is unaffected -- it means the "
    "probe never actually reached the application logic. Route around it (a real browser, a different "
    "header set, a different path) before concluding a hypothesis is refuted.",
    "If the exact same error or response shows up for both the control and the attack payload, that is "
    "a sign the TEST ITSELF is malformed (wrong content-type, wrong encoding, wrong method) -- not "
    "evidence the hypothesis is false. Fix the test before recording a refutation.",
]


# Per-vuln-class "does the evidence actually match the claim" checks --
# re-authored from the general adversarial-review / severity-recalibration
# methodology seen in pentest-agents' validator and devil's-advocate review
# roles (same license caveat as above: re-authored, not copied). These are
# the specific questions that catch an optimistic over-claim BEFORE it is
# recorded, complementing (not replacing) _confirmation_strength()'s own
# structural WEAK-signal rejection in agents/base_agent.py.
PER_CLASS_IMPACT_CHECKS: List[str] = [
    "XSS: was it actually confirmed executing in a real browser/DOM, or only inferred from a curl "
    "reflection? A reflected payload in raw HTML is a lead for XSS, not a confirmed one -- real "
    "execution (browser_exec, a console marker, an actual dialog) is what makes it proven.",
    "IDOR: does the response contain another user's SENSITIVE data (email, phone, payment, private "
    "content), or only a display name / already-public field? A 200 OK alone proves nothing without "
    "reading what's actually in the body.",
    "SSRF: did the probe retrieve real data from the internal target (a response body, cloud metadata "
    "contents), or only observe a timing/DNS callback? A DNS-only callback is blind SSRF -- real, but a "
    "lower-impact variant than one that returns actual internal data.",
    "Race condition: was the underlying STATE actually violated (the single-use resource shows consumed "
    "twice, the balance is inconsistent with one legitimate deduction), or did N requests just each "
    "return a 2xx? Counting '2xx responses' is not the same as checking the resulting state.",
    "Auth bypass: does removing/forging the credential truly grant access to protected data, or does the "
    "endpoint return an empty/different body that merely LOOKS like success? Compare the actual response "
    "content, not just the status code.",
]

# The devil's-advocate framing: before recording a finding, actively try to
# argue it is WEAKER than first thought, not just confirm it looks right.
DEVILS_ADVOCATE_FRAMING: str = (
    "Before recording a finding as proven, argue AGAINST it for a moment: replay the exact probe once "
    "more and confirm it still reproduces; check whether the 'leaked' data is already visible without "
    "authentication at all (not a leak if so); and ask whether a less dramatic explanation (a different "
    "error path, an already-public field, a patched endpoint) fits the evidence just as well. A finding "
    "that survives this check is worth recording; one that doesn't should be downgraded to a lead or "
    "dropped, not inflated."
)


def render_campaign_judgment(max_chain_hints: int = 4, max_lessons: int = 4, max_impact_checks: int = 4) -> str:
    """
    Render a compact, bounded prompt section. Capped (not the full lists) so
    this stays a short standing reminder, not a wall of text competing with
    the state-specific sections of the prompt for the model's attention.
    """
    lines: List[str] = []
    lines.append("Before proposing propose_completion or record_finding, weigh these:")
    lines.append("")
    lines.append("NOT worth a finding on their own (reject unless chained into something real):")
    for item in WEAK_ALONE_PATTERNS[:6]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("Before calling a confirmed primitive \"done\", check if it chains further:")
    for item in CHAIN_HINTS[:max_chain_hints]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append("Does the evidence actually match the claimed impact?")
    for item in PER_CLASS_IMPACT_CHECKS[:max_impact_checks]:
        lines.append(f"  - {item}")
    lines.append("")
    lines.append(f"Devil's advocate: {DEVILS_ADVOCATE_FRAMING}")
    lines.append("")
    lines.append("Operational discipline:")
    for item in OPERATIONAL_LESSONS[:max_lessons]:
        lines.append(f"  - {item}")
    return "\n".join(lines)
