"""
agents/bug_bounty/command_builders.py

Pure Python command translators for BugBountyAgent actions.
Translates typed action parameter models into executable argument vectors.

SAFETY INVARIANTS:
  - Returns strictly List[List[str]]
  - Never shell strings
  - No shell=True
  - Cross-platform compatible
"""

import os
import sys
from typing import List

# Bundled fallback wordlist so dir_brute works out-of-the-box with no external
# wordlist package installed (the test box had none; ffuf then exits before even
# running). Resolved only when the caller's wordlist path does not exist.
_BUNDLED_WORDLIST = os.path.join(os.path.dirname(__file__), "wordlists", "common.txt")


def _resolve_wordlist(wordlist: str) -> str:
    """Use the caller's wordlist if it is an existing file; otherwise fall back
    to the repo-bundled common.txt so dir_brute never dies on a missing list."""
    if wordlist and os.path.isfile(wordlist):
        return wordlist
    return _BUNDLED_WORDLIST

from agents.bug_bounty.actions import (
    SubdomainEnumParams,
    PortScanParams,
    EndpointDiscoveryParams,
    DirBruteParams,
    ParamFuzzParams,
    HeaderAnalysisParams,
    IdorProbeParams,
    SqliProbeParams,
    XssProbeParams,
    VulnValidationParams,
    SqliDeepScanParams,
    NucleiScanParams,
    DalfoxScanParams,
)


def build_subdomain_enum(params: SubdomainEnumParams) -> List[List[str]]:
    """Build command vector for subdomain discovery."""
    return [
        ["subfinder", "-d", params.target, "-silent", "-timeout", str(params.timeout_seconds)]
    ]


def build_port_scan(params: PortScanParams) -> List[List[str]]:
    """Build command vector for port scanning web services."""
    return [
        ["nmap", "-sS", "-sV", "-p", params.ports, "--open", params.target]
    ]


# NOTE: every curl probe below passes -g (--globoff). The planner (a 7B model)
# sometimes emits placeholder/templated params like 'productId={}' or 'id=[0]';
# curl treats {} and [] as glob ranges and exits 3 ("URL malformed") before
# sending anything (observed live: xss_probe, 2026-11-14). -g makes those
# characters literal so a clumsy param string degrades to a normal request
# instead of a hard failure that burns the action's retry budget.

def build_endpoint_discovery(params: EndpointDiscoveryParams) -> List[List[str]]:
    """Build command vector to crawl links and endpoints."""
    return [
        ["curl", "-s", "-g", "-L", "-k", "--max-time", "15", params.target]
    ]


def build_dir_brute(params: DirBruteParams) -> List[List[str]]:
    """Build command vector for directory/endpoint fuzzing.

    ffuf v2 renamed '-silent' to '-s' (the old flag makes ffuf exit 2 with a
    usage dump, observed live 2026-11-14). The wordlist is resolved to a bundled
    fallback when the caller's path does not exist, so this works even with no
    wordlist package installed.
    """
    return [
        ["ffuf", "-u", f"{params.target.rstrip('/')}/FUZZ",
         "-w", _resolve_wordlist(params.wordlist),
         "-e", params.extensions,
         "-mc", "200,204,301,302,307,401,403", "-s"]
    ]


def build_param_fuzz(params: ParamFuzzParams) -> List[List[str]]:
    """Build command vector for parameter fuzzing."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-g", "-k", "-G", url, "-d", "test=probe"]
    ]


def build_header_analysis(params: HeaderAnalysisParams) -> List[List[str]]:
    """Build command vector for inspecting response headers and CORS."""
    return [
        ["curl", "-s", "-g", "-I", "-k", "-H", "Origin: https://evil.corp", params.target]
    ]


def build_idor_probe(params: IdorProbeParams) -> List[List[str]]:
    """Build command vector for testing unauthorized cross-tenant object access."""
    url_b = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}/{params.user_b_id}"
    return [
        ["curl", "-s", "-g", "-k", "-H", f"X-User-ID: {params.user_a_id}", url_b]
    ]


def build_sqli_probe(params: SqliProbeParams) -> List[List[str]]:
    """Build command vector for benign SQL syntax distortion probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-g", "-k", "-G", url, "--data-urlencode", f"{params.test_param}=' OR '1'='1"]
    ]


def build_xss_probe(params: XssProbeParams) -> List[List[str]]:
    """Build command vector for benign canary reflection probe."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-g", "-k", "-G", url, "--data-urlencode", f"{params.param_name}=canary_test_probe_xyz"]
    ]


def build_vuln_validation(params: VulnValidationParams) -> List[List[str]]:
    """Build command vector to confirm/refute vulnerability hypothesis."""
    url = f"{params.target.rstrip('/')}/{params.endpoint.lstrip('/')}"
    return [
        ["curl", "-s", "-g", "-k", "-v", url, "-d", params.payload]
    ]


def build_sqli_deep_scan(params: SqliDeepScanParams) -> List[List[str]]:
    """
    Build a DETECTION-ONLY, non-destructive sqlmap invocation.

    Every flag here is hardcoded and safety-reviewed -- the params model exposes
    no knobs for them, so the planner can never escalate this into a
    data-dumping or data-modifying run:

      --batch            Never prompt interactively (no hang in an unattended
                         run); takes sqlmap's safe default for every question.
      --level=2          Test GET/POST params + cookies-lite. Conservative.
      --risk=1           Lowest risk tier: excludes the OR-based (risk 2) and the
                         heavy stacked/time risk-3 payloads that can modify or
                         stress the DB. risk=1 payloads are read-only probes.
      --technique=BEUT   Boolean, Error, Union, Time-based blind. Deliberately
                         OMITS 'S' (stacked queries) -- the only technique that
                         can execute data-modifying statements.
      --time-sec=5       Time-based blind delay threshold (default).
      --timeout=15,      Per-request timeout + a single retry: keep one hung
      --retries=1        request from stalling the whole scan.
      --flush-session    Start clean each run so a stale cache never masks a
                         real result or reports a phantom one.
      --disable-coloring Plain ASCII output so the fact extractor parses cleanly.

    NOTHING here enumerates or exfiltrates data: there is no --dump, --dump-all,
    --sql-shell, --os-shell, --os-cmd, --file-read/-write, or --passwords. sqlmap
    with only these flags DETECTS injectability and reports it -- it does not pull
    rows. That keeps the action within state_change="none" even though it is an
    active network probe.
    """
    has_query = "?" in params.target
    tech = "BEU" if has_query else "BEUT"
    argv = [
        "sqlmap",
        "-u", params.target,
        "--batch",
        "--level=2",
        "--risk=1",
        f"--technique={tech}",
        "--time-sec=5",
        "--timeout=15",
        "--retries=1",
        "--flush-session",
        "--disable-coloring",
    ]
    if not has_query:
        # --crawl=2: when the target URL has no query parameter of its own (a bare
        # root like https://site/ -- exactly what the planner passed on the first
        # live run), sqlmap otherwise reports "no parameter(s) found for testing"
        # and tests NOTHING. Crawling depth 2 lets sqlmap discover the site's own
        # parameterized GET endpoints (e.g. /product?productId=1, /filter?category=)
        # and test those. Crawl stays on the target host, so it remains within the
        # scope the executor already locked. Deliberately NOT adding --forms: that
        # would submit POST forms (a state change); this action is state_change=
        # "none", GET-parameter detection only.
        argv += [
            "--crawl=2",
            "--crawl-exclude", "logout|sign-out|signout|delete",
        ]
    if params.test_parameter:
        argv += ["-p", params.test_parameter]
    return [argv]


def build_nuclei_scan(params: NucleiScanParams) -> List[List[str]]:
    """
    Build a DETECTION-ONLY nuclei invocation.

    Safe, hardcoded posture (no knobs exposed to the planner):
      -jsonl             Machine-parseable: one JSON object per match, so the
                         fact extractor reads a real verdict, not scraped text.
      -silent            Suppress banner/progress; only results on stdout.
      -severity low,...  Severity floor: skip 'info' templates (tech/version
                         fingerprinting) so a clean target is a true non-finding
                         rather than a pile of noise promoted to findings.
      -rate-limit 50     Modest pacing. (The real stealth layer -- jitter,
                         passive-first, WAF backoff -- is a later workstream;
                         this is a sane default so a lab scan isn't a flood.)
      -timeout 10,       Bound per-request time and retries so one slow host
      -retries 1         can't stall the whole scan.
      -no-color          Plain output for clean parsing.

    Deliberately NOT passing -dast or any fuzzing template flag: those send
    active mutation payloads (state-changing risk). Default nuclei templates are
    detection matchers, which keeps this state_change="none".

    (nuclei auto-downloads its template store to ~/.nuclei-templates on first
    run; that needs network and happens once.)
    """
    return [[
        "nuclei",
        "-u", params.target,
        "-jsonl",
        "-silent",
        "-severity", "low,medium,high,critical",
        "-rate-limit", "50",
        "-timeout", "10",
        "-retries", "1",
        "-no-color",
    ]]


def build_dalfox_scan(params: DalfoxScanParams) -> List[List[str]]:
    """
    Build a DETECTION-ONLY dalfox XSS scan.

    Safe, hardcoded flags:
      url <target>   Scan mode: mine the URL's parameters and test them for XSS.
      --format json  Machine-parseable findings (one JSON array/stream) so the
                     fact extractor reads a real verdict.
      --silence      Suppress the banner/progress chatter.
      --no-spinner   No TTY spinner (clean captured output).
      --skip-bav     Skip the "basic additional vulnerability" grep noise; keep
                     the run focused on XSS detection.
      --timeout 10   Per-request timeout.

    dalfox's payloads are reflection/DOM XSS probes (benign: they prove the sink,
    they don't persist or destroy), so this stays state_change="none". No
    --custom-payload, no blind-XSS callback host, no file output.
    """
    return [[
        "dalfox", "url", params.target,
        "--format", "json",
        "--silence",
        "--no-spinner",
        "--skip-bav",
        "--timeout", "10",
    ]]

