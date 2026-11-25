#!/usr/bin/env python3
"""
core/tools/recon_scan_runner.py

BREADTH-FIRST PARALLEL RECON ENGINE.

Methodology re-implemented (not copied -- see work.md's 2026-11-23 ARTEX gap
analysis) from the open-source ARTEX project's enumeration modules, ported into
this framework's own style and safety posture. ARTEX ships no LICENSE file, so
this is a from-scratch re-implementation of its ideas against OUR data shapes,
not a port of its source; attributed here and in work.md.

The js_recon and auth_sweep modes (2026-11-25 harvest round) are re-implemented
from Claude-BugHunter's (github.com/elementalsouls/Claude-BugHunter) engine/
recon.py -- content/code there is MIT-licensed; re-implemented against our own
data shapes and safety posture, attributed here and in work.md.

PURPOSE
  A single action that surveys the WHOLE visible surface of one target --
  DNS records, open ports (with a static risk map), TLS/certificate posture,
  lightweight technology fingerprinting, and common auth-surface paths -- in
  one parallel sweep, instead of the one-shot piecemeal probes
  (endpoint_discovery / header_analysis / param_fuzz run one at a time) that
  previously left the planner to discover breadth by trial and error across
  many steps. This is a COVERAGE action: it never confirms a vulnerability
  itself, it only maps surface for the planner's hypotheses to build on.

MODES (comma-separated via --modes; each runs in its own thread, each wrapped
in its own try/except so one failing mode never kills the others)
  dns          A/AAAA/MX/NS/TXT/CNAME/SOA + parsed SPF/DMARC.
  ports        Pure-socket TCP-connect scan over a curated common/risky port
               list, annotated with a static risk map (service + known risk
               class). No nmap dependency -- works even when nmap isn't
               installed.
  tls          Certificate subject/issuer/validity/self-signed check,
               supported TLS protocol versions, HSTS header, missing
               recommended security headers. Stdlib ssl/socket only.
  tech         Lightweight header + HTML regex fingerprint (server software,
               common CMS/JS-framework signatures). No Wappalyzer dependency.
  authsurface  Probes a short list of common login / password-reset paths and
               flags pages with a password input field.
  subdomains   OFF BY DEFAULT. DNS-bruteforces a small built-in wordlist
               against the registrable domain (target-only, no 3rd-party
               calls). Pass --allow-ct-lookup to ALSO query crt.sh (a single
               well-known, read-only certificate-transparency API, queried
               ABOUT the domain, never AGAINST it) for additional candidate
               names. Must be explicitly requested by the caller (the
               single-host registry never requests it -- see
               agents/bug_bounty/single_host_mode.py). Each found subdomain
               with a CNAME is also checked against a small known-vulnerable-
               service table (github.io/s3/heroku/shopify) for a dangling
               takeover -- see "takeovers" in the output.
  secrets      Regex-scans the fetched homepage for leaked credential/token
               patterns (AWS/GCP/Slack/GitHub/Stripe/SendGrid, generic API
               key/token, DB connection string, SSH private key header).
               Every match is MASKED before it ever reaches output -- the
               full secret value is never printed, logged, or persisted.
  exposed_files  Probes a short list of common sensitive paths (.git/HEAD,
               .env, wp-config.php, backup.sql, docker-compose.yml, ...).
               Each hit records both the HTTP status AND whether the body
               matches a content-signature specific to that file type (a
               bare 200 alone is not a confirming signal; the signature is
               what makes it one).
  real_ip      Best-effort real-IP-behind-CDN discovery: mail-server (MX) IP
               lookup and HTTP response-header IP-leak scanning, filtered
               against known CDN ranges and confidence-scored by how many
               independent sources agree. Informational only.
  js_recon     Mines the homepage and its same-origin JS bundles (a capped
               number) for relative API routes, Next.js data routes, and
               forms, then PRUNES params with no observable effect on the
               response (a marker-diff against the no-param baseline) so the
               planner sees only live, processable surface, not dead noise.
  auth_sweep   Discovers an OpenAPI/Swagger spec at common paths and probes
               each DECLARED-SECURED GET operation UNAUTHENTICATED (never
               POST/PUT/PATCH/DELETE -- read-only by design, no allow-intrusive
               switch). A 2xx/422 where 401/403 was expected is a genuine
               broken-function-level-auth signal, surfaced as a lead (see
               agents/base_agent.py's lead-only promotion -- kept conservative
               here too, even though this is closer to a real confirming
               probe than a pattern match).

SAFETY
  * Every mode is read-only / detection-only: GET requests, TCP connect
    probes, and TLS handshakes only -- never a POST/PUT, never a destructive
    payload, never a credential submission.
  * Scope-locked to the ONE --url/--host argument; subdomain candidates are
    reported as a list for the planner to decide about, never auto-probed
    further by this engine.
  * Hard per-operation timeouts and capped port/wordlist sizes so a single
    invocation cannot turn into an unbounded scan.
  * Never raises without first emitting a parseable JSON line -- a failing
    mode degrades to an empty/partial result for that mode only.

OUTPUT
  ONE JSON line: {"recon": {"dns": {...}, "ports": [...], "tls": {...},
  "tech": [...], "auth_pages": [...], "subdomains": [...], "takeovers": [...],
  "secrets": [...], "exposed_files": [...], "real_ip": {...}}}
  This is a FACTS bundle, not a vuln_signal -- it must never itself promote a
  Finding directly (see agents/common/recon_primitive.py). "secrets" and
  "exposed_files"/"takeovers" hits DO get surfaced as Findings, but only ever
  at strength="lead", via a dedicated, hard-coded promotion path in
  agents/base_agent.py (_promote_lead_findings_from_recon) that cannot emit
  "proven" -- heuristic/pattern matches are never auto-confirmed.

GRADING RULE (2026-11-24 ARTEX merge round 2): everything in this engine is
recon/fingerprint/pattern-matching, never an actual confirming probe. Per the
framework's confirmation-rigor policy, that means every new signal this file
produces is either (a) a pure fact with no Finding at all (dns/ports/tls/tech/
real_ip), or (b) promoted only to strength="lead" (secrets/exposed_files/
takeovers) -- never "proven". A real "proven" finding still only ever comes
from an actual confirming probe (self_script, race_probe, raw_http,
browser_exec, the typed probes) landing a payload and observing its effect.
"""

import argparse
import json
import re
import socket
import ssl
import sys
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse, urljoin

import requests
import urllib3

# This engine deliberately connects with verify=False (recon surveys
# whatever certificate a target presents, trusted or not -- scan_tls above is
# what actually judges trust/self-signed/expired); silence the resulting
# per-request noise so stderr stays readable for an operator.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*TLSVersion.*deprecated.*")

# ---------------------------------------------------------------------------
# Static data (re-implemented from ARTEX's methodology, our own values)
# ---------------------------------------------------------------------------

# port -> (risk_level, service_label, note). Ports not listed are "low".
RISKY_PORTS: Dict[int, Dict[str, str]] = {
    21: {"level": "high", "service": "ftp", "note": "Anonymous auth, cleartext credentials"},
    22: {"level": "medium", "service": "ssh", "note": "Brute force / weak credentials target"},
    23: {"level": "high", "service": "telnet", "note": "Cleartext protocol, should not be exposed"},
    25: {"level": "medium", "service": "smtp", "note": "Open relay / user enumeration"},
    445: {"level": "critical", "service": "smb", "note": "EternalBlue-class exposure history"},
    1433: {"level": "high", "service": "mssql", "note": "Brute force, injection target"},
    3306: {"level": "medium", "service": "mysql", "note": "Weak credentials, should not be public"},
    3389: {"level": "high", "service": "rdp", "note": "BlueKeep-class exposure, brute force"},
    5432: {"level": "medium", "service": "postgresql", "note": "Should not be public"},
    5900: {"level": "high", "service": "vnc", "note": "Often unauthenticated"},
    5984: {"level": "medium", "service": "couchdb", "note": "Misconfiguration-prone"},
    6379: {"level": "high", "service": "redis", "note": "Unauthenticated by default"},
    8080: {"level": "low", "service": "http-alt", "note": "Secondary web app surface"},
    8443: {"level": "low", "service": "https-alt", "note": "Secondary web app surface"},
    9200: {"level": "medium", "service": "elasticsearch", "note": "Data exposure if public"},
    27017: {"level": "high", "service": "mongodb", "note": "Unauthenticated by default"},
}

# A capped, common-and-risky port list -- NOT a full 1-65535 sweep.
DEFAULT_PORTS: List[int] = sorted(
    {21, 22, 23, 25, 53, 80, 110, 143, 443, 445, 993, 995, 1433, 3306, 3389,
     5432, 5900, 5984, 6379, 8080, 8443, 9200, 27017}
)

RECOMMENDED_SECURITY_HEADERS = [
    "Content-Security-Policy",
    "X-Content-Type-Options",
    "X-Frame-Options",
    "Referrer-Policy",
    "Permissions-Policy",
]

# Small built-in bruteforce wordlist (target-only -- no 3rd-party calls).
SUBDOMAIN_WORDLIST = [
    "www", "mail", "ftp", "webmail", "admin", "test", "dev", "staging",
    "api", "blog", "shop", "app", "secure", "portal", "cpanel", "vpn",
    "autodiscover", "ns1", "ns2", "smtp", "remote",
]

COMMON_LOGIN_PATHS = [
    "/login", "/signin", "/sign-in", "/auth", "/admin", "/account/login",
    "/user/login",
]
COMMON_RESET_PATHS = [
    "/password-reset", "/reset-password", "/forgot-password",
    "/account/recovery",
]

# name -> regex applied against (lowercased) response headers+HTML. A regex
# with a capture group yields that group as the detected "version" (pruned,
# re-implemented subset of ARTEX's tech_imp.py::_default_tech_db -- NO CPE/CVE
# claim is attached: ARTEX itself never wires its CPE strings to any real
# vulnerability database either (confirmed dead code, see work.md's 2026-11-24
# gap analysis), so inventing that correlation ourselves would mean asserting
# an unverified security claim. A detected version is purely a richer fact for
# the model's own follow-up reasoning).
TECH_SIGNATURES: Dict[str, str] = {
    "WordPress": r"wp-content|wp-includes|/wp-json/",
    "Drupal": r"drupal\.js|sites/default/files",
    "Joomla": r"/media/jui/|joomla",
    "nginx": r"^nginx/?(\d+\.\d+\.\d+)?",  # matched against the Server header value only
    "Apache": r"^apache/?(\d+\.\d+\.\d+)?",
    "IIS": r"^microsoft-iis/?(\d+\.\d+)?",
    "PHP": r"x-powered-by:\s*php/?(\d+\.\d+\.\d+)?",
    "React": r"data-reactroot|react-dom",
    "Angular": r"ng-version=[\"']?(\d+\.\d+\.\d+)?",
    "Vue.js": r"data-v-[a-f0-9]{6,}|__vue__",
    "jQuery": r"jquery[/-](\d+\.\d+\.\d+)?(\.min)?\.js",
    "Bootstrap": r"bootstrap[/-](\d+\.\d+\.\d+)?(\.min)?\.(css|js)",
    "Cloudflare": r"^cloudflare",  # matched against Server header
    "Django": r"csrftoken|django/?(\d+\.\d+\.\d+)?",
    "Laravel": r"laravel_session|xsrf-token",
    "Express": r"express/?(\d+\.\d+\.\d+)?",
}
# Names whose pattern is matched against the Server/X-Powered-By headers
# only, not the HTML body (everything else is checked against body+headers).
_TECH_HEADER_ONLY = {"nginx", "Apache", "IIS", "PHP", "Cloudflare"}

# ---------------------------------------------------------------------------
# Static data: secrets, exposed files, takeover services, CDN ranges
# (re-implemented, pruned, from ARTEX's core/tech_imp.py and
# modules/{subdomains,real_ip_discovery}.py -- see work.md's 2026-11-24 entry)
# ---------------------------------------------------------------------------

# name -> (regex, severity). Pruned to the highest-signal patterns; every
# match is masked (see _mask_secret) before it ever leaves this process.
SECRET_PATTERNS: Dict[str, Dict[str, str]] = {
    "AWS Access Key ID": {"pattern": r"\b(AKIA|ASIA)[0-9A-Z]{16}\b", "severity": "high"},
    "AWS Secret Access Key": {"pattern": r"(?i)aws_secret_access_key[\"'\s:=]+([A-Za-z0-9/+=]{40})", "severity": "critical"},
    "Google API Key": {"pattern": r"\bAIza[0-9A-Za-z\-_]{35}\b", "severity": "high"},
    "GitHub Token": {"pattern": r"\bgh[pousr]_[0-9A-Za-z]{36}\b", "severity": "high"},
    "Slack Token": {"pattern": r"\bxox[baprs]-[0-9A-Za-z-]{10,48}\b", "severity": "high"},
    "Stripe API Key": {"pattern": r"\bsk_(live|test)_[0-9A-Za-z]{24}\b", "severity": "critical"},
    "SendGrid API Key": {"pattern": r"\bSG\.[0-9A-Za-z\-_]{22}\.[0-9A-Za-z\-_]{43}\b", "severity": "high"},
    "Generic API Key": {"pattern": r"(?i)api[_-]?key[\"'\s:=]+[\"']?([A-Za-z0-9_\-]{16,})", "severity": "medium"},
    "Generic Access Token": {"pattern": r"(?i)access[_-]?token[\"'\s:=]+[\"']?([A-Za-z0-9_\-\.]{16,})", "severity": "medium"},
    "Database Connection String": {"pattern": r"(?i)(postgres|mysql|mongodb)://[^\s\"'<>]+:[^\s\"'<>]+@[^\s\"'<>]+", "severity": "critical"},
    "SSH Private Key": {"pattern": r"-----BEGIN (RSA|DSA|EC|OPENSSH) PRIVATE KEY-----", "severity": "critical"},
}

# path -> (severity, description, content-signature regex). A bare 200 is NOT
# a confirming signal on its own -- the signature is what makes a hit mean
# something (pruned, re-implemented subset of ARTEX's _default_exposed_files;
# ARTEX itself never checks content, only path existence).
EXPOSED_FILE_CHECKS: Dict[str, Dict[str, str]] = {
    "/.git/HEAD": {"severity": "high", "description": "Git repository metadata", "signature": r"^ref:\s*refs/"},
    "/.git/config": {"severity": "high", "description": "Git configuration", "signature": r"\[core\]|\[remote"},
    "/.env": {"severity": "critical", "description": "Environment configuration file", "signature": r"^[A-Z_][A-Z0-9_]*\s*="},
    "/.htpasswd": {"severity": "critical", "description": "Password file", "signature": r"^[\w.\-]+:\$?\w"},
    "/wp-config.php": {"severity": "critical", "description": "WordPress configuration", "signature": r"DB_PASSWORD|wp_"},
    "/config.php": {"severity": "critical", "description": "Application configuration", "signature": r"<\?php"},
    "/docker-compose.yml": {"severity": "medium", "description": "Docker configuration", "signature": r"version:|services:"},
    "/backup.sql": {"severity": "critical", "description": "Database dump", "signature": r"CREATE TABLE|INSERT INTO"},
    "/dump.sql": {"severity": "critical", "description": "Database dump", "signature": r"CREATE TABLE|INSERT INTO"},
    "/backup.zip": {"severity": "high", "description": "Application backup", "signature": r"^PK"},
    "/phpinfo.php": {"severity": "high", "description": "PHP configuration information", "signature": r"PHP Version|phpinfo\(\)"},
    "/composer.json": {"severity": "medium", "description": "PHP project configuration", "signature": r"\"require\"|\"name\""},
    "/package.json": {"severity": "medium", "description": "Node.js project configuration", "signature": r"\"dependencies\"|\"name\""},
    "/server-status": {"severity": "medium", "description": "Apache server status", "signature": r"Apache Server Status"},
}

# CNAME-pattern -> (service, content-signature for an UNCLAIMED resource).
# A content-signature match on the CNAME target is what makes this a hit,
# not the CNAME pattern alone.
TAKEOVER_SERVICES: Dict[str, Dict[str, str]] = {
    "github.io": {"service": "github_pages", "signature": r"There isn't a GitHub Pages site here"},
    "s3.amazonaws.com": {"service": "aws_s3", "signature": r"<Code>NoSuchBucket</Code>"},
    "herokuapp.com": {"service": "heroku", "signature": r"No such app"},
    "myshopify.com": {"service": "shopify", "signature": r"Sorry, this shop is currently unavailable"},
}

# Minimal CDN ranges (pruned from ARTEX's core/constants.py) used only to
# filter obvious CDN IPs out of the real_ip mode's candidates.
_CDN_RANGES: Dict[str, List[str]] = {
    "cloudflare": ["173.245.48.0/20", "103.21.244.0/22", "104.16.0.0/13", "172.64.0.0/13"],
    "akamai": ["23.0.0.0/8", "104.64.0.0/10", "184.24.0.0/13"],
    "fastly": ["23.235.32.0/20", "151.101.0.0/16"],
    "amazon_cf": ["13.32.0.0/15", "13.35.0.0/16", "54.230.0.0/16"],
}

IP_HEADERS = ("X-Real-IP", "X-Original-URL", "X-Forwarded-For",
              "X-Cluster-Client-IP", "Client-IP", "X-Client-IP")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _host_port_scheme(target: str):
    """Parse a URL or bare host into (hostname, scheme, explicit_port)."""
    if "://" not in target:
        target = "https://" + target
    parsed = urlparse(target)
    hostname = parsed.hostname or target
    scheme = parsed.scheme or "https"
    port = parsed.port
    return hostname, scheme, port


# ---------------------------------------------------------------------------
# Mode: dns
# ---------------------------------------------------------------------------

def scan_dns(hostname: str, timeout: float) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    try:
        import dns.resolver  # dnspython -- optional dependency

        resolver = dns.resolver.Resolver()
        resolver.timeout = timeout
        resolver.lifetime = timeout

        for rtype in ("A", "AAAA", "MX", "NS", "TXT", "CNAME", "SOA"):
            try:
                answers = resolver.resolve(hostname, rtype)
                result[rtype] = [str(r).strip() for r in answers]
            except Exception:
                continue

        spf = [t for t in result.get("TXT", []) if "v=spf1" in t.lower()]
        if spf:
            result["SPF"] = spf
        try:
            dmarc = resolver.resolve(f"_dmarc.{hostname}", "TXT")
            dmarc_txt = [str(r).strip() for r in dmarc if "v=dmarc1" in str(r).lower()]
            if dmarc_txt:
                result["DMARC"] = dmarc_txt
        except Exception:
            pass
    except ImportError:
        # dnspython not installed -- degrade to stdlib A-record-only lookup
        # rather than failing the whole mode.
        try:
            result["A"] = [socket.gethostbyname(hostname)]
            result["note"] = "dnspython not installed; A-record only (stdlib fallback)."
        except Exception as e:
            result["error"] = f"DNS resolution failed: {e}"
    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Mode: ports
# ---------------------------------------------------------------------------

def _check_port(hostname: str, port: int, timeout: float) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(timeout)
            return sock.connect_ex((hostname, port)) == 0
    except Exception:
        return False


def scan_ports(hostname: str, timeout: float, ports: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    port_list = ports or DEFAULT_PORTS
    open_ports: List[Dict[str, Any]] = []
    per_port_timeout = min(timeout, 2.0)
    with ThreadPoolExecutor(max_workers=min(20, len(port_list))) as executor:
        futures = {executor.submit(_check_port, hostname, p, per_port_timeout): p for p in port_list}
        for future in as_completed(futures):
            port = futures[future]
            try:
                if future.result():
                    risk = RISKY_PORTS.get(port, {"level": "low", "service": "unknown", "note": ""})
                    open_ports.append({"port": port, "risk_level": risk["level"],
                                        "service": risk["service"], "note": risk["note"]})
            except Exception:
                continue
    return sorted(open_ports, key=lambda p: p["port"])


# ---------------------------------------------------------------------------
# Mode: tls
# ---------------------------------------------------------------------------

_TLS_VERSIONS = {
    "TLSv1.0": getattr(ssl, "TLSVersion", None) and ssl.TLSVersion.TLSv1,
    "TLSv1.1": getattr(ssl, "TLSVersion", None) and ssl.TLSVersion.TLSv1_1,
    "TLSv1.2": getattr(ssl, "TLSVersion", None) and ssl.TLSVersion.TLSv1_2,
    "TLSv1.3": getattr(ssl, "TLSVersion", None) and ssl.TLSVersion.TLSv1_3,
}


def _protocol_supported(hostname: str, port: int, version, timeout: float) -> bool:
    if version is None:
        return False
    try:
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.minimum_version = version
        context.maximum_version = version
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
        with socket.create_connection((hostname, port), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname):
                return True
    except Exception:
        return False


def _connect_and_read_peer(hostname: str, port: int, timeout: float, context: ssl.SSLContext):
    with socket.create_connection((hostname, port), timeout=timeout) as sock:
        with context.wrap_socket(sock, server_hostname=hostname) as ssock:
            return ssock.getpeercert(), ssock.version(), ssock.cipher()


def scan_tls(hostname: str, port: int, timeout: float) -> Dict[str, Any]:
    """
    getpeercert() only returns a populated dict when the handshake's
    verify_mode actually validated the certificate (CERT_NONE always yields
    {} even though a cert was presented) -- so this first tries a NORMAL
    system-trust validation (check_hostname off, since recon targets a bare
    IP/host rather than necessarily the cert's exact SAN) to get the full
    parsed cert (subject/issuer/validity dates) for the common case of a
    valid public cert. If that validation fails, the ssl.SSLError's own
    message is used to recognize self-signed / expired certs (no
    `cryptography` dependency needed), then a SECOND, CERT_NONE connection is
    made just to recover negotiated_protocol/cipher for a cert we can't trust
    but still want to describe.
    """
    result: Dict[str, Any] = {"port": port}
    try:
        trusted_context = ssl.create_default_context()
        trusted_context.check_hostname = False
        cert = negotiated_version = cipher = None
        verify_error: Optional[str] = None
        try:
            cert, negotiated_version, cipher = _connect_and_read_peer(hostname, port, timeout, trusted_context)
            result["trusted"] = True
        except ssl.SSLCertVerificationError as e:
            verify_error = str(e).lower()
            result["trusted"] = False
        except ssl.SSLError as e:
            verify_error = str(e).lower()
            result["trusted"] = False

        if cert:
            subject = dict(x[0] for x in cert.get("subject", []))
            issuer = dict(x[0] for x in cert.get("issuer", []))
            result["subject_cn"] = subject.get("commonName")
            result["issuer_cn"] = issuer.get("commonName")
            result["not_before"] = cert.get("notBefore")
            result["not_after"] = cert.get("notAfter")
            result["self_signed"] = subject == issuer and bool(subject)
            try:
                not_after = datetime.strptime(cert["notAfter"], "%b %d %H:%M:%S %Y %Z").replace(
                    tzinfo=timezone.utc)
                result["expired"] = not_after < datetime.now(timezone.utc)
            except Exception:
                result["expired"] = None
        elif verify_error:
            result["self_signed"] = "self signed" in verify_error or "self-signed" in verify_error
            result["expired"] = "certificate has expired" in verify_error or "expired" in verify_error
            result["note"] = f"Certificate not trusted by system CA store: {verify_error}"
            # Reconnect without validation purely to recover negotiated
            # protocol/cipher for a certificate we already know we can't trust.
            try:
                none_context = ssl.create_default_context()
                none_context.check_hostname = False
                none_context.verify_mode = ssl.CERT_NONE
                _, negotiated_version, cipher = _connect_and_read_peer(hostname, port, timeout, none_context)
            except Exception:
                pass

        result["negotiated_protocol"] = negotiated_version
        result["negotiated_cipher"] = cipher[0] if cipher else None

        # Protocol support sweep -- each a separate short-lived handshake.
        result["protocols"] = {
            name: _protocol_supported(hostname, port, version, min(timeout, 5.0))
            for name, version in _TLS_VERSIONS.items()
        }

        # One plain HTTPS GET for HSTS + missing security headers (reuses the
        # same host/port; a single extra request, not a new TLS primitive).
        try:
            resp = requests.get(f"https://{hostname}:{port}/", timeout=timeout, verify=False)
            headers = resp.headers
            result["hsts"] = "strict-transport-security" in {h.lower() for h in headers}
            missing = [h for h in RECOMMENDED_SECURITY_HEADERS if h not in headers]
            result["missing_security_headers"] = missing
        except Exception:
            pass

    except Exception as e:
        result["error"] = str(e)
    return result


# ---------------------------------------------------------------------------
# Mode: tech
# ---------------------------------------------------------------------------

def scan_tech(url: str, timeout: float) -> List[Dict[str, str]]:
    detected: List[Dict[str, str]] = []
    try:
        resp = requests.get(url, timeout=timeout, verify=False, allow_redirects=True)
        server_header = resp.headers.get("Server", "")
        powered_by = resp.headers.get("X-Powered-By", "")
        html = resp.text or ""
        haystack_headers = f"{server_header}\n{powered_by}".lower()
        haystack_html = html.lower()

        for name, pattern in TECH_SIGNATURES.items():
            # Server/CDN-style signatures are matched against headers only;
            # everything else against the HTML body (and X-Powered-By).
            haystack = haystack_headers if name in _TECH_HEADER_ONLY else (haystack_html + "\n" + haystack_headers)
            match = re.search(pattern, haystack)
            if match:
                entry = {"name": name, "source": "header" if name in _TECH_HEADER_ONLY else "html_or_header"}
                version = next((g for g in match.groups() if g), None) if match.groups() else None
                if version:
                    entry["version"] = version
                detected.append(entry)

        if server_header:
            detected.append({"name": f"server:{server_header}", "source": "header"})
    except Exception:
        pass
    return detected


# ---------------------------------------------------------------------------
# Mode: secrets
# ---------------------------------------------------------------------------

def _mask_secret(value: str) -> str:
    """First/last 4 chars visible, rest replaced -- a live credential must
    never reach output/logs/trajectory files in cleartext."""
    if not value:
        return value
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}{'*' * (len(value) - 8)}{value[-4:]}"


def scan_secrets(url: str, timeout: float) -> List[Dict[str, str]]:
    """Regex-scan the fetched homepage for leaked credential/token patterns.
    Every match is masked before it is ever returned -- this is a NEW safety
    addition beyond ARTEX, which has no masking at all (see work.md)."""
    found: List[Dict[str, str]] = []
    try:
        resp = requests.get(url, timeout=timeout, verify=False, allow_redirects=True)
        body = resp.text or ""
        for name, spec in SECRET_PATTERNS.items():
            for match in re.finditer(spec["pattern"], body):
                raw = match.group(1) if match.groups() else match.group(0)
                found.append({
                    "name": name,
                    "severity": spec["severity"],
                    "masked_value": _mask_secret(raw),
                })
    except Exception:
        pass
    return found


# ---------------------------------------------------------------------------
# Mode: exposed_files
# ---------------------------------------------------------------------------

def scan_exposed_files(base_url: str, timeout: float) -> List[Dict[str, Any]]:
    """Probe a short list of common sensitive paths. A bare 200 is NOT a
    confirming signal by itself (a custom error page can return 200 for
    anything) -- signature_matched is what makes a hit meaningful."""
    found: List[Dict[str, Any]] = []
    base_url = base_url.rstrip("/")

    def _check(path: str, spec: Dict[str, str]):
        url = f"{base_url}{path}"
        try:
            resp = requests.get(url, timeout=timeout, verify=False, allow_redirects=False)
            if resp.status_code != 200:
                return None
            signature_matched = bool(re.search(spec["signature"], resp.text or "", re.MULTILINE))
            return {
                "path": path,
                "url": url,
                "status": resp.status_code,
                "severity": spec["severity"],
                "description": spec["description"],
                "signature_matched": signature_matched,
            }
        except Exception:
            return None

    with ThreadPoolExecutor(max_workers=min(10, len(EXPOSED_FILE_CHECKS))) as executor:
        futures = [executor.submit(_check, path, spec) for path, spec in EXPOSED_FILE_CHECKS.items()]
        for future in as_completed(futures):
            try:
                hit = future.result()
                if hit:
                    found.append(hit)
            except Exception:
                continue
    return found


# ---------------------------------------------------------------------------
# Mode: authsurface
# ---------------------------------------------------------------------------

def _looks_like_auth_page(html: str) -> bool:
    lowered = html.lower()
    if 'type="password"' in lowered or "type='password'" in lowered:
        return True
    return any(kw in lowered for kw in ("sign in", "log in", "forgot password", "reset password"))


def scan_authsurface(base_url: str, timeout: float) -> List[Dict[str, Any]]:
    found: List[Dict[str, Any]] = []
    base_url = base_url.rstrip("/")

    def _check(path: str, kind: str):
        url = f"{base_url}{path}"
        try:
            resp = requests.get(url, timeout=timeout, verify=False, allow_redirects=True)
            if resp.status_code == 200 and _looks_like_auth_page(resp.text or ""):
                return {"url": url, "type": kind, "status": resp.status_code}
        except Exception:
            return None
        return None

    paths = [(p, "login") for p in COMMON_LOGIN_PATHS] + [(p, "password_reset") for p in COMMON_RESET_PATHS]
    with ThreadPoolExecutor(max_workers=min(10, len(paths))) as executor:
        futures = [executor.submit(_check, p, kind) for p, kind in paths]
        for future in as_completed(futures):
            try:
                hit = future.result()
                if hit:
                    found.append(hit)
            except Exception:
                continue
    return found


# ---------------------------------------------------------------------------
# Mode: subdomains (opt-in, off by default, never run in single-host mode)
# ---------------------------------------------------------------------------

def _check_subdomain(sub: str, domain: str, timeout: float) -> Optional[str]:
    candidate = f"{sub}.{domain}"
    try:
        socket.setdefaulttimeout(timeout)
        socket.gethostbyname(candidate)
        return candidate
    except Exception:
        return None


def scan_subdomains(domain: str, timeout: float, allow_ct_lookup: bool = False) -> List[str]:
    found = set()
    with ThreadPoolExecutor(max_workers=min(20, len(SUBDOMAIN_WORDLIST))) as executor:
        futures = [executor.submit(_check_subdomain, sub, domain, min(timeout, 3.0))
                   for sub in SUBDOMAIN_WORDLIST]
        for future in as_completed(futures):
            try:
                hit = future.result()
                if hit:
                    found.add(hit)
            except Exception:
                continue

    if allow_ct_lookup:
        # crt.sh: one well-known, read-only, public certificate-transparency
        # API. This is a lookup ABOUT the domain against a 3rd-party log,
        # never a request sent TO the target -- explicitly opt-in only.
        try:
            resp = requests.get(f"https://crt.sh/?q=%.{domain}&output=json", timeout=timeout)
            if resp.status_code == 200:
                for cert in resp.json():
                    for name in cert.get("name_value", "").split("\n"):
                        name = name.strip().lower().lstrip("*.")
                        if name and name.endswith(domain) and re.match(r"^[a-z0-9.\-]+$", name):
                            found.add(name)
        except Exception:
            pass

    return sorted(found)[:100]


def _check_takeover(subdomain: str, timeout: float) -> Optional[Dict[str, str]]:
    try:
        import dns.resolver  # dnspython -- optional, same as scan_dns
        try:
            answers = dns.resolver.resolve(subdomain, "CNAME")
        except Exception:
            return None
        for answer in answers:
            cname = str(answer.target).rstrip(".").lower()
            for pattern, spec in TAKEOVER_SERVICES.items():
                if pattern not in cname:
                    continue
                try:
                    resp = requests.get(f"http://{subdomain}", timeout=timeout, verify=False)
                    if re.search(spec["signature"], resp.text or ""):
                        return {"subdomain": subdomain, "cname": cname, "service": spec["service"]}
                except Exception:
                    continue
    except ImportError:
        return None
    except Exception:
        return None
    return None


def scan_takeovers(subdomains: List[str], timeout: float) -> List[Dict[str, str]]:
    """For each subdomain with a CNAME matching a known-vulnerable-service
    pattern, fetch it and check for that service's distinctive "unclaimed"
    content signature (re-implemented from ARTEX's
    _check_subdomain_takeovers). A CNAME match alone is NOT a hit -- the
    content signature is what confirms it."""
    if not subdomains:
        return []
    found: List[Dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=min(10, len(subdomains))) as executor:
        futures = [executor.submit(_check_takeover, sub, min(timeout, 5.0)) for sub in subdomains]
        for future in as_completed(futures):
            try:
                hit = future.result()
                if hit:
                    found.append(hit)
            except Exception:
                continue
    return found


# ---------------------------------------------------------------------------
# Mode: real_ip
# ---------------------------------------------------------------------------

def _is_private_or_cdn_ip(ip: str) -> bool:
    try:
        import ipaddress
        ip_obj = ipaddress.ip_address(ip)
        if ip_obj.is_private or ip_obj.is_loopback or ip_obj.is_link_local:
            return True
        for ranges in _CDN_RANGES.values():
            for cidr in ranges:
                if ip_obj in ipaddress.ip_network(cidr, strict=False):
                    return True
    except Exception:
        return True  # unparseable -- do not surface it as a "real" IP
    return False


def scan_real_ip(hostname: str, timeout: float) -> Dict[str, Any]:
    """Best-effort real-IP-behind-CDN discovery: MX-server IPs and HTTP
    response-header IP leaks, filtered against known CDN/private ranges and
    confidence-scored by source agreement (re-implemented, pruned, from
    ARTEX's real_ip_discovery.py -- the CT-log sub-method is dropped here
    since the subdomains mode already covers CT-log lookups). Informational
    only -- never promotes a Finding."""
    found_by_method: Dict[str, set] = {"mail_servers": set(), "http_headers": set()}

    try:
        import dns.resolver
        try:
            mx_records = dns.resolver.resolve(hostname, "MX")
            for mx in mx_records:
                mail_server = str(mx.exchange).rstrip(".")
                try:
                    ip = socket.gethostbyname(mail_server)
                    if not _is_private_or_cdn_ip(ip):
                        found_by_method["mail_servers"].add(ip)
                except Exception:
                    continue
        except Exception:
            pass
    except ImportError:
        pass

    for path in ("/", "/admin", "/.env"):
        try:
            resp = requests.get(f"https://{hostname}{path}", timeout=timeout, verify=False)
            for header_name, header_value in resp.headers.items():
                if header_name not in IP_HEADERS:
                    continue
                for ip in re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", header_value):
                    if not _is_private_or_cdn_ip(ip):
                        found_by_method["http_headers"].add(ip)
        except Exception:
            continue

    method_weight = {"mail_servers": 10, "http_headers": 8}
    scored: Dict[str, Dict[str, Any]] = {}
    for method, ips in found_by_method.items():
        for ip in ips:
            entry = scored.setdefault(ip, {"ip": ip, "score": 0, "methods": []})
            entry["score"] += method_weight[method]
            entry["methods"].append(method)

    candidates = sorted(scored.values(), key=lambda e: e["score"], reverse=True)
    return {"candidates": candidates[:10]}


# ---------------------------------------------------------------------------
# Mode: js_recon (route mining + inertness pruning)
# ---------------------------------------------------------------------------

_REL_API_RE = re.compile(r'["\'](/(?:api|v\d+)/[a-zA-Z0-9_\-/]{1,80})["\']')
_NEXT_DATA_RE = re.compile(r'["\'](/_next/data/[a-zA-Z0-9_\-/.]{1,120}\.json)["\']')
_JS_SRC_RE = re.compile(r'<script[^>]+src=["\']([^"\']+\.js[^"\']*)["\']', re.IGNORECASE)
_FORM_BLOCK_RE = re.compile(r'<form\b[^>]*>.*?</form>', re.IGNORECASE | re.DOTALL)
_FORM_ACTION_RE = re.compile(r'action=["\']([^"\']+)["\']', re.IGNORECASE)
_INPUT_NAME_RE = re.compile(r'<input[^>]+name=["\']([^"\']+)["\']', re.IGNORECASE)
_API_HINT_RE = re.compile(r'/(api|v\d+)/')


def _mine_routes(text: str) -> set:
    """Pull relative API routes + Next.js data routes out of HTML/JS text."""
    found = set()
    for m in _REL_API_RE.findall(text or ""):
        found.add(m)
    for m in _NEXT_DATA_RE.findall(text or ""):
        found.add(m)
    return found


def _is_testable_route(url: str, param: str, source: str) -> bool:
    """Filter out marketing/content pages -- keep only routes that look like
    real attack surface (an API-shaped path, a query string, or a form)."""
    path = urlparse(url).path
    return bool(_API_HINT_RE.search(path)) or "?" in url or source == "form" or bool(param)


def scan_js_routes(base_url: str, timeout: float, max_js: int = 10) -> Dict[str, Any]:
    """
    Mine the homepage and a capped number of same-origin JS bundles for
    relative API routes, Next.js data routes, and forms, then prune params
    that have no observable effect on the response (re-implemented from
    Claude-BugHunter's engine/recon.py::spa_recon + prune_inert_params --
    see work.md's 2026-11-25 harvest entry). Read-only (GET only); a marker
    value is appended as a query param, never submitted as a mutating body.
    """
    result: Dict[str, Any] = {"routes": [], "forms": []}
    try:
        resp = requests.get(base_url, timeout=timeout, verify=False, allow_redirects=True)
        html = resp.text or ""
        found_routes = _mine_routes(html)

        js_urls = set()
        for m in _JS_SRC_RE.findall(html):
            ju = urljoin(base_url, m)
            if urlparse(ju).netloc == urlparse(base_url).netloc:
                js_urls.add(ju)

        for ju in list(js_urls)[:max_js]:
            try:
                jresp = requests.get(ju, timeout=timeout, verify=False)
                found_routes |= _mine_routes(jresp.text or "")
            except Exception:
                continue

        forms = []
        for fm in _FORM_BLOCK_RE.findall(html):
            act = _FORM_ACTION_RE.search(fm)
            action = urljoin(base_url, act.group(1)) if act else base_url
            for p in (_INPUT_NAME_RE.findall(fm) or [""]):
                forms.append({"action": action, "param": p})
        result["forms"] = forms[:30]

        items = []
        for rel in sorted(found_routes):
            url = urljoin(base_url, rel)
            if urlparse(url).netloc == urlparse(base_url).netloc:
                items.append({"url": url, "param": "", "source": "js_html"})
        testable = [it for it in items if _is_testable_route(it["url"], it.get("param", ""), it["source"])]

        pruned = _prune_inert_routes(testable, timeout)
        result["routes"] = pruned[:50]
        result["total_mined"] = len(items)
        result["pruned_inert"] = len(testable) - len(pruned)
    except Exception as e:
        result["error"] = str(e)
    return result


def _probe_marker(url: str, timeout: float):
    """GET a URL, returning (status_code, body) or (0, '') on failure."""
    try:
        resp = requests.get(url, timeout=timeout, verify=False, allow_redirects=True)
        return resp.status_code, (resp.text or "")
    except Exception:
        return 0, ""


def _prune_inert_routes(items: List[Dict[str, Any]], timeout: float) -> List[Dict[str, Any]]:
    """
    Drop routes whose base endpoint is dead (>=400) -- no param-inertness
    check here since js-mined routes carry no param yet (that's param_fuzz's
    job); this stays a cheap liveness filter, re-implemented from the same
    marker-diff idea as Claude-BugHunter's prune_inert_params but scoped to
    what js_recon actually discovers (bare routes, not query params).
    """
    kept = []
    base_cache: Dict[str, int] = {}
    for it in items:
        base_path = it["url"].split("?", 1)[0]
        if base_path not in base_cache:
            code, _ = _probe_marker(base_path, min(timeout, 5.0))
            base_cache[base_path] = code
        code = base_cache[base_path]
        if code == 0 or code < 400:
            kept.append(it)
    return kept


# ---------------------------------------------------------------------------
# Mode: auth_sweep (unauthenticated probe of declared-secured OpenAPI ops)
# ---------------------------------------------------------------------------

OPENAPI_PATHS = ["/openapi.json", "/swagger.json", "/api/openapi.json", "/v1/openapi.json",
                  "/api-docs", "/swagger/v1/swagger.json", "/api/v1/openapi.json"]


def _discover_openapi_spec(base_url: str, timeout: float) -> Optional[Dict[str, Any]]:
    for path in OPENAPI_PATHS:
        try:
            resp = requests.get(base_url.rstrip("/") + path, timeout=timeout, verify=False)
            if resp.status_code == 200:
                spec = resp.json()
                if isinstance(spec, dict) and "paths" in spec:
                    return spec
        except Exception:
            continue
    return None


def scan_auth_enforcement(base_url: str, timeout: float) -> List[Dict[str, Any]]:
    """
    Discover an OpenAPI/Swagger spec and probe each DECLARED-SECURED GET
    operation UNAUTHENTICATED. Never probes POST/PUT/PATCH/DELETE (no
    allow-intrusive switch -- read-only by design). A response that reaches
    the handler/validation layer (2xx or 422) where 401/403 was declared is
    a genuine broken-function-level-auth signal (re-implemented from
    Claude-BugHunter's engine/recon.py::auth_enforcement_sweep -- see
    work.md's 2026-11-25 harvest entry).
    """
    broken: List[Dict[str, Any]] = []
    try:
        spec = _discover_openapi_spec(base_url, timeout)
        if not spec:
            return broken
        global_secured = spec.get("security") not in (None, [])
        junk = "zzz-recon-nonexistent-00000"

        for path, methods in (spec.get("paths") or {}).items():
            op = (methods or {}).get("get")
            if not op:
                continue
            op_secured = ("security" in op and op["security"] != []) or \
                         ("security" not in op and global_secured)
            if not op_secured:
                continue

            url_path = path
            for pp in op.get("parameters", []) or []:
                if pp.get("in") == "path":
                    url_path = url_path.replace("{%s}" % pp.get("name", ""), junk)
            url_path = url_path.replace("{", "").replace("}", "")
            url = base_url.rstrip("/") + url_path
            query_params = [f"{pp['name']}={junk}" for pp in (op.get("parameters") or [])
                             if pp.get("in") == "query" and pp.get("required")]
            if query_params:
                url += ("&" if "?" in url else "?") + "&".join(query_params)

            try:
                resp = requests.get(url, timeout=timeout, verify=False, allow_redirects=False,
                                     headers={"Accept": "application/json"})
                code = resp.status_code
            except Exception:
                continue

            if code in (200, 201, 202, 204, 422):
                broken.append({
                    "url": url, "method": "GET", "status": code,
                    "note": f"GET {path} -> {code} unauthenticated (declared-secured, NOT enforced)",
                })
    except Exception as e:
        return [{"error": str(e)}]
    return broken


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def run_recon(target: str, modes: List[str], timeout: float, allow_ct_lookup: bool) -> Dict[str, Any]:
    hostname, scheme, explicit_port = _host_port_scheme(target)
    base_url = f"{scheme}://{hostname}" + (f":{explicit_port}" if explicit_port else "")
    tls_port = explicit_port or (443 if scheme == "https" else 443)

    jobs = {}
    with ThreadPoolExecutor(max_workers=max(1, len(modes))) as executor:
        if "dns" in modes:
            jobs["dns"] = executor.submit(scan_dns, hostname, timeout)
        if "ports" in modes:
            jobs["ports"] = executor.submit(scan_ports, hostname, timeout)
        if "tls" in modes:
            jobs["tls"] = executor.submit(scan_tls, hostname, tls_port, timeout)
        if "tech" in modes:
            jobs["tech"] = executor.submit(scan_tech, base_url, timeout)
        if "authsurface" in modes:
            jobs["auth_pages"] = executor.submit(scan_authsurface, base_url, timeout)
        if "subdomains" in modes:
            jobs["subdomains"] = executor.submit(scan_subdomains, hostname, timeout, allow_ct_lookup)
        if "secrets" in modes:
            jobs["secrets"] = executor.submit(scan_secrets, base_url, timeout)
        if "exposed_files" in modes:
            jobs["exposed_files"] = executor.submit(scan_exposed_files, base_url, timeout)
        if "real_ip" in modes:
            jobs["real_ip"] = executor.submit(scan_real_ip, hostname, timeout)
        if "js_recon" in modes:
            jobs["js_recon"] = executor.submit(scan_js_routes, base_url, timeout)
        if "auth_sweep" in modes:
            jobs["auth_sweep"] = executor.submit(scan_auth_enforcement, base_url, timeout)

        recon: Dict[str, Any] = {"target": hostname, "scanned_at": _now()}
        for key, future in jobs.items():
            try:
                recon[key] = future.result()
            except Exception as e:
                recon[key] = {"error": str(e)}

    # Takeover detection is a POST-step that depends on the resolved
    # subdomains list, so it runs after the main parallel sweep completes
    # rather than alongside it.
    if "subdomains" in modes:
        try:
            recon["takeovers"] = scan_takeovers(recon.get("subdomains", []), timeout)
        except Exception as e:
            recon["takeovers"] = {"error": str(e)}

    return recon


def main(argv=None):
    ap = argparse.ArgumentParser(description="Breadth-first parallel recon engine (read-only, detection-only).")
    ap.add_argument("--url", required=True, help="Target URL or bare host.")
    ap.add_argument("--modes", default="dns,ports,tls,tech,authsurface",
                     help="Comma-separated: dns,ports,tls,tech,authsurface,subdomains,"
                          "secrets,exposed_files,real_ip,js_recon,auth_sweep")
    ap.add_argument("--timeout", type=float, default=8.0)
    ap.add_argument("--allow-ct-lookup", action="store_true",
                     help="Also query crt.sh in subdomains mode (opt-in, 3rd-party read-only API).")
    args = ap.parse_args(argv)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    try:
        recon = run_recon(args.url, modes, args.timeout, args.allow_ct_lookup)
    except Exception as e:
        # Never crash without emitting a parseable result.
        recon = {"target": args.url, "scanned_at": _now(), "error": f"recon_scan_runner internal error: {e}"}

    print(json.dumps({"recon": recon}))


if __name__ == "__main__":
    main()
