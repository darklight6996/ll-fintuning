#!/usr/bin/env python3
"""
core/tools/http_probe.py

THE RICH HTTP PRIMITIVE (engine) -- the hunter substrate.

A surgical, low-noise HTTP prober with a DIFFERENTIAL + TIMING ORACLE. Unlike
the one-shot curl probes (which fire a fixed payload and barely look at the
reply) or sqlmap (which is thorough but loud), this sends a small, precise set
of requests and REASONS about the responses: status / length / timing deltas,
SQL-error signatures, boolean-pair divergence, and unencoded reflection. That
"observe baseline -> mutate one thing -> read the exact difference" loop is the
core of real bug hunting, and it is the substrate the browser and self-scripting
primitives build on.

It is invoked as an external tool (argv in, JSON out) exactly like sqlmap/nuclei,
so it flows through the executor's 10 gates, the scope lock, and the
verdict-aware evidence promotion unchanged -- and it is framework-level, usable
by every agent, not a bug_bounty bolt-on.

MODES
  auto      (default) For every query parameter in --url, run the benign
            differential battery (error / boolean-pair / LFI / open-redirect /
            reflection) and report the strongest signal.
  targeted  Send ONE mutation (--param/--payload) and diff it against baseline.
  raw       Send a raw HTTP request over a socket (from --raw-file) and report
            status/length/timing. This is the request-smuggling substrate
            (CL.TE/TE.CL desync detection builds on raw timing differentials);
            exposed now so the capability exists, with detection logic layered
            on next.

SAFETY / STEALTH
  * Benign payloads only: a single quote (error oracle), a classic boolean pair
    (' AND '1'='1 vs ' AND '1'='2), a path-traversal read of /etc/passwd (LFI
    oracle -- read-only, a file read is not a write), an external canary host
    for the open-redirect oracle, and a random alphanumeric canary (reflection).
    No destructive SQL, no stacked queries, no data exfiltration, no shell.
  * Hard caps: --timeout per request, --max-requests total. A handful of
    requests per run, not a flood -- the surgical move is also the un-bannable
    one.
  * Reads nothing but the target's own responses; changes no state. The
    open-redirect oracle never follows the redirect it provokes -- it inspects
    the Location header of a single un-followed response.

OUTPUT
  A single JSON object on stdout: a vuln_signal consumed by the fact extractor.
  confirmed=True only on a real oracle hit (SQL error, strong boolean
  divergence, an /etc/passwd signature, or a Location header that echoes the
  injected external host); reflection is reported as a LEAD (confirmed False,
  lead=True) -- a reflected canary is an XSS *surface*, not a confirmed bug.
  No hit -> a clean confirmed=False signal. It never fabricates a verdict.

  The LFI and open-redirect typed oracles (2026-11-25 4-repo harvest, asset 6)
  are re-implemented from Claude-BugHunter's engine/verifier.py "fact-is-proof"
  pattern (github.com/elementalsouls/Claude-BugHunter, MIT) -- its core idea
  that LFI/open-redirect/SQLi/XSS are "covered" classes where a single
  deterministic, LLM-free signature check in the replayed response IS the
  confirming evidence, no model judgment required. Re-authored against this
  file's existing single-shot differential-battery shape rather than ported as
  verifier.py's separate capture/corroborate two-step (this engine already
  captures-and-judges in one pass for sqli/xss) -- see work.md's 2026-11-25
  harvest entry for the full attribution and design rationale. Any excerpt
  placed in a vuln_signal's "evidence" field is redacted (secrets/emails
  stripped) before being written, same discipline as the rest of this engine's
  --max-requests and benign-payload-only posture.
"""

import argparse
import difflib
import json
import random
import re
import socket
import ssl
import string
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_SQL_ERRORS = (
    "sql syntax", "mysql_fetch", "you have an error in your sql",
    "warning: pg_", "postgresql", "unterminated quoted string",
    "quoted string not properly terminated", "syntax error at or near",
    "unclosed quotation mark", "ora-0", "odbc", "sqlite3.operationalerror",
    "sqlstate", "native client", "incorrect syntax near",
)

# LFI oracle signature: a classic /etc/passwd root line. A fact, not a guess --
# re-implemented from Claude-BugHunter's verifier.py::_LFI_RE.
_LFI_RE = re.compile(r"\broot:[^:]*:0:0:")

# Open-redirect canary host: distinctive enough that a real app would never
# legitimately redirect here, so any Location match is unambiguous.
_REDIRECT_CANARY_HOST = "oracle-canary.invalid-test.example"

# Redaction, reused (re-authored, not copied) from verifier.py's redaction
# catalog -- applied to any response excerpt before it is ever written into a
# vuln_signal's "evidence" field, so a real secret in a target's response body
# never ends up in cleartext in report.md / investigation.json / the
# training-trajectory logs this framework already writes every run to.
_SECRET_RES = [re.compile(p) for p in (
    r"(?:AKIA|ASIA)[0-9A-Z]{16}",
    r"AIza[0-9A-Za-z_\-]{35}",
    r"gh[pousr]_[A-Za-z0-9]{36,}",
    r"sk-(?:ant-)?[A-Za-z0-9_\-]{20,}",
    r"xox[abpors]-[0-9A-Za-z\-]{10,}",
    r"eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}",  # JWT
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----",
)]
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")


def _redact(text):
    if not text:
        return text
    for rx in _SECRET_RES:
        text = rx.sub("[REDACTED-SECRET]", text)
    return _EMAIL_RE.sub("[REDACTED-EMAIL]", text)


_UA = "Mozilla/5.0 (compatible; ll-hunter/1.0; authorized-assessment)"


def _opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx))


def _fetch(url, timeout):
    """Return (status, body, elapsed_seconds) or (None, '', elapsed) on error."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    t0 = time.monotonic()
    try:
        with _opener().open(req, timeout=timeout) as r:
            body = r.read(200_000).decode("utf-8", "replace")
            return r.status, body, time.monotonic() - t0
    except urllib.error.HTTPError as e:  # 4xx/5xx still carry a body we want
        try:
            body = e.read(200_000).decode("utf-8", "replace")
        except Exception:
            body = ""
        return e.code, body, time.monotonic() - t0
    except Exception:
        return None, "", time.monotonic() - t0


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Suppress automatic redirect-following so Location stays readable --
    re-implemented from verifier.py's default_fetch pattern."""
    def redirect_request(self, *a, **k):
        return None


def _no_redirect_opener():
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx), _NoRedirect())


def _fetch_no_redirect(url, timeout):
    """Single GET, redirects NOT followed. Returns (status, location) or
    (None, None) on error -- used only by the open-redirect oracle, which
    needs the raw Location header rather than the page it points to."""
    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with _no_redirect_opener().open(req, timeout=timeout) as r:
            return r.status, r.headers.get("Location")
    except urllib.error.HTTPError as e:
        return e.code, (e.headers.get("Location") if e.headers else None)
    except Exception:
        return None, None


def _set_param(url, param, value):
    parts = urllib.parse.urlsplit(url)
    q = dict(urllib.parse.parse_qsl(parts.query, keep_blank_values=True))
    q[param] = value
    new_q = urllib.parse.urlencode(q)
    return urllib.parse.urlunsplit((parts.scheme, parts.netloc, parts.path, new_q, parts.fragment))


def _similar(a, b):
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a or "", b or "").quick_ratio()


def _has_sql_error(body):
    low = (body or "").lower()
    return next((sig for sig in _SQL_ERRORS if sig in low), None)


def _signal(**kw):
    print(json.dumps({"vuln_signal": kw}))


def _probe_param(url, param, base_status, base_body, timeout, budget):
    """Benign differential battery for one parameter. Returns a signal dict or None."""
    orig = dict(urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True)).get(param, "")

    # 1) Error oracle: a lone single quote.
    if budget[0] > 0:
        budget[0] -= 1
        s, b, _ = _fetch(_set_param(url, param, orig + "'"), timeout)
        sig = _has_sql_error(b)
        if sig and not _has_sql_error(base_body):
            return {
                "confirmed": True, "severity": "high", "vuln_class": "sqli",
                "parameter": param,
                "summary": f"Differential probe: SQL error signature ('{sig}') appeared on parameter "
                           f"'{param}' after injecting a single quote (not present in baseline).",
            }

    # 2) Boolean-pair oracle: TRUE vs FALSE should diverge iff injectable.
    if budget[0] > 1:
        budget[0] -= 2
        _, b_true, _ = _fetch(_set_param(url, param, orig + "' AND '1'='1"), timeout)
        _, b_false, _ = _fetch(_set_param(url, param, orig + "' AND '1'='2"), timeout)
        sim_tf = _similar(b_true, b_false)
        sim_tb = _similar(b_true, base_body)
        # TRUE close to baseline, FALSE clearly different -> classic boolean SQLi.
        if sim_tb > 0.95 and sim_tf < 0.85:
            return {
                "confirmed": True, "severity": "high", "vuln_class": "sqli",
                "parameter": param,
                "summary": f"Differential probe: boolean-based divergence on parameter '{param}' "
                           f"(TRUE~baseline sim={sim_tb:.2f}, TRUE/FALSE sim={sim_tf:.2f}).",
            }

    # 3) LFI oracle: a path-traversal read of /etc/passwd. A read-only GET;
    # the signature itself (root:x:0:0:) is the fact, no judgment needed --
    # re-implemented from Claude-BugHunter's verifier.py::_LFI_RE.
    if budget[0] > 0 and not _LFI_RE.search(base_body or ""):
        budget[0] -= 1
        _, b, _ = _fetch(_set_param(url, param, "../" * 8 + "etc/passwd"), timeout)
        m = _LFI_RE.search(b or "")
        if m:
            excerpt = (b or "")[max(0, m.start() - 20):m.end() + 60]
            return {
                "confirmed": True, "severity": "high", "vuln_class": "lfi",
                "parameter": param,
                "summary": f"Path-traversal probe on parameter '{param}' returned an /etc/passwd "
                           f"signature not present in the baseline response.",
                "evidence": _redact(excerpt),
            }

    # 4) Open-redirect oracle: does the app redirect to a host WE supplied?
    # Never follows the redirect -- only reads the un-followed Location header,
    # so this probes the app's own redirect logic, not whatever that header
    # points at. Re-implemented from verifier.py's open-redirect oracle.
    if budget[0] > 0:
        budget[0] -= 1
        canary_url = f"https://{_REDIRECT_CANARY_HOST}/"
        for candidate in (canary_url, "//" + _REDIRECT_CANARY_HOST):
            status, location = _fetch_no_redirect(_set_param(url, param, candidate), timeout)
            if status in (301, 302, 303, 307, 308) and location:
                host = (urllib.parse.urlsplit(
                    location if "://" in location or location.startswith("//") else "//" + location
                ).hostname or "").lower()
                if host == _REDIRECT_CANARY_HOST:
                    return {
                        "confirmed": True, "severity": "medium", "vuln_class": "open-redirect",
                        "parameter": param,
                        "summary": f"Parameter '{param}' set to an externally-controlled URL produced "
                                   f"an HTTP {status} redirect whose Location host matches the injected "
                                   f"host -- the app redirects to attacker-supplied destinations.",
                        "evidence": _redact(location),
                    }

    # 5) Reflection LEAD (not a confirmed bug): unencoded canary echo = XSS surface.
    if budget[0] > 0:
        budget[0] -= 1
        canary = "zq" + "".join(random.choices(string.ascii_lowercase + string.digits, k=10))
        _, b, _ = _fetch(_set_param(url, param, canary), timeout)
        if canary in (b or ""):
            return {
                "confirmed": False, "lead": True, "severity": "low", "vuln_class": "xss",
                "parameter": param,
                "summary": f"Lead: parameter '{param}' reflects input unencoded (canary echoed in "
                           f"response) -- possible XSS surface; validate context with dalfox/browser.",
            }
    return None


def mode_auto(url, timeout, max_requests):
    base_status, base_body, _ = _fetch(url, timeout)
    if base_status is None:
        _signal(confirmed=False, vuln_class="http", summary="Target did not respond to the baseline request.")
        return
    params = [k for k, _ in urllib.parse.parse_qsl(urllib.parse.urlsplit(url).query, keep_blank_values=True)]
    if not params:
        _signal(confirmed=False, vuln_class="http",
                summary="No query parameters in the URL to differentially probe (supply a URL with ?param=value).")
        return
    budget = [max_requests - 1]  # baseline already spent
    best = None
    for p in params:
        if budget[0] <= 0:
            break
        res = _probe_param(url, p, base_status, base_body, timeout, budget)
        if res and res.get("confirmed"):
            _signal(**res)  # a confirmed hit wins immediately
            return
        if res and best is None:
            best = res  # keep the first lead as fallback
    if best:
        _signal(**best)
    else:
        _signal(confirmed=False, vuln_class="http",
                summary=f"Differential probe of {len(params)} parameter(s) found no SQL error, boolean "
                        f"divergence, or unencoded reflection.")


def mode_targeted(url, param, payload, timeout):
    """
    Test ONE human- or model-chosen payload against ONE named parameter with
    the SAME typed oracles the auto battery uses (2026-11-25: extended beyond
    the original SQL-error-only check so --manual-payload's PAYLOAD: mode --
    which always dispatches here -- grades whichever vuln class the human is
    actually testing, not just SQLi).
    """
    base_status, base_body, base_t = _fetch(url, timeout)
    mut_url = _set_param(url, param, payload)
    s, b, t = _fetch(mut_url, timeout)

    # 1) SQL-error oracle.
    err = _has_sql_error(b)
    if err and not _has_sql_error(base_body):
        _signal(confirmed=True, severity="high", vuln_class="sqli", parameter=param,
                summary=f"Targeted tamper on '{param}' triggered SQL error '{err}'.")
        return

    # 2) LFI oracle: an /etc/passwd signature appearing only in the mutated response.
    if _LFI_RE.search(b or "") and not _LFI_RE.search(base_body or ""):
        excerpt = b[max(0, _LFI_RE.search(b).start() - 20):_LFI_RE.search(b).end() + 60]
        _signal(confirmed=True, severity="high", vuln_class="lfi", parameter=param,
                summary=f"Targeted tamper on '{param}' returned an /etc/passwd signature "
                        f"not present in the baseline response.",
                evidence=_redact(excerpt))
        return

    # 3) Open-redirect oracle: if the payload itself looks like an absolute or
    # protocol-relative URL, check whether the mutated request's UN-FOLLOWED
    # Location header host matches the payload's OWN host (not a fixed canary
    # -- the human/model already chose the external host via the payload).
    if "://" in payload or payload.startswith("//"):
        payload_host = (urllib.parse.urlsplit(
            payload if "://" in payload else "//" + payload.lstrip("/")
        ).hostname or "").lower()
        if payload_host:
            status_nr, location = _fetch_no_redirect(mut_url, timeout)
            if status_nr in (301, 302, 303, 307, 308) and location:
                loc_host = (urllib.parse.urlsplit(
                    location if "://" in location or location.startswith("//") else "//" + location
                ).hostname or "").lower()
                if loc_host == payload_host:
                    _signal(confirmed=True, severity="medium", vuln_class="open-redirect", parameter=param,
                            summary=f"Targeted tamper on '{param}' with an externally-controlled URL "
                                    f"produced a redirect whose Location host matches the payload's own host.",
                            evidence=_redact(location))
                    return

    # 4) Reflection LEAD: the payload (or a distinctive quote/tag fragment of it)
    # echoed verbatim and unencoded in the mutated response but not the baseline.
    canary_frag = payload if len(payload) <= 40 else payload[:40]
    if canary_frag and ("<" in canary_frag or '"' in canary_frag or "'" in canary_frag) \
            and canary_frag in (b or "") and canary_frag not in (base_body or ""):
        _signal(confirmed=False, lead=True, severity="low", vuln_class="xss", parameter=param,
                summary=f"Lead: parameter '{param}' reflects the payload unencoded "
                        f"(not present in the baseline response) -- possible XSS surface.")
        return

    sim = _similar(b, base_body)
    _signal(confirmed=False, vuln_class="http", parameter=param,
            summary=f"Targeted tamper on '{param}' (payload len {len(payload)}): status "
                    f"{base_status}->{s}, body-sim {sim:.2f}, timing {base_t:.2f}s->{t:.2f}s. "
                    f"No SQL-error, LFI, open-redirect, or reflection signature.")


def mode_raw(raw_file, host, port, timeout):
    """Send a raw HTTP request over a socket (request-smuggling substrate)."""
    try:
        with open(raw_file, "rb") as f:
            raw = f.read()
    except Exception as e:
        _signal(confirmed=False, vuln_class="http", summary=f"Could not read raw request file: {e}")
        return
    t0 = time.monotonic()
    try:
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        with socket.create_connection((host, port), timeout=timeout) as sock:
            s = ctx.wrap_socket(sock, server_hostname=host) if port == 443 else sock
            s.sendall(raw)
            s.settimeout(timeout)
            chunks = []
            try:
                while len(b"".join(chunks)) < 100_000:
                    d = s.recv(8192)
                    if not d:
                        break
                    chunks.append(d)
            except socket.timeout:
                pass
            resp = b"".join(chunks).decode("utf-8", "replace")
            elapsed = time.monotonic() - t0
            status_line = resp.split("\r\n", 1)[0] if resp else ""
            _signal(confirmed=False, vuln_class="http", raw=True,
                    summary=f"Raw request sent ({len(raw)} bytes). Response: '{status_line}', "
                            f"{len(resp)} bytes in {elapsed:.2f}s. (Smuggling CL.TE/TE.CL detection "
                            f"compares these timing/response differentials across crafted variants.)")
    except Exception as e:
        _signal(confirmed=False, vuln_class="http",
                summary=f"Raw send failed after {time.monotonic()-t0:.2f}s: {e}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rich HTTP differential prober (benign, detection-only).")
    ap.add_argument("--url", help="Target URL (include ?param=value for auto/targeted modes).")
    ap.add_argument("--mode", choices=["auto", "targeted", "raw"], default="auto")
    ap.add_argument("--param", help="Parameter name (targeted mode).")
    ap.add_argument("--payload", help="Payload value (targeted mode).")
    ap.add_argument("--raw-file", help="File with a raw HTTP request (raw mode).")
    ap.add_argument("--host", help="Host for raw mode.")
    ap.add_argument("--port", type=int, default=443, help="Port for raw mode (default 443).")
    ap.add_argument("--timeout", type=float, default=10.0)
    ap.add_argument("--max-requests", type=int, default=12, help="Hard cap on total requests (auto mode).")
    args = ap.parse_args(argv)

    try:
        if args.mode == "raw":
            if not (args.raw_file and args.host):
                _signal(confirmed=False, vuln_class="http", summary="raw mode needs --raw-file and --host.")
                return
            mode_raw(args.raw_file, args.host, args.port, args.timeout)
        elif args.mode == "targeted":
            if not (args.url and args.param and args.payload is not None):
                _signal(confirmed=False, vuln_class="http", summary="targeted mode needs --url --param --payload.")
                return
            mode_targeted(args.url, args.param, args.payload, args.timeout)
        else:
            if not args.url:
                _signal(confirmed=False, vuln_class="http", summary="auto mode needs --url.")
                return
            mode_auto(args.url, args.timeout, max(2, args.max_requests))
    except Exception as e:
        # Never crash without emitting a parseable verdict.
        _signal(confirmed=False, vuln_class="http", summary=f"http_probe internal error: {e}")


if __name__ == "__main__":
    main()
