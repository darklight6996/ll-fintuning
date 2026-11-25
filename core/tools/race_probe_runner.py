#!/usr/bin/env python3
"""
core/tools/race_probe_runner.py

RELIABLE RACE / LIMIT-OVERRUN engine (Shape B). The model kept hand-rolling
CSRF + session + threading in self_script and failing on the CSRF scrape; this
engine does that plumbing robustly and uses an OBJECTIVE oracle:

  * CONTROL  = fire the action COUNT times SEQUENTIALLY (re-priming the CSRF each
               time). This is how many the server allows when its limit IS enforced.
  * TEST     = fire the action COUNT times CONCURRENTLY (shared token, released by a
               barrier for maximum simultaneity).
  * CONFIRM  = concurrent successes EXCEED the sequential control (and > 1): the race
               window defeated a limit the sequential run respected == overrun.

The verdict is computed HERE from the two counts, not asserted by the model. argv
in, one JSON vuln_signal line out, so it flows through the executor gates + scope
lock unchanged.

SAFETY: like self_script it performs repeated state-changing requests (inherent to
race testing) against ONLY the authorized target, behind the HIGH-RISK approval gate.
Detection posture: it counts successes; it does not escalate, exfiltrate, or loop
beyond COUNT (hard-capped).
"""

import argparse
import base64
import json
import re
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

_CSRF_NAMES = r"(?:csrf|csrf_token|csrfmiddlewaretoken|_token|token|authenticity_token|nonce)"


def _emit(sig):
    print(json.dumps({"vuln_signal": sig}))


def _extract_token(html):
    if not html:
        return None
    for pat in (rf'name=["\']?{_CSRF_NAMES}["\']?[^>]*?value=["\']([^"\']+)["\']',
                rf'value=["\']([^"\']+)["\'][^>]*?name=["\']?{_CSRF_NAMES}["\']?'):
        m = re.search(pat, html, re.I)
        if m:
            return m.group(1)
    return None


def _success(resp, marker):
    # A 2xx/3xx is a successful state-change (add-to-cart redirects 302 with an empty
    # body, so a body-marker must NEVER negate a status success -- that false-negatived
    # a whole run on 2026-11-21). A marker, when given, only ADDS a positive signal.
    try:
        if resp.status_code in (200, 302, 303):
            return True
        if marker and marker in resp.text:
            return True
        return False
    except Exception:
        return False


def _sequential(requests, target, prime_url, params, method, count, marker, timeout):
    s = requests.Session()
    ok = 0
    for _ in range(count):
        try:
            tok = _extract_token(s.get(prime_url, timeout=timeout).text)
            data = dict(params)
            if tok:
                data.setdefault("csrf", tok)
            r = s.request(method, target, data=data, timeout=timeout, allow_redirects=False)
            if _success(r, marker):
                ok += 1
        except Exception:
            pass
    return ok


def _concurrent(requests, target, prime_url, params, method, count, marker, timeout):
    s = requests.Session()
    try:
        tok = _extract_token(s.get(prime_url, timeout=timeout).text)
    except Exception:
        tok = None
    data = dict(params)
    if tok:
        data.setdefault("csrf", tok)
    try:
        barrier = threading.Barrier(count, timeout=timeout)
    except Exception:
        barrier = None
    results = []

    def fire():
        if barrier is not None:
            try:
                barrier.wait()
            except Exception:
                pass
        try:
            r = s.request(method, target, data=dict(data), timeout=timeout, allow_redirects=False)
            results.append(_success(r, marker))
        except Exception:
            results.append(False)

    with ThreadPoolExecutor(max_workers=count) as ex:
        for _ in range(count):
            ex.submit(fire)
    return sum(1 for x in results if x), (tok is not None)


def run(target, params, method, prime_url, count, marker, timeout):
    try:
        import requests
    except Exception:
        _emit({"confirmed": False, "vuln_class": "race-condition",
               "summary": "requests is not installed; cannot run race_probe."})
        return
    prime_url = prime_url or target

    control_ok = _sequential(requests, target, prime_url, params, method, count, marker, timeout)
    conc_ok, had_token = _concurrent(requests, target, prime_url, params, method, count, marker, timeout)

    if control_ok == 0 and conc_ok == 0:
        _emit({"confirmed": False, "vuln_class": "race-condition",
               "summary": ("No request succeeded sequentially OR concurrently -- the action likely needs "
                           "login/auth or different endpoint/params"
                           + ("" if had_token else "; NO CSRF token was found on the prime page (check the endpoint or log in first)")
                           + "."),
               "evidence": f"sequential_successes=0, concurrent_successes=0, count={count}, csrf_found={had_token}"})
        return
    if conc_ok > control_ok and conc_ok > 1:
        _emit({"confirmed": True, "vuln_class": "race-condition", "severity": "high",
               "summary": "Race / limit-overrun CONFIRMED: concurrency defeated a limit the sequential control enforced.",
               "evidence": f"sequential allowed {control_ok} success(es); {count} CONCURRENT requests allowed "
                           f"{conc_ok} -- the extra successes are the overrun."})
        return
    _emit({"confirmed": False, "vuln_class": "race-condition", "severity": "low",
           "summary": "No overrun: concurrency did not exceed the sequential limit.",
           "evidence": f"sequential={control_ok}, concurrent={conc_ok}, count={count}"})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Reliable race / limit-overrun probe (sequential control vs concurrent).")
    ap.add_argument("--target", required=True)
    ap.add_argument("--params-b64", required=True)
    ap.add_argument("--method", default="POST")
    ap.add_argument("--prime-url", default=None)
    ap.add_argument("--count", type=int, default=20)
    ap.add_argument("--success-marker", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    a = ap.parse_args(argv)
    try:
        params = json.loads(base64.b64decode(a.params_b64).decode("utf-8", "replace"))
        if not isinstance(params, dict):
            params = {}
    except Exception:
        params = {}
    count = max(2, min(a.count, 40))
    try:
        run(a.target, params, a.method.upper(), a.prime_url, count, a.success_marker, max(3, min(a.timeout, 15)))
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "race-condition",
               "summary": f"race_probe internal error: {type(e).__name__}: {e}"})


if __name__ == "__main__":
    main()
