#!/usr/bin/env python3
"""
core/tools/browser_probe.py

THE HEADLESS-BROWSER PRIMITIVE (engine) -- the COVERAGE engine.

curl sees the raw HTML a server returns; a real app's attack surface only exists
after JavaScript runs -- routes mounted client-side, endpoints called by XHR/
fetch, forms and parameters injected by a framework, URLs buried in JS bundles.
This renders pages in headless Chromium (Playwright) and harvests ALL of it:
links, forms + their inputs, every network request the page makes, and
endpoint-like strings mined out of inline + external JavaScript.

SHALLOW CRAWL (2026-11-16): a single page is not enough -- the interesting
parameters (e.g. a stock-check `stockApi` URL param, a `productId`) live on
*inner* pages, not the homepage. So after harvesting the entry page we follow a
bounded set of same-host links (depth 1, capped) and harvest each. This is the
"inspect every nook and cranny -- every endpoint, input, request" surface the
hunter reasons over; most real bugs are known classes in places nobody looked.

Invoked as an external tool (argv in, JSON out) like the other engines, so it
flows through the executor's gates + scope lock unchanged, and it is
framework-level (any agent can register it).

OUTPUT (one JSON object on stdout, consumed by the fact extractor):
  {
    "discovered_endpoints": [...],   # same-host links + form actions (coverage)
    "forms": [{action, method, inputs:[...]}],
    "network_endpoints": [...],      # URLs the page actually requested (XHR/fetch/assets)
    "js_endpoints": [...],           # path/URL strings mined from JS
    "parameters": [...],             # query-param + form-input names seen across all of the above
    "pages_crawled": [...],          # the URLs actually rendered
    "vuln_signal": {confirmed:false, ...}   # a coverage summary; never a fake finding
  }

SAFETY / STEALTH
  * Read-only: navigations + observation only. No form submission, no clicking
    destructive controls, no credentials.
  * Bounded: capped page count, hard per-page timeout, capped harvest sizes.
  * GRACEFUL: if Playwright/Chromium isn't usable (not installed, or missing
    system libs), it emits a clean JSON note with the REAL root cause and exits 0
    -- it never crashes the run.
"""

import argparse
import json
import re
import sys
from urllib.parse import urlsplit, urljoin, parse_qsl

_URL_IN_JS = re.compile(r"""["'`](/[A-Za-z0-9_\-./]{2,160}|https?://[A-Za-z0-9_\-./:?=&%]{4,200})["'`]""")
_HARVEST_CAP = 300
_CRAWL_MAX_PAGES = 8      # entry page + up to this many same-host inner pages
_CRAWL_MAX_FANOUT = 40    # consider at most this many candidate links for the frontier


def _same_host(u, host):
    try:
        netloc = urlsplit(u).netloc
        return netloc == host or netloc == ""
    except Exception:
        return False


def _params_of(url):
    try:
        return [k for k, _ in parse_qsl(urlsplit(url).query, keep_blank_values=True)]
    except Exception:
        return []


def _emit(payload):
    print(json.dumps(payload))


def _root_cause(exc_text: str) -> str:
    """
    Pull the ACTUAL reason out of a Playwright launch exception. Playwright hides
    the real cause inside a multi-line "Browser logs:" block (e.g. a missing
    shared library), while the top line is only the useless generic
    "Target page, context or browser has been closed". Surface the informative
    line so an environment problem self-diagnoses in the report instead of looking
    like a mysterious crash.
    """
    text = exc_text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    # Most informative first: a missing shared library is the classic WSL case.
    for ln in lines:
        if "error while loading shared libraries" in ln:
            # e.g. "...chrome-headless-shell: error while loading shared libraries: libasound.so.2: cannot open..."
            m = re.search(r"(lib[\w.+-]+\.so[\w.]*)", ln)
            lib = m.group(1) if m else "a shared library"
            return (f"Chromium launched but a system library is missing ({lib}). "
                    f"Install the browser's OS deps: `playwright install-deps chromium` "
                    f"(or apt-get the lib*t64 packages).")
    for needle in ("error while loading", "cannot open shared object", "[err]",
                   "No usable sandbox", "Failed to move to new namespace"):
        for ln in lines:
            if needle in ln:
                return ln[:200]
    # Fall back to the first line, trimmed.
    return (lines[0][:200] if lines else "unknown browser launch failure")


def _harvest(page, url, host, endpoints, forms, js_endpoints, params):
    """Harvest coverage from one already-navigated page into the shared sets."""
    # Links (same host).
    try:
        for href in page.eval_on_selector_all("a[href]", "els => els.map(e => e.href)") or []:
            if href and _same_host(href, host):
                endpoints.add(href)
                for pr in _params_of(href):
                    params.add(pr)
    except Exception:
        pass

    # Forms + inputs (action is a strong endpoint signal; input names are params).
    try:
        raw_forms = page.eval_on_selector_all(
            "form",
            """els => els.map(f => ({
                action: f.action || '',
                method: (f.method || 'get').toLowerCase(),
                inputs: Array.from(f.querySelectorAll('input,textarea,select'))
                             .map(i => i.name).filter(Boolean)
            }))""",
        ) or []
        for f in raw_forms[:50]:
            forms.append(f)
            if f.get("action") and _same_host(f["action"], host):
                endpoints.add(f["action"])
            for nm in f.get("inputs", []):
                params.add(nm)
    except Exception:
        pass

    # Mine JS: external script srcs + inline + the rendered HTML.
    try:
        html = page.content()
    except Exception:
        html = ""
    try:
        srcs = page.eval_on_selector_all("script[src]", "els => els.map(e => e.src)") or []
    except Exception:
        srcs = []
    for s in srcs:
        if s and _same_host(s, host):
            endpoints.add(s)
    for m in _URL_IN_JS.findall(html or ""):
        frag = m if isinstance(m, str) else (m[0] if m else "")
        if frag:
            joined = urljoin(url, frag) if frag.startswith("/") else frag
            js_endpoints.add(joined)
            for pr in _params_of(joined):
                params.add(pr)
            if len(js_endpoints) >= _HARVEST_CAP:
                break


def run(url, timeout_ms, max_pages=_CRAWL_MAX_PAGES):
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        _emit({"vuln_signal": {
            "confirmed": False, "vuln_class": "coverage",
            "summary": "Headless browser (Playwright) is not installed yet; skipping DOM/JS coverage. "
                       "Install it at setup time (pip install playwright && playwright install chromium)."}})
        return

    host = urlsplit(url).netloc
    network = []
    endpoints = set()
    forms = []
    js_endpoints = set()
    params = set()
    pages_crawled = []

    for p in _params_of(url):
        params.add(p)

    try:
        with sync_playwright() as pw:
            # WSL/container-hardened launch: --no-sandbox + chromium_sandbox=False
            # (no user namespaces), --disable-dev-shm-usage (small /dev/shm),
            # --disable-gpu/--disable-software-rasterizer (no GPU in headless WSL).
            browser = pw.chromium.launch(
                headless=True,
                chromium_sandbox=False,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
                      "--disable-gpu", "--disable-software-rasterizer"],
            )
            ctx = browser.new_context(ignore_https_errors=True)
            page = ctx.new_page()
            page.on("request", lambda req: network.append(req.url) if len(network) < _HARVEST_CAP else None)

            def _goto(u, nav_timeout, settle_ms=0):
                # domcontentloaded (fast, reliable) -- NOT networkidle, which never
                # settles on pages with analytics/keepalive/polling and made the crawl
                # bail with 0 harvested pages even though the DOM was fully present
                # (observed live on a PortSwigger shop, 2026-11-19). On timeout we still
                # harvest if the document actually reached interactive/complete.
                loaded = False
                try:
                    page.goto(u, wait_until="domcontentloaded", timeout=nav_timeout)
                    loaded = True
                except Exception:
                    try:
                        state = page.evaluate("document.readyState")
                        loaded = state in ("interactive", "complete")
                    except Exception:
                        loaded = False
                if loaded and settle_ms:
                    try:
                        page.wait_for_timeout(settle_ms)  # let late JS/XHR render
                    except Exception:
                        pass
                return loaded

            # Entry page gets the full timeout; inner pages get a tight cap so the
            # whole crawl stays inside the action's subprocess budget.
            inner_timeout = min(timeout_ms, 8000)

            # Entry page.
            if _goto(url, timeout_ms, settle_ms=1200):
                pages_crawled.append(url)
                _harvest(page, url, host, endpoints, forms, js_endpoints, params)

            # Shallow crawl: visit a bounded frontier of same-host links discovered
            # on the entry page, so inner pages' params/forms/JS get covered too.
            frontier = []
            seen = {url.split("#")[0]}
            for e in sorted(endpoints):
                base = e.split("#")[0]
                if base in seen:
                    continue
                # Skip obvious non-HTML assets to spend the budget on real pages.
                if re.search(r"\.(css|js|png|jpe?g|gif|svg|ico|woff2?|ttf|map|pdf)(\?|$)", base.lower()):
                    continue
                frontier.append(base)
                seen.add(base)
                if len(frontier) >= _CRAWL_MAX_FANOUT:
                    break

            budget = max(0, max_pages - len(pages_crawled))
            for u in frontier[:budget]:
                if _goto(u, inner_timeout, settle_ms=800):
                    pages_crawled.append(u)
                    _harvest(page, u, host, endpoints, forms, js_endpoints, params)

            browser.close()
    except Exception as e:
        _emit({"vuln_signal": {
            "confirmed": False, "vuln_class": "coverage",
            "summary": f"Headless browser run failed: {_root_cause(str(e))}"}})
        return

    net_same = sorted({u for u in network if _same_host(u, host)})[:_HARVEST_CAP]
    for u in net_same:
        for pr in _params_of(u):
            params.add(pr)

    result = {
        "discovered_endpoints": sorted(endpoints)[:_HARVEST_CAP],
        "forms": forms,
        "network_endpoints": net_same,
        "js_endpoints": sorted(js_endpoints)[:_HARVEST_CAP],
        "parameters": sorted(params)[:_HARVEST_CAP],
        "pages_crawled": pages_crawled,
        "vuln_signal": {
            "confirmed": False, "vuln_class": "coverage",
            "summary": (f"Browser coverage of {host}: crawled {len(pages_crawled)} page(s); "
                        f"{len(endpoints)} link/form endpoint(s), {len(net_same)} network endpoint(s), "
                        f"{len(js_endpoints)} JS-mined endpoint(s), {len(params)} parameter name(s) "
                        f"discovered."),
        },
    }
    _emit(result)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Headless-browser coverage primitive (read-only, shallow crawl).")
    ap.add_argument("--url", required=True)
    ap.add_argument("--timeout-ms", type=int, default=20000)
    ap.add_argument("--max-pages", type=int, default=_CRAWL_MAX_PAGES)
    args = ap.parse_args(argv)
    try:
        run(args.url, args.timeout_ms, max_pages=args.max_pages)
    except Exception as e:
        _emit({"vuln_signal": {"confirmed": False, "vuln_class": "coverage",
                               "summary": f"browser_probe internal error: {_root_cause(str(e))}"}})


if __name__ == "__main__":
    main()
