#!/usr/bin/env python3
"""
core/tools/browser_exec_runner.py

MODEL-AUTHORED BROWSER-EXECUTION engine (Shape D: client / browser context). Loads
one authorized URL in headless Chromium, EXECUTES the page JavaScript, and reports
whether an injected benign payload actually RAN -- the gold-standard oracle for
DOM / reflected / stored XSS: real script execution in the target origin, not a
reflection-diff guess.

The model authors the payload (embedded in the URL) plus a unique MARKER its benign
canary surfaces (alert('MARKER') / console.log('MARKER')), and optionally a JS
detector expression evaluated after load. The verdict is computed HERE from what
actually fired, not asserted by the model.

argv in, one JSON line out (a vuln_signal), like the other detection engines.

SAFETY (detection-only):
  * Loads ONLY the authorized --url; a benign marker canary; dialogs are captured
    and dismissed (never blocks). No exfiltration, no second victim, no weaponized
    payload -- it observes EXECUTION of a marker, nothing more.
  * Bounded nav timeout; headless; WSL-hardened launch flags.
  * GRACEFUL: if Playwright/Chromium is unavailable, emits a clean non-finding.
"""

import argparse
import base64
import json
import re
import sys


def _emit(sig):
    print(json.dumps({"vuln_signal": sig}))


def _root_cause(exc_text):
    text = exc_text or ""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines:
        if "error while loading shared libraries" in ln:
            m = re.search(r"(lib[\w.+-]+\.so[\w.]*)", ln)
            return (f"Chromium is missing a system library ({m.group(1) if m else 'a lib'}); "
                    f"run `playwright install-deps chromium`.")
    return (lines[0][:200] if lines else "unknown browser failure")


def run(url, marker, detector, click_selector, timeout_ms):
    try:
        from playwright.sync_api import sync_playwright
    except Exception:
        _emit({"confirmed": False, "vuln_class": "xss",
               "summary": "Headless browser (Playwright) is not installed; cannot run browser_exec. "
                          "Install at setup (pip install playwright && playwright install chromium)."})
        return

    dialogs, consoles, errors = [], [], []
    detector_result, detector_ok = None, False

    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                headless=True, chromium_sandbox=False,
                args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage",
                      "--disable-gpu", "--disable-software-rasterizer"],
            )
            ctx = browser.new_context(ignore_https_errors=True)
            page = ctx.new_page()

            def _on_dialog(d):
                try:
                    dialogs.append({"type": d.type, "message": d.message})
                finally:
                    try:
                        d.dismiss()
                    except Exception:
                        pass
            page.on("dialog", _on_dialog)
            page.on("console", lambda m: consoles.append(m.text) if len(consoles) < 100 else None)
            page.on("pageerror", lambda e: errors.append(str(e)) if len(errors) < 50 else None)

            try:
                page.goto(url, wait_until="domcontentloaded", timeout=timeout_ms)
            except Exception:
                pass
            try:
                page.wait_for_timeout(1200)  # let late JS / event-driven sinks fire
            except Exception:
                pass

            if click_selector:
                try:
                    page.click(click_selector, timeout=3000)
                    page.wait_for_timeout(800)
                except Exception:
                    pass

            if detector:
                try:
                    detector_result = page.evaluate(detector)
                    detector_ok = True
                except Exception as e:
                    detector_result, detector_ok = f"{type(e).__name__}: {e}", False
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "xss",
               "summary": f"browser_exec launch/navigation failed: {_root_cause(str(e))}"})
        return

    # Oracle: a dialog firing, the marker surfaced in a dialog/console, or a truthy
    # detector == the injected script EXECUTED in the target origin == XSS proven.
    if marker:
        for d in dialogs:
            if marker in (d.get("message") or ""):
                _emit({"confirmed": True, "vuln_class": "xss", "severity": "high",
                       "summary": "browser_exec: injected canary EXECUTED (script ran in the target origin).",
                       "evidence": f"{d['type']} dialog message contained {marker!r}."})
                return
        for c in consoles:
            if marker in c:
                _emit({"confirmed": True, "vuln_class": "xss", "severity": "high",
                       "summary": "browser_exec: injected canary EXECUTED (console write from the payload).",
                       "evidence": f"console output contained {marker!r}."})
                return

    if dialogs:
        d = dialogs[0]
        _emit({"confirmed": True, "vuln_class": "xss", "severity": "high",
               "summary": "browser_exec: a JavaScript dialog fired from the injected payload (script executed).",
               "evidence": f"{d['type']} dialog: {str(d.get('message'))[:160]!r}."})
        return

    if detector and detector_ok and detector_result:
        _emit({"confirmed": True, "vuln_class": "xss", "severity": "high",
               "summary": "browser_exec: the authored JS detector returned truthy in the target origin.",
               "evidence": f"detector result: {str(detector_result)[:160]!r}."})
        return

    _emit({"confirmed": False, "vuln_class": "xss", "severity": "low",
           "summary": ("browser_exec: payload did not execute (no dialog / marker / detector hit). "
                       f"{len(dialogs)} dialog(s), {len(consoles)} console line(s), {len(errors)} page error(s). "
                       "Adjust the injection context (HTML / attribute / JS-string / URL) or the sink trigger.")})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model-authored headless-browser execution probe (detection-only).")
    ap.add_argument("--url", required=True)
    ap.add_argument("--marker", default=None)
    ap.add_argument("--detector-b64", default=None)
    ap.add_argument("--click-selector", default=None)
    ap.add_argument("--timeout-ms", type=int, default=15000)
    args = ap.parse_args(argv)
    detector = base64.b64decode(args.detector_b64).decode("utf-8", "replace") if args.detector_b64 else None
    try:
        run(args.url, args.marker, detector, args.click_selector, max(3000, min(args.timeout_ms, 25000)))
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "xss",
               "summary": f"browser_exec internal error: {_root_cause(str(e))}"})


if __name__ == "__main__":
    main()
