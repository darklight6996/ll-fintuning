#!/usr/bin/env python3
"""
core/tools/raw_http_runner.py

MODEL-AUTHORED RAW-HTTP engine (Shape C: protocol / parser desync). Lets the
hunter send EXACT request bytes over a raw socket -- the one thing a normal HTTP
client cannot do -- to detect front-end/back-end desync (HTTP request smuggling),
cache / Host-header routing quirks, and anything below the request-line
abstraction. The test logic is the raw bytes the model supplies.

argv in, one JSON line out (a vuln_signal), like the other detection engines, so
it flows through the executor's gates + scope lock unchanged.

SAFETY (detection-only):
  * The TCP connection goes ONLY to the authorized --target host:port. Any Host
    header inside the payload is sent as bytes (so Host-routing is testable) but
    never changes where we connect -- the scope lock holds.
  * Single connection per request, hard socket timeout, response capped. No
    request-queue poisoning loop, no second victim request, no OOB / exfil.
  * The verdict is computed HERE from objective signals (a timing differential vs
    a well-formed control, or a model-named marker echoed back), not asserted by
    the model.
  * GRACEFUL: any socket/TLS error emits a clean non-finding and exits 0.
"""

import argparse
import base64
import json
import socket
import ssl
import sys
import time
from urllib.parse import urlsplit

_MAX_READ = 16384
_FAST = 2.0  # a control returning below this many seconds is "prompt"


def _emit(sig):
    print(json.dumps({"vuln_signal": sig}))


def _parse_target(target):
    t = target if "://" in target else "http://" + target
    u = urlsplit(t)
    tls = (u.scheme == "https")
    return u.hostname, (u.port or (443 if tls else 80)), tls


def _send_once(host, port, tls, raw_bytes, timeout):
    """Open ONE socket to the authorized host:port, send exact bytes, read the
    (bounded) reply. Returns (elapsed_s, data_bytes, timed_out, error_str)."""
    start = time.monotonic()
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        if tls:
            ctx = ssl.create_default_context()
            ctx.check_hostname = False
            ctx.verify_mode = ssl.CERT_NONE
            sock = ctx.wrap_socket(sock, server_hostname=host)
        sock.sendall(raw_bytes)
        sock.settimeout(timeout)
        chunks, total, timed_out = [], 0, False
        try:
            while total < _MAX_READ:
                b = sock.recv(4096)
                if not b:
                    break
                chunks.append(b)
                total += len(b)
        except socket.timeout:
            timed_out = True
        return time.monotonic() - start, b"".join(chunks), timed_out, None
    except Exception as e:
        return time.monotonic() - start, b"", False, f"{type(e).__name__}: {e}"
    finally:
        if sock is not None:
            try:
                sock.close()
            except Exception:
                pass


def run(target, payload, control, marker, timeout):
    host, port, tls = _parse_target(target)
    if not host:
        _emit({"confirmed": False, "vuln_class": "request-smuggling",
               "summary": f"Could not parse a host from target {target!r}."})
        return

    p_elapsed, p_data, p_timeout, p_err = _send_once(host, port, tls, payload, timeout)
    if p_err:
        _emit({"confirmed": False, "vuln_class": "request-smuggling",
               "summary": f"Raw request could not be sent: {p_err}."})
        return
    p_text = p_data.decode("latin-1", "replace")

    # Strongest signal: a model-named marker reflected = the crafted/smuggled bytes landed.
    if marker and marker in p_text:
        _emit({"confirmed": True, "vuln_class": "request-smuggling", "severity": "high",
               "summary": "Raw-HTTP probe: the authored marker was reflected in the response, "
                          "indicating the crafted/smuggled bytes were processed.",
               "evidence": f"marker {marker!r} present in response ({len(p_data)} bytes); "
                           f"response time {p_elapsed:.2f}s."})
        return

    # Timing desync: a well-formed CONTROL returns fast while the desync payload stalls
    # until the socket timeout (the back-end waiting for bytes that never come).
    if control is not None:
        c_elapsed, _c_data, _c_to, c_err = _send_once(host, port, tls, control, timeout)
        if not c_err and c_elapsed < _FAST and (p_timeout or p_elapsed >= timeout * 0.8):
            _emit({"confirmed": True, "vuln_class": "request-smuggling", "severity": "high",
                   "summary": "Raw-HTTP timing desync: the well-formed control returned promptly but "
                              "the crafted request stalled, consistent with a front-end/back-end length "
                              "disagreement (request smuggling).",
                   "evidence": f"control {c_elapsed:.2f}s vs payload {p_elapsed:.2f}s"
                               f"{' (payload read timed out)' if p_timeout else ''}; "
                               f"fast<{_FAST}s, timeout {timeout}s."})
            return
        _emit({"confirmed": False, "lead": bool(p_timeout), "vuln_class": "request-smuggling",
               "severity": "low",
               "summary": "Raw-HTTP probe ran; no clear timing desync "
                          f"(control {c_elapsed:.2f}s, payload {p_elapsed:.2f}s"
                          f"{', payload timed out' if p_timeout else ''}). Adjust the CL/TE variant or marker."})
        return

    _emit({"confirmed": False, "vuln_class": "request-smuggling", "severity": "low",
           "summary": f"Raw-HTTP probe sent; response {len(p_data)} bytes in {p_elapsed:.2f}s"
                      f"{' (read timed out)' if p_timeout else ''}. Provide a CONTROL request for a timing "
                      "comparison, or a marker to detect a reflected smuggled prefix."})


def main(argv=None):
    ap = argparse.ArgumentParser(description="Model-authored raw-HTTP desync probe (detection-only).")
    ap.add_argument("--target", required=True)
    ap.add_argument("--payload-b64", required=True)
    ap.add_argument("--control-b64", default=None)
    ap.add_argument("--marker", default=None)
    ap.add_argument("--timeout", type=int, default=10)
    args = ap.parse_args(argv)
    timeout = max(3, min(args.timeout, 15))
    payload = base64.b64decode(args.payload_b64)
    control = base64.b64decode(args.control_b64) if args.control_b64 else None
    try:
        run(args.target, payload, control, args.marker, timeout)
    except Exception as e:
        _emit({"confirmed": False, "vuln_class": "request-smuggling",
               "summary": f"Raw-HTTP runner error: {type(e).__name__}: {e}."})


if __name__ == "__main__":
    main()
