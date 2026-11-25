#!/usr/bin/env python3
"""
run_portswigger_batch.py

Run the BugBountyAgent against a LIST of PortSwigger lab instances in ONE
process, to validate that the autonomous hunter generalizes across vuln classes
(SSRF, SQLi, XSS, IDOR, ...) rather than being tuned to a single lab. The GPU
model loads once and is reused for every lab (run_portswigger_lab.run_lab).

LAB LIST (--file): one lab per line. Blank lines and lines starting with '#'
are ignored. Each line is:

    <url>
    <url> | <name>
    <url> | <name> | <expected_class>

<expected_class> (optional, e.g. "ssrf", "sqli") is only used to label the
summary -- it does NOT change how the agent runs (no answer is fed to it).

Each lab instance URL must be LIVE (open the lab in your browser first;
PortSwigger mints a fresh hash per instance and tears old ones down). Dead/
unreachable URLs are skipped with a note instead of wasting a model run.

Usage:
    python3 run_portswigger_batch.py --file labs.txt [--max-steps 30]
    python3 run_portswigger_batch.py <url1> <url2> ... [--max-steps 30]

Unattended note: each self_script/nuclei_template still hits the interactive
[y/N] HIGH-RISK gate unless the operator has explicitly enabled lab
auto-approval. Without that, stay at the keyboard to approve, or test fewer
labs per batch.
"""
import argparse
import socket
import ssl
import sys
import urllib.request
from urllib.parse import urlparse

from run_portswigger_lab import run_lab


def _parse_lab_line(line: str):
    parts = [p.strip() for p in line.split("|")]
    url = parts[0]
    name = parts[1] if len(parts) > 1 and parts[1] else url
    expected = parts[2].lower() if len(parts) > 2 and parts[2] else ""
    return url, name, expected


def _load_labs(args):
    labs = []
    if args.file:
        with open(args.file, encoding="utf-8") as f:
            for raw in f:
                line = raw.strip()
                if not line or line.startswith("#"):
                    continue
                labs.append(_parse_lab_line(line))
    for u in args.urls:
        labs.append((u, u, ""))
    return labs


def _reachable(url: str, timeout: int = 10) -> bool:
    """Quick liveness check so a dead lab instance doesn't burn a model run."""
    if "://" not in url:
        url = "https://" + url
    ctx = ssl.create_default_context()
    try:
        req = urllib.request.Request(url, method="GET", headers={"User-Agent": "ll-hunt-batch"})
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            return 200 <= resp.status < 500   # any real HTTP answer means it's up
    except urllib.error.HTTPError as e:
        return 200 <= e.code < 500            # 4xx still means the host is serving
    except (urllib.error.URLError, socket.timeout, ssl.SSLError, ConnectionError, ValueError):
        return False


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("urls", nargs="*", help="Lab instance URLs (or use --file).")
    ap.add_argument("--file", help="Path to a lab list file (one lab per line).")
    ap.add_argument("--max-steps", type=int, default=30, help="Max autonomous steps per lab (default 30).")
    ap.add_argument("--no-preflight", action="store_true", help="Skip the reachability check.")
    args = ap.parse_args()

    labs = _load_labs(args)
    if not labs:
        ap.error("No labs given. Provide URLs positionally or via --file.")

    print("=" * 70)
    print(f"PORTSWIGGER BATCH: {len(labs)} lab(s), max_steps={args.max_steps} each")
    print("=" * 70)

    summaries = []
    for i, (url, name, expected) in enumerate(labs, 1):
        print(f"\n########## LAB {i}/{len(labs)}: {name} ##########")
        if not args.no_preflight and not _reachable(url):
            print(f"[SKIP] {url} is not reachable (dead/expired instance?). Skipping.")
            summaries.append({"lab_name": name, "lab_url": url, "status": "skipped_unreachable",
                              "steps_run": 0, "findings_count": 0, "confirmed_vuln": False,
                              "expected": expected})
            continue
        try:
            s = run_lab(url, name, args.max_steps, verbose=True)
        except KeyboardInterrupt:
            print("\n[ABORTED by operator]")
            break
        except Exception as e:
            print(f"[ERROR] lab run raised: {e}")
            s = {"lab_name": name, "lab_url": url, "status": f"error:{type(e).__name__}",
                 "steps_run": 0, "findings_count": 0, "confirmed_vuln": False}
        s["expected"] = expected
        summaries.append(s)

    # --- summary table ---
    print("\n\n" + "=" * 78)
    print("BATCH SUMMARY")
    print("=" * 78)
    hdr = f"{'#':>2}  {'CONFIRMED':9}  {'status':22}  {'steps':>5}  {'find':>4}  {'expect':8}  name"
    print(hdr)
    print("-" * 78)
    confirmed = 0
    for i, s in enumerate(summaries, 1):
        mark = "YES" if s.get("confirmed_vuln") else "no"
        if s.get("confirmed_vuln"):
            confirmed += 1
        print(f"{i:>2}  {mark:9}  {str(s.get('status'))[:22]:22}  "
              f"{str(s.get('steps_run')):>5}  {str(s.get('findings_count')):>4}  "
              f"{(s.get('expected') or '-')[:8]:8}  {s.get('lab_name')}")
    print("-" * 78)
    print(f"Confirmed a real vulnerability on {confirmed}/{len(summaries)} lab(s).")
    print("(CONFIRMED = a probe/engine emitted vuln_signal.confirmed=true; "
          "findings_count also counts info-severity phase milestones.)")


if __name__ == "__main__":
    main()
