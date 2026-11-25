#!/usr/bin/env python3
"""
core/tools/bounty_platform.py

BUG-BOUNTY PROGRAM SCOPE/POLICY FETCHER (read-only engagement setup).

Re-implemented (2026-11-25 4-repo harvest, asset 7) from pentest-agents'
mcp-bounty-server (github.com/H-mmer/pentest-agents) -- that repo ships no
LICENSE file anywhere in its tree (its pyproject.toml MIT classifier is not
backed by an actual license grant/text), so this is a from-scratch
re-implementation of its READ-ONLY tool set against our own data shapes, not
a port of its source; attributed here and in work.md.

WHAT WAS PORTED vs. DELIBERATELY NOT PORTED
  Ported: get_program_scope / get_program_policy / sync_program's fail-closed
  placeholder-scope design -- all three use HackerOne's OFFICIAL, documented
  REST Hacker API (https://api.hackerone.com/v1) only.

  NOT ported: search_hacktivity. The source implements this via an
  UNDOCUMENTED website GraphQL endpoint (hackerone.com/graphql), seeding an
  anonymous session and scraping a CSRF token out of a marketing page's HTML
  to mimic the browser SPA's own internal API calls -- because HackerOne's
  official REST API does not expose cross-researcher hacktivity data at all.
  This is the same category of fragile, ToS-grey, reverse-engineered scraping
  this framework has already excluded twice before (DNS Dumpster scraping and
  fake JA3 fingerprinting, ARTEX merge round 1) -- so it is excluded here too,
  on the same principle, rather than silently reimplemented. See work.md's
  2026-11-25 harvest entry for the full rationale.

  NOT ported: draft_report / submit_report (both excluded per the harvest
  task's hard rules -- no code path in this framework submits a bug report to
  a platform on the operator's behalf; drafting/submission stays a manual,
  human action outside this tool entirely).

  NOT ported: Bugcrowd/Intigriti/YesWeHack providers (the source's Bugcrowd
  provider needs a stateful email+password+TOTP login flow -- out of scope
  for a first pass). --platform accepts only "hackerone" today; any other
  value returns a clear "not configured" signal, never a fabricated result.

SAFETY / APPROVAL
  This is the ONLY action in this framework that calls a THIRD-PARTY SaaS
  platform (not the engagement target) using the operator's own real account
  credentials. It is registered in agents/bug_bounty/actions.py ONLY when the
  operator has explicitly set PENTEST_ENABLE_BOUNTY_PLATFORM=1 -- the same
  opt-in-env-var gate self_scripting.py uses -- on top of needing
  HACKERONE_USERNAME/HACKERONE_TOKEN to actually be configured. Read-only GET
  requests only; sync_program writes LOCAL files, never calls any mutating
  platform endpoint.

MODES
  scope   Fetch in-scope/out-of-scope assets for one program.
  policy  Fetch the program's policy text + safe-harbor status.
  sync    Fetch both and write them to local files (bounty_scope.yaml,
          bounty_policy.md) in --target-dir. On ANY API failure, writes a
          placeholder file carrying a machine-readable `scope_mode:
          placeholder` sentinel (re-implemented from the source's
          scope_check "refuse an unavailable-scope file" design) so a
          placeholder can never be silently mistaken for "scope is
          legitimately empty" by anything that later reads it.

OUTPUT
  A single JSON object on stdout: {"bounty": {...}}. Never raises without
  emitting a parseable result; credentials are read from env and NEVER
  echoed into output, logs, or error messages.
"""

import argparse
import json
import os
import sys
from base64 import b64encode
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

API_BASE = "https://api.hackerone.com/v1"


def _result(**kw):
    print(json.dumps({"bounty": kw}))


def _auth_header(username, token):
    creds = b64encode(f"{username}:{token}".encode()).decode()
    return f"Basic {creds}"


def _get(path, username, token, timeout=30):
    """Authenticated GET against HackerOne's official REST Hacker API.
    Returns a parsed JSON dict, {} on 404, or raises PermissionError on
    401/403 (credentials are never included in the exception message)."""
    req = Request(
        f"{API_BASE}{path}",
        headers={"Authorization": _auth_header(username, token), "Accept": "application/json"},
    )
    try:
        with urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        if e.code == 401:
            raise PermissionError("HackerOne: invalid credentials (check HACKERONE_USERNAME/HACKERONE_TOKEN).")
        if e.code == 403:
            raise PermissionError("HackerOne: forbidden -- token may lack access to this program.")
        if e.code == 404:
            return {}
        raise


def _get_paginated(path, username, token, max_pages=10, timeout=30):
    items = []
    current = path
    for _ in range(max_pages):
        data = _get(current, username, token, timeout)
        page_items = data.get("data", [])
        if not page_items:
            break
        items.extend(page_items)
        next_url = (data.get("links") or {}).get("next")
        if not next_url:
            break
        current = next_url.replace(API_BASE, "") if next_url.startswith("http") else next_url
    return items


def fetch_scope(program, username, token, timeout=30):
    """Fetch program info + structured scopes. Returns a dict or raises
    PermissionError on bad credentials; a program that doesn't exist (404)
    returns None."""
    prog = _get(f"/hackers/programs/{quote(program)}", username, token, timeout)
    if not prog:
        return None
    attrs = (prog.get("data") or {}).get("attributes", {})

    scope_items = _get_paginated(
        f"/hackers/programs/{quote(program)}/structured_scopes?page%5Bsize%5D=100", username, token, timeout=timeout
    )
    in_scope, out_of_scope = [], []
    for item in scope_items:
        a = item.get("attributes", {})
        asset = {
            "asset": a.get("asset_identifier", ""),
            "asset_type": a.get("asset_type", "OTHER"),
            "eligible_for_bounty": bool(a.get("eligible_for_bounty", False)),
            "max_severity": a.get("max_severity"),
        }
        (in_scope if a.get("eligible_for_submission", True) else out_of_scope).append(asset)

    return {
        "program_handle": program,
        "program_name": attrs.get("name", program),
        "program_url": f"https://hackerone.com/{program}",
        "in_scope": in_scope,
        "out_of_scope": out_of_scope,
        "last_updated": attrs.get("updated_at", ""),
    }


def fetch_policy(program, username, token, timeout=30):
    """Fetch policy text from the OFFICIAL REST program-info response only.
    Some programs' REST payload has an empty `policy` field (they use
    structured policy sections the website renders client-side) -- that is a
    real, documented API limitation, reported honestly rather than worked
    around via the undocumented website GraphQL endpoint (see module
    docstring)."""
    prog = _get(f"/hackers/programs/{quote(program)}", username, token, timeout)
    if not prog:
        return None
    attrs = (prog.get("data") or {}).get("attributes", {})
    policy_text = (attrs.get("policy") or "").strip()
    submission_state = attrs.get("submission_state")
    restrictions = []
    if submission_state and submission_state != "open":
        restrictions.append(f"Submission state: {submission_state}")
    return {
        "program_handle": program,
        "policy_text": policy_text,
        "policy_text_available": bool(policy_text),
        "safe_harbor": bool(attrs.get("gold_standard_safe_harbor", False)),
        "testing_restrictions": restrictions,
    }


def _scope_to_yaml(scope):
    lines = [f"program: {scope['program_name']!r}", f"platform: hackerone",
              f"url: {scope['program_url']!r}", f"last_updated: {scope['last_updated']!r}", "",
              "in_scope:"]
    for a in scope["in_scope"]:
        lines.append(f"  - asset: {a['asset']!r}")
        lines.append(f"    asset_type: {a['asset_type']}")
        lines.append(f"    eligible_for_bounty: {a['eligible_for_bounty']}")
    lines.append("")
    lines.append("out_of_scope:")
    for a in scope["out_of_scope"]:
        lines.append(f"  - asset: {a['asset']!r}")
        lines.append(f"    asset_type: {a['asset_type']}")
    lines.append("")
    return "\n".join(lines)


def _scope_placeholder_yaml(program, reason):
    """Fail-closed placeholder, re-implemented from the source's
    scope_mode: placeholder sentinel -- a machine-readable flag so this file
    can never be mistaken for a legitimately-empty scope by anything that
    later parses it."""
    return "\n".join([
        "# Auto-generated placeholder: the HackerOne API did not return scope data.",
        "# DO NOT test any target against this file.",
        f"# Reason: {reason}",
        "scope_mode: placeholder",
        f"program: {program!r}",
        "platform: hackerone",
        "in_scope: []",
        "out_of_scope: []",
        "",
    ])


def _policy_to_markdown(policy):
    lines = [f"# Program Policy: {policy['program_handle']}", "",
             f"- Platform: HackerOne", f"- Safe harbor: {'Yes' if policy['safe_harbor'] else 'No/Unknown'}"]
    lines.extend(["", "## Testing Restrictions"])
    if policy["testing_restrictions"]:
        lines.extend(f"- {r}" for r in policy["testing_restrictions"])
    else:
        lines.append("- (none returned by API)")
    lines.extend(["", "## Full Policy Text"])
    if policy["policy_text_available"]:
        lines.append(policy["policy_text"])
    else:
        lines.append(
            "No structured policy text was returned by HackerOne's REST API for this program "
            "(some programs use policy sections the website renders client-side). Review the "
            "program page manually before active testing."
        )
    lines.append("")
    return "\n".join(lines)


def _policy_placeholder_markdown(program, reason):
    return "\n".join([
        f"# Program Policy: {program}", "", "- Platform: HackerOne",
        f"- Policy status: unavailable from API ({reason})", "",
        "Review the program page manually before active testing.", "",
    ])


def do_sync(platform, program, username, token, target_dir, timeout=30):
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    results = []

    try:
        scope = fetch_scope(program, username, token, timeout)
    except PermissionError as e:
        scope, scope_error = None, str(e)
    else:
        scope_error = None if scope is not None else f"Program '{program}' not found on HackerOne."

    if scope is not None:
        (target / "bounty_scope.yaml").write_text(_scope_to_yaml(scope), encoding="utf-8")
        results.append(f"scope written ({len(scope['in_scope'])} in-scope, {len(scope['out_of_scope'])} out-of-scope)")
    else:
        (target / "bounty_scope.yaml").write_text(_scope_placeholder_yaml(program, scope_error), encoding="utf-8")
        results.append(f"scope unavailable ({scope_error}); wrote placeholder (scope_mode: placeholder)")

    try:
        policy = fetch_policy(program, username, token, timeout)
    except PermissionError as e:
        policy, policy_error = None, str(e)
    else:
        policy_error = None if policy is not None else f"Program '{program}' not found on HackerOne."

    if policy is not None:
        (target / "bounty_policy.md").write_text(_policy_to_markdown(policy), encoding="utf-8")
        results.append("policy written")
    else:
        (target / "bounty_policy.md").write_text(_policy_placeholder_markdown(program, policy_error), encoding="utf-8")
        results.append(f"policy unavailable ({policy_error}); wrote placeholder")

    return {"target_dir": str(target), "results": results}


def main(argv=None):
    ap = argparse.ArgumentParser(description="Bug-bounty program scope/policy fetcher (read-only).")
    ap.add_argument("--mode", choices=["scope", "policy", "sync"], default="sync")
    ap.add_argument("--platform", default="hackerone")
    ap.add_argument("--program", required=True)
    ap.add_argument("--target-dir", default=".")
    ap.add_argument("--timeout", type=float, default=30.0)
    args = ap.parse_args(argv)

    if args.platform != "hackerone":
        _result(mode=args.mode, platform=args.platform, program=args.program,
                 configured=False, error=f"Platform '{args.platform}' is not configured in this "
                 f"framework yet -- only 'hackerone' is supported.")
        return

    username = os.environ.get("HACKERONE_USERNAME", "")
    token = os.environ.get("HACKERONE_TOKEN", "")
    if not (username and token):
        _result(mode=args.mode, platform="hackerone", program=args.program, configured=False,
                 error="HackerOne is not configured -- set HACKERONE_USERNAME and HACKERONE_TOKEN.")
        return

    try:
        if args.mode == "scope":
            scope = fetch_scope(args.program, username, token, args.timeout)
            if scope is None:
                _result(mode="scope", platform="hackerone", program=args.program, configured=True,
                         found=False, error=f"Program '{args.program}' not found on HackerOne.")
            else:
                _result(mode="scope", platform="hackerone", program=args.program, configured=True,
                         found=True, scope=scope)
        elif args.mode == "policy":
            policy = fetch_policy(args.program, username, token, args.timeout)
            if policy is None:
                _result(mode="policy", platform="hackerone", program=args.program, configured=True,
                         found=False, error=f"Program '{args.program}' not found on HackerOne.")
            else:
                _result(mode="policy", platform="hackerone", program=args.program, configured=True,
                         found=True, policy=policy)
        else:
            sync_result = do_sync("hackerone", args.program, username, token, args.target_dir, args.timeout)
            _result(mode="sync", platform="hackerone", program=args.program, configured=True, **sync_result)
    except PermissionError as e:
        _result(mode=args.mode, platform="hackerone", program=args.program, configured=True,
                 found=False, error=str(e))
    except Exception as e:
        # Never crash without emitting a parseable result. Exception text is
        # from urllib/stdlib, never includes the credential values themselves.
        _result(mode=args.mode, platform="hackerone", program=args.program, configured=True,
                 found=False, error=f"bounty_platform internal error: {e}")


if __name__ == "__main__":
    main()
