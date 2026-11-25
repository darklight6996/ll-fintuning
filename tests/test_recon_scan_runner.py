"""
tests/test_recon_scan_runner.py

Unit tests for the breadth-first parallel recon engine
(core/tools/recon_scan_runner.py). No real network calls -- sockets/requests
are monkeypatched so the dispatch/parsing logic is tested deterministically.
"""

import json

import pytest

import core.tools.recon_scan_runner as rsr


# --- static data sanity -----------------------------------------------------

def test_risky_ports_map_has_known_services():
    assert rsr.RISKY_PORTS[22]["service"] == "ssh"
    assert rsr.RISKY_PORTS[445]["level"] == "critical"


def test_default_ports_capped_and_sane():
    assert 0 < len(rsr.DEFAULT_PORTS) <= 50
    assert 443 in rsr.DEFAULT_PORTS and 80 in rsr.DEFAULT_PORTS


# --- host/url parsing --------------------------------------------------------

def test_host_port_scheme_parses_bare_host():
    hostname, scheme, port = rsr._host_port_scheme("example.com")
    assert hostname == "example.com"
    assert scheme == "https"
    assert port is None


def test_host_port_scheme_parses_full_url():
    hostname, scheme, port = rsr._host_port_scheme("http://example.com:8080/path")
    assert hostname == "example.com"
    assert scheme == "http"
    assert port == 8080


# --- ports mode --------------------------------------------------------------

def test_scan_ports_annotates_risk(monkeypatch):
    def fake_check(hostname, port, timeout):
        return port in (22, 9999)
    monkeypatch.setattr(rsr, "_check_port", fake_check)
    result = rsr.scan_ports("example.com", timeout=1, ports=[22, 80, 9999])
    by_port = {p["port"]: p for p in result}
    assert by_port[22]["risk_level"] == "medium"
    assert by_port[22]["service"] == "ssh"
    assert 80 not in by_port
    assert by_port[9999]["risk_level"] == "low"


def test_scan_ports_no_open_ports_returns_empty(monkeypatch):
    monkeypatch.setattr(rsr, "_check_port", lambda h, p, t: False)
    assert rsr.scan_ports("example.com", timeout=1, ports=[22, 80]) == []


# --- dns mode -----------------------------------------------------------------

def test_scan_dns_falls_back_without_dnspython(monkeypatch):
    # Simulate dnspython not being importable by making the import fail inside
    # the function -- easiest reliable way is to monkeypatch socket.gethostbyname
    # and force an ImportError path via sys.modules trickery.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dns.resolver" or name.startswith("dns"):
            raise ImportError("simulated missing dnspython")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(rsr.socket, "gethostbyname", lambda h: "93.184.216.34")
    result = rsr.scan_dns("example.com", timeout=1)
    assert result.get("A") == ["93.184.216.34"]
    assert "note" in result


# --- tech mode ----------------------------------------------------------------

class _FakeResponse:
    def __init__(self, headers, text):
        self.headers = headers
        self.text = text


def test_scan_tech_detects_wordpress_and_server(monkeypatch):
    fake_resp = _FakeResponse({"Server": "nginx/1.18.0"}, "<html>wp-content/themes/x</html>")
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: fake_resp)
    detected = rsr.scan_tech("https://example.com", timeout=1)
    names = {d["name"] for d in detected}
    assert "WordPress" in names
    assert "nginx" in names
    assert any(n.startswith("server:") for n in names)


def test_scan_tech_network_failure_returns_empty(monkeypatch):
    def boom(*a, **k):
        raise ConnectionError("no route")
    monkeypatch.setattr(rsr.requests, "get", boom)
    assert rsr.scan_tech("https://example.com", timeout=1) == []


# --- authsurface mode ---------------------------------------------------------

def test_looks_like_auth_page_detects_password_field():
    assert rsr._looks_like_auth_page('<form><input type="password"></form>')
    assert rsr._looks_like_auth_page("Please Sign In to continue")
    assert not rsr._looks_like_auth_page("<html>just a homepage</html>")


class _FakeResp200(_FakeResponse):
    status_code = 200


def test_scan_authsurface_only_returns_200_auth_pages(monkeypatch):
    def fake_get(url, **kwargs):
        if url.endswith("/login"):
            return _FakeResp200({}, '<input type="password">')
        return _FakeResp200({}, "<html>not found</html>")
    monkeypatch.setattr(rsr.requests, "get", fake_get)

    found = rsr.scan_authsurface("https://example.com", timeout=1)
    urls = [f["url"] for f in found]
    assert any(u.endswith("/login") for u in urls)
    assert all(f["type"] in ("login", "password_reset") for f in found)


# --- subdomains mode (opt-in) --------------------------------------------------

def test_scan_subdomains_bruteforce_only_when_resolvable(monkeypatch):
    def fake_resolve(sub, domain, timeout):
        return f"{sub}.{domain}" if sub == "www" else None
    monkeypatch.setattr(rsr, "_check_subdomain", fake_resolve)
    found = rsr.scan_subdomains("example.com", timeout=1, allow_ct_lookup=False)
    assert found == ["www.example.com"]


def test_scan_subdomains_ct_lookup_is_opt_in(monkeypatch):
    monkeypatch.setattr(rsr, "_check_subdomain", lambda s, d, t: None)
    called = {"ct": False}

    def fake_get(url, **kwargs):
        called["ct"] = True
        class R:
            status_code = 200
            def json(self):
                return [{"name_value": "ct.example.com"}]
        return R()
    monkeypatch.setattr(rsr.requests, "get", fake_get)

    rsr.scan_subdomains("example.com", timeout=1, allow_ct_lookup=False)
    assert called["ct"] is False

    found = rsr.scan_subdomains("example.com", timeout=1, allow_ct_lookup=True)
    assert called["ct"] is True
    assert "ct.example.com" in found


# --- dispatch isolation --------------------------------------------------------

def test_run_recon_isolates_failing_mode(monkeypatch):
    monkeypatch.setattr(rsr, "scan_dns", lambda h, t: {"A": ["1.2.3.4"]})
    monkeypatch.setattr(rsr, "scan_ports", lambda h, t: (_ for _ in ()).throw(RuntimeError("boom")))
    recon = rsr.run_recon("https://example.com", ["dns", "ports"], timeout=1, allow_ct_lookup=False)
    assert recon["dns"] == {"A": ["1.2.3.4"]}
    assert "error" in recon["ports"]


def test_main_never_crashes_without_emitting_json(monkeypatch, capsys):
    def boom(*a, **k):
        raise RuntimeError("total failure")
    monkeypatch.setattr(rsr, "run_recon", boom)
    rsr.main(["--url", "https://example.com", "--modes", "dns"])
    out = capsys.readouterr().out.strip()
    obj = json.loads(out)
    assert "recon" in obj
    assert "error" in obj["recon"]


# --- secrets mode (2026-11-24 ARTEX merge round 2) ---------------------------

def test_mask_secret_shows_only_first_last_four_chars():
    masked = rsr._mask_secret("AKIAABCDEFGHIJKLMNOP")
    assert masked.startswith("AKIA")
    assert masked.endswith("MNOP")
    assert "ABCDEFGHIJKL" not in masked


def test_mask_secret_short_value_fully_masked():
    assert rsr._mask_secret("short") == "*****"


def test_scan_secrets_finds_and_masks_aws_key(monkeypatch):
    class _Resp:
        text = "some config leaked: AKIAABCDEFGHIJKLMNOP in the page"
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())
    found = rsr.scan_secrets("https://example.com", timeout=1)
    names = {f["name"] for f in found}
    assert "AWS Access Key ID" in names
    for f in found:
        assert "AKIAABCDEFGHIJKLMNOP" not in f["masked_value"]


def test_scan_secrets_clean_page_finds_nothing(monkeypatch):
    class _Resp:
        text = "<html>nothing sensitive here</html>"
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())
    assert rsr.scan_secrets("https://example.com", timeout=1) == []


# --- exposed_files mode -------------------------------------------------------

class _FakeFileResp:
    def __init__(self, status_code, text):
        self.status_code = status_code
        self.text = text


def test_scan_exposed_files_requires_both_200_and_signature(monkeypatch):
    def fake_get(url, **kwargs):
        if url.endswith("/.git/HEAD"):
            return _FakeFileResp(200, "ref: refs/heads/main\n")
        if url.endswith("/.env"):
            return _FakeFileResp(200, "<html>custom 200 page, not an env file</html>")
        return _FakeFileResp(404, "not found")
    monkeypatch.setattr(rsr.requests, "get", fake_get)
    found = rsr.scan_exposed_files("https://example.com", timeout=1)
    by_path = {f["path"]: f for f in found if f["status"] == 200}
    assert by_path["/.git/HEAD"]["signature_matched"] is True
    assert by_path["/.env"]["signature_matched"] is False


def test_scan_exposed_files_404_not_included_as_hit(monkeypatch):
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _FakeFileResp(404, ""))
    assert rsr.scan_exposed_files("https://example.com", timeout=1) == []


# --- tech mode: version capture -----------------------------------------------

def test_scan_tech_extracts_version_from_server_header(monkeypatch):
    class _Resp:
        headers = {"Server": "nginx/1.18.0"}
        text = ""
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())
    detected = rsr.scan_tech("https://example.com", timeout=1)
    nginx = next(d for d in detected if d["name"] == "nginx")
    assert nginx.get("version") == "1.18.0"


def test_scan_tech_no_version_omits_version_key(monkeypatch):
    class _Resp:
        headers = {"Server": "nginx"}
        text = ""
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())
    detected = rsr.scan_tech("https://example.com", timeout=1)
    nginx = next(d for d in detected if d["name"] == "nginx")
    assert "version" not in nginx


# --- takeover detection --------------------------------------------------------

def test_check_takeover_requires_content_signature_not_just_cname(monkeypatch):
    class _FakeAnswer:
        target = "something.github.io."

    def fake_resolve(name, rtype):
        if rtype == "CNAME":
            return [_FakeAnswer()]
        raise Exception("no answer")

    fake_dns_resolver = type("M", (), {"resolve": staticmethod(fake_resolve)})()
    fake_dns_module = type("M", (), {"resolver": fake_dns_resolver})()

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "dns", fake_dns_module)
    monkeypatch.setitem(_sys.modules, "dns.resolver", fake_dns_resolver)

    class _Resp:
        text = "There isn't a GitHub Pages site here."
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())

    hit = rsr._check_takeover("sub.example.com", timeout=1)
    assert hit is not None
    assert hit["service"] == "github_pages"


def test_check_takeover_no_signature_match_is_not_a_hit(monkeypatch):
    class _FakeAnswer:
        target = "something.github.io."

    def fake_resolve(name, rtype):
        if rtype == "CNAME":
            return [_FakeAnswer()]
        raise Exception("no answer")

    fake_dns_resolver = type("M", (), {"resolve": staticmethod(fake_resolve)})()
    fake_dns_module = type("M", (), {"resolver": fake_dns_resolver})()

    import sys as _sys
    monkeypatch.setitem(_sys.modules, "dns", fake_dns_module)
    monkeypatch.setitem(_sys.modules, "dns.resolver", fake_dns_resolver)

    class _Resp:
        text = "<html>a perfectly normal, claimed GitHub Pages site</html>"
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())

    assert rsr._check_takeover("sub.example.com", timeout=1) is None


def test_scan_takeovers_empty_subdomains_returns_empty():
    assert rsr.scan_takeovers([], timeout=1) == []


# --- real_ip mode --------------------------------------------------------------

def test_is_private_or_cdn_ip_filters_private_and_cdn():
    assert rsr._is_private_or_cdn_ip("10.0.0.5") is True
    assert rsr._is_private_or_cdn_ip("127.0.0.1") is True
    assert rsr._is_private_or_cdn_ip("173.245.48.1") is True  # cloudflare range
    assert rsr._is_private_or_cdn_ip("8.8.8.8") is False


def test_scan_real_ip_scores_header_leak(monkeypatch):
    class _Resp:
        headers = {"X-Real-IP": "45.33.32.156"}
        text = ""

    def fake_get(url, **kwargs):
        return _Resp()
    monkeypatch.setattr(rsr.requests, "get", fake_get)

    # Force MX lookup to find nothing (dnspython unavailable in this test).
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dns.resolver" or name.startswith("dns"):
            raise ImportError("simulated missing dnspython")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = rsr.scan_real_ip("example.com", timeout=1)
    ips = [c["ip"] for c in result["candidates"]]
    assert "45.33.32.156" in ips


def test_scan_real_ip_no_leak_returns_empty_candidates(monkeypatch):
    class _Resp:
        headers = {}
        text = ""
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _Resp())

    import builtins
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "dns.resolver" or name.startswith("dns"):
            raise ImportError("simulated missing dnspython")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)

    result = rsr.scan_real_ip("example.com", timeout=1)
    assert result["candidates"] == []


# --- js_recon mode (route mining + inertness pruning) ------------------------

def test_mine_routes_extracts_api_and_next_data_routes():
    text = '<a href="/api/users">u</a> <script>f("/v2/orders")</script> ' \
           '<a href="/_next/data/abc123/page.json">n</a> <a href="/about">x</a>'
    found = rsr._mine_routes(text)
    assert "/api/users" in found
    assert "/v2/orders" in found
    assert "/_next/data/abc123/page.json" in found
    assert "/about" not in found


def test_is_testable_route_matches_api_shape_query_form_or_param():
    assert rsr._is_testable_route("https://example.com/api/users", "", "js_html") is True
    assert rsr._is_testable_route("https://example.com/page?id=1", "", "js_html") is True
    assert rsr._is_testable_route("https://example.com/submit", "", "form") is True
    assert rsr._is_testable_route("https://example.com/profile", "id", "js_html") is True
    assert rsr._is_testable_route("https://example.com/about", "", "js_html") is False


class _JsResp:
    def __init__(self, text, status_code=200):
        self.text = text
        self.status_code = status_code


def test_scan_js_routes_mines_and_keeps_live_routes(monkeypatch):
    base_url = "https://example.com"
    html = (
        '<html><script src="/static/app.js"></script>'
        '<form action="/submit"><input name="csrf"></form>'
        '<a href="/api/users">users</a></html>'
    )
    js_text = 'fetch("/api/orders")'

    def fake_get(url, **kwargs):
        if url == base_url:
            return _JsResp(html)
        if url.endswith("/static/app.js"):
            return _JsResp(js_text)
        if url in ("https://example.com/api/users", "https://example.com/api/orders"):
            return _JsResp("", status_code=200)
        return _JsResp("", status_code=404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    result = rsr.scan_js_routes(base_url, timeout=1)

    urls = {r["url"] for r in result["routes"]}
    assert "https://example.com/api/users" in urls
    assert "https://example.com/api/orders" in urls
    assert result["forms"] == [{"action": "https://example.com/submit", "param": "csrf"}]
    assert result["pruned_inert"] == 0


def test_scan_js_routes_prunes_dead_route(monkeypatch):
    base_url = "https://example.com"
    html = '<a href="/api/gone">gone</a>'

    def fake_get(url, **kwargs):
        if url == base_url:
            return _JsResp(html)
        if url == "https://example.com/api/gone":
            return _JsResp("", status_code=404)
        return _JsResp("", status_code=404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    result = rsr.scan_js_routes(base_url, timeout=1)

    assert result["routes"] == []
    assert result["pruned_inert"] == 1


def test_scan_js_routes_network_failure_returns_error_key(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("network down")
    monkeypatch.setattr(rsr.requests, "get", boom)
    result = rsr.scan_js_routes("https://example.com", timeout=1)
    assert "error" in result
    assert result["routes"] == []


# --- auth_sweep mode (unauthenticated probe of declared-secured GET ops) -----

class _SpecResp:
    def __init__(self, status_code, payload=None):
        self.status_code = status_code
        self._payload = payload or {}

    def json(self):
        return self._payload


def test_discover_openapi_spec_finds_first_matching_path(monkeypatch):
    spec = {"paths": {"/admin/users": {}}}

    def fake_get(url, **kwargs):
        if url.endswith("/openapi.json"):
            return _SpecResp(404)
        if url.endswith("/swagger.json"):
            return _SpecResp(200, spec)
        return _SpecResp(404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    found = rsr._discover_openapi_spec("https://example.com", timeout=1)
    assert found == spec


def test_discover_openapi_spec_none_when_nothing_matches(monkeypatch):
    monkeypatch.setattr(rsr.requests, "get", lambda *a, **k: _SpecResp(404))
    assert rsr._discover_openapi_spec("https://example.com", timeout=1) is None


def test_scan_auth_enforcement_flags_unauthenticated_200_on_declared_secured_get(monkeypatch):
    spec = {
        "security": [{"bearerAuth": []}],
        "paths": {
            "/admin/users": {
                "get": {"parameters": []},
            }
        },
    }

    def fake_get(url, **kwargs):
        if url.endswith("/openapi.json"):
            return _SpecResp(200, spec)
        if "/admin/users" in url:
            return _SpecResp(200)
        return _SpecResp(404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    broken = rsr.scan_auth_enforcement("https://example.com", timeout=1)
    assert len(broken) == 1
    assert broken[0]["status"] == 200
    assert broken[0]["method"] == "GET"


def test_scan_auth_enforcement_not_flagged_when_401(monkeypatch):
    spec = {
        "security": [{"bearerAuth": []}],
        "paths": {"/admin/users": {"get": {"parameters": []}}},
    }

    def fake_get(url, **kwargs):
        if url.endswith("/openapi.json"):
            return _SpecResp(200, spec)
        if "/admin/users" in url:
            return _SpecResp(401)
        return _SpecResp(404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    broken = rsr.scan_auth_enforcement("https://example.com", timeout=1)
    assert broken == []


def test_scan_auth_enforcement_never_probes_non_get_methods(monkeypatch):
    spec = {
        "security": [{"bearerAuth": []}],
        "paths": {"/admin/users": {"delete": {"parameters": []}}},
    }
    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if url.endswith("/openapi.json"):
            return _SpecResp(200, spec)
        return _SpecResp(404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    broken = rsr.scan_auth_enforcement("https://example.com", timeout=1)
    assert broken == []
    assert not any("/admin/users" in c for c in calls)


def test_scan_auth_enforcement_unsecured_op_is_never_flagged(monkeypatch):
    spec = {
        "security": [],
        "paths": {"/public/info": {"get": {"parameters": []}}},
    }

    def fake_get(url, **kwargs):
        if url.endswith("/openapi.json"):
            return _SpecResp(200, spec)
        if "/public/info" in url:
            return _SpecResp(200)
        return _SpecResp(404)

    monkeypatch.setattr(rsr.requests, "get", fake_get)
    broken = rsr.scan_auth_enforcement("https://example.com", timeout=1)
    assert broken == []


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
