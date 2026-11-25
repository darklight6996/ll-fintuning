#!/bin/bash
# =====================================================================
#  Ubuntu / Linux Setup Script for ll-finetuning
# =====================================================================
# FIX (2026-11-14, second pass): the first pass of this script (which only
# checked for CLI tools, never installed them) left three real bugs:
#   1. One combined `pip install a b c d` call is ATOMIC -- pip resolves
#      every package before installing ANY of them, so a single unresolvable
#      name (netexec) silently blocked arjun/certipy-ad/impacket too, even
#      though those three install fine on their own. Every tool below now
#      gets its own separate `pip install` call so one failure can't take
#      the others down with it.
#   2. "netexec" is NOT on PyPI under that name -- the project's own install
#      docs (netexec.wiki) recommend `pipx install git+<repo>`, and the
#      resulting command is `nxc`, never literally "netexec". Switched to
#      the documented pipx+git method.
#   3. The Go-bin PATH gap was only ever *printed* as a suggestion, so the
#      check_tools.py run at the end of THIS SAME script invocation still
#      reported subfinder as missing even after installing it successfully
#      one step earlier. Now exports it into the current shell immediately
#      AND persists it to ~/.bashrc (idempotently) for future shells.
# Safe to re-run: every install step checks `command -v` first and skips
# anything already present.
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "====================================================="
echo "  Ubuntu / Linux Setup Script for ll-finetuning"
echo "====================================================="

echo
echo "[*] Installing Python dependencies (requirements-wsl.txt, pinned)..."
pip install -r requirements-wsl.txt

# ---------------------------------------------------------------------
# Headless browser for the browser_probe coverage primitive (Playwright +
# Chromium). Installed here so the primitive renders JS-heavy targets; the
# primitive degrades gracefully (emits a "not installed" note) if this is
# skipped, so this is safe to omit on a headless-less box.
# ---------------------------------------------------------------------
echo
echo "[*] Installing Playwright + Chromium (browser_probe coverage primitive)..."
if python3 -c "import playwright" >/dev/null 2>&1; then
    echo "    [OK] playwright python package already installed."
else
    pip install playwright
fi
# Download the Chromium build Playwright drives (idempotent; --with-deps pulls
# the system libs Chromium needs on a fresh box).
python3 -m playwright install --with-deps chromium 2>&1 | tail -5 || \
    python3 -m playwright install chromium 2>&1 | tail -5 || \
    echo "    [!] Chromium download skipped/failed -- browser_probe will degrade gracefully."

# ---------------------------------------------------------------------
# CLI assessment tools -- apt-installable ones
# ---------------------------------------------------------------------
echo
echo "[*] Installing apt-available assessment tools (nmap, whatweb, sqlmap, ffuf)..."
APT_TOOLS_MISSING=()
for pair in "nmap:nmap" "whatweb:whatweb" "sqlmap:sqlmap" "ffuf:ffuf"; do
    bin="${pair%%:*}"
    pkg="${pair##*:}"
    if ! command -v "$bin" >/dev/null 2>&1; then
        APT_TOOLS_MISSING+=("$pkg")
    fi
done
if [ "${#APT_TOOLS_MISSING[@]}" -gt 0 ]; then
    echo "    Installing via apt: ${APT_TOOLS_MISSING[*]}"
    sudo apt-get update -qq && sudo apt-get install -y "${APT_TOOLS_MISSING[@]}"
else
    echo "    [OK] nmap, whatweb, sqlmap, ffuf already installed."
fi

# ---------------------------------------------------------------------
# CLI assessment tools -- pip-installable ones (each its OWN call -- see
# fix #1 above)
# ---------------------------------------------------------------------
echo
echo "[*] Installing pip-based assessment tools (arjun, certipy-ad, impacket)..."
# arjun      -> HTTP parameter discovery, provides \`arjun\`
# certipy-ad -> AD CS abuse tooling, provides \`certipy\`
# impacket   -> provides secretsdump.py and friends (console scripts, not a
#               single \`impacket\` binary -- check_tools.py checks for the
#               script name directly)
for pkg in arjun certipy-ad impacket; do
    echo "    Installing $pkg..."
    pip install --quiet "$pkg" || echo "    [!] $pkg failed to install -- see output above."
done

# ---------------------------------------------------------------------
# NetExec -- needs its own path: it's not on PyPI as "netexec", and the
# binary it installs is `nxc`, never "netexec" (see check_tools.py's note).
# Official method (netexec.wiki): pipx install from git. Rust/cargo is
# installed first since the official docs flag a Rust-dependent transitive
# build dependency (aardwolf) as the most common install failure.
# ---------------------------------------------------------------------
echo
echo "[*] Installing NetExec (provides the \`nxc\` command, via pipx+git)..."
if command -v nxc >/dev/null 2>&1; then
    echo "    [OK] nxc already installed."
else
    MISSING_PREREQS=()
    command -v pipx >/dev/null 2>&1 || MISSING_PREREQS+=("pipx")
    command -v git  >/dev/null 2>&1 || MISSING_PREREQS+=("git")
    command -v cargo >/dev/null 2>&1 || MISSING_PREREQS+=("cargo" "rustc")
    if [ "${#MISSING_PREREQS[@]}" -gt 0 ]; then
        echo "    Installing prerequisites via apt: ${MISSING_PREREQS[*]}"
        sudo apt-get update -qq && sudo apt-get install -y "${MISSING_PREREQS[@]}"
    fi
    pipx ensurepath >/dev/null 2>&1 || true
    PIPX_BIN="$HOME/.local/bin"
    export PATH="$PATH:$PIPX_BIN"
    if pipx install "git+https://github.com/Pennyw0rth/NetExec" ; then
        echo "    [OK] NetExec installed -- run it as \`nxc\`."
    else
        echo "    [!] NetExec install failed -- see netexec.wiki/getting-started/installation for manual steps."
    fi
fi

# ---------------------------------------------------------------------
# CLI assessment tools -- Go-based ones (subfinder; ProjectDiscovery's httpx
# is intentionally NOT installed here -- see note below)
# ---------------------------------------------------------------------
# NOTE: ProjectDiscovery's `httpx` (a recon CLI tool) is a COMPLETELY
# different thing from PyPI's `httpx` (an async Python HTTP client library,
# already pulled in transitively by requirements-wsl.txt via datasets). Since
# nothing in this codebase actually invokes an `httpx` binary today (confirmed
# by grep), installing the Go one would only create a confusing PATH
# collision with the pip one for zero present benefit -- skipped on purpose.
# If a future agent action needs it, install with:
#   go install github.com/projectdiscovery/httpx/cmd/httpx@latest
# and be aware the pip `httpx` console script may still win on PATH.
echo
echo "[*] Installing Go-based assessment tools (subfinder, nuclei, dalfox)..."
if ! command -v go >/dev/null 2>&1; then
    echo "    Go toolchain not found -- installing golang-go via apt..."
    sudo apt-get update -qq && sudo apt-get install -y golang-go
fi
if command -v go >/dev/null 2>&1; then
    GOBIN="$(go env GOPATH 2>/dev/null)/bin"
    mkdir -p "$GOBIN"
    if ! command -v subfinder >/dev/null 2>&1; then
        echo "    Installing subfinder..."
        go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest 2>&1 | tail -5
    else
        echo "    [OK] subfinder already installed."
    fi
    # nuclei -- templated CVE/misconfig detection engine (bug_bounty nuclei_scan
    # action). v3 module path. On first actual run nuclei downloads its template
    # store to ~/.nuclei-templates (needs network, one time).
    if ! command -v nuclei >/dev/null 2>&1; then
        echo "    Installing nuclei..."
        go install -v github.com/projectdiscovery/nuclei/v3/cmd/nuclei@latest 2>&1 | tail -5
    else
        echo "    [OK] nuclei already installed."
    fi
    # dalfox -- XSS detection/confirmation engine (bug_bounty dalfox_scan action).
    if ! command -v dalfox >/dev/null 2>&1; then
        echo "    Installing dalfox..."
        go install -v github.com/hahwul/dalfox/v2@latest 2>&1 | tail -5
    else
        echo "    [OK] dalfox already installed."
    fi
    # Make them usable for the check_tools.py run at the end of THIS script
    # (fix #3 above), not just in a future shell.
    export PATH="$PATH:$GOBIN"
    BASHRC_PATH_LINE="export PATH=\"\$PATH:$GOBIN\""
    if [ -f "$HOME/.bashrc" ] && ! grep -qF "$GOBIN" "$HOME/.bashrc"; then
        echo "$BASHRC_PATH_LINE" >> "$HOME/.bashrc"
        echo "    Added $GOBIN to PATH in ~/.bashrc for future shells."
    fi
    # Pre-fetch nuclei templates now (idempotent) so the first real scan isn't
    # delayed by a template download mid-run.
    if command -v nuclei >/dev/null 2>&1; then
        echo "    Updating nuclei templates (one-time, may take a moment)..."
        nuclei -update-templates -silent 2>&1 | tail -3 || true
    fi
else
    echo "    [!] Go toolchain unavailable -- skipping subfinder/nuclei/dalfox."
fi

# ---------------------------------------------------------------------
# Cloud CLIs -- az (Azure) and aws (AWS)
# ---------------------------------------------------------------------
# Each uses its own vendor-published install method (not pip/apt). Using the
# exact official commands Microsoft/AWS publish themselves, idempotently.
echo
echo "[*] Installing cloud CLIs (az, aws)..."
if command -v az >/dev/null 2>&1; then
    echo "    [OK] az (Azure CLI) already installed."
else
    echo "    Installing Azure CLI (official Microsoft one-liner for Debian/Ubuntu)..."
    if ! curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash; then
        echo "    [!] Azure CLI install failed -- see https://learn.microsoft.com/cli/azure/install-azure-cli-linux"
    fi
fi

if command -v aws >/dev/null 2>&1; then
    echo "    [OK] aws (AWS CLI) already installed."
else
    echo "    Installing AWS CLI v2 (official AWS installer)..."
    if ! command -v unzip >/dev/null 2>&1; then
        sudo apt-get update -qq && sudo apt-get install -y unzip
    fi
    AWS_TMP_DIR="$(mktemp -d)"
    if curl -s "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "$AWS_TMP_DIR/awscliv2.zip" \
        && unzip -q "$AWS_TMP_DIR/awscliv2.zip" -d "$AWS_TMP_DIR" \
        && sudo "$AWS_TMP_DIR/aws/install"; then
        echo "    [OK] AWS CLI installed."
    else
        echo "    [!] AWS CLI install failed -- see https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
    fi
    rm -rf "$AWS_TMP_DIR"
fi

echo
echo "[*] Running Assessment Tools Pre-Flight Check..."
python3 "$SCRIPT_DIR/../check_tools.py"

echo
echo "Setup complete."
echo "(If any Go/pipx tool just got installed for the first time, open a NEW"
echo " shell to pick up the PATH changes permanently -- this run already"
echo " exported them for its own check above.)"
