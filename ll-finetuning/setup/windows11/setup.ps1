# =====================================================================
#  Windows 11 Setup Script for ll-finetuning
# =====================================================================
# FIX (2026-11-14): this script used to only `pip install -r requirements.txt`
# and CHECK for CLI tools without ever installing them -- same gap the Ubuntu
# setup.sh had, fixed the same day after the pentest-agent smoke test got
# blocked on a missing `nmap`. Now actually installs what it safely can via
# winget/pip/go, and prints a clear manual-install note (with the official
# doc link) for anything that needs a bigger, riskier dependency chain this
# script won't silently pull in on your behalf (NetExec needs Rust + the
# Visual C++ Build Tools; whatweb is Ruby-gem-only, no native Windows path).
# Safe to re-run: every install step checks for the command first and skips
# anything already present.

$ErrorActionPreference = "Continue"
Set-Location -Path $PSScriptRoot

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "  Windows 11 Setup Script for ll-finetuning" -ForegroundColor Cyan
Write-Host "====================================================="

function Test-Cmd($name) {
    return [bool](Get-Command $name -ErrorAction SilentlyContinue)
}

Write-Host "`n[*] Installing Python dependencies..." -ForegroundColor Yellow
pip install -r requirements.txt

# ---------------------------------------------------------------------
# winget-installable tools
# ---------------------------------------------------------------------
Write-Host "`n[*] Installing winget-available assessment tools (nmap, Go)..." -ForegroundColor Yellow
if (-not (Test-Cmd "winget")) {
    Write-Host "    [!] winget not found -- install tools manually: https://aka.ms/getwinget" -ForegroundColor Red
} else {
    if (-not (Test-Cmd "nmap")) {
        Write-Host "    Installing nmap via winget..."
        winget install --id Insecure.Nmap --silent --accept-package-agreements --accept-source-agreements
    } else {
        Write-Host "    [OK] nmap already installed."
    }
    if (-not (Test-Cmd "go")) {
        Write-Host "    Installing Go toolchain via winget (needed for subfinder/ffuf below)..."
        winget install --id GoLang.Go --silent --accept-package-agreements --accept-source-agreements
    } else {
        Write-Host "    [OK] Go toolchain already installed."
    }
}

# ---------------------------------------------------------------------
# pip-installable tools (each its OWN call, same reasoning as setup.sh:
# one bad/unresolvable package name must not block the others)
# ---------------------------------------------------------------------
Write-Host "`n[*] Installing pip-based assessment tools (sqlmap, arjun, certipy-ad, impacket)..." -ForegroundColor Yellow
foreach ($pkg in @("sqlmap", "arjun", "certipy-ad", "impacket")) {
    Write-Host "    Installing $pkg..."
    pip install --quiet $pkg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "    [!] $pkg failed to install -- see output above." -ForegroundColor Red
    }
}

# ---------------------------------------------------------------------
# Go-based tools (subfinder, ffuf) -- re-check for `go` since winget may
# have just installed it in THIS session without updating PATH yet; refresh
# from the registry so the rest of this script can use it immediately.
# ---------------------------------------------------------------------
Write-Host "`n[*] Installing Go-based assessment tools (subfinder, ffuf)..." -ForegroundColor Yellow
$machinePath = [System.Environment]::GetEnvironmentVariable("Path", "Machine")
$userPath = [System.Environment]::GetEnvironmentVariable("Path", "User")
$env:Path = "$machinePath;$userPath"

if (Test-Cmd "go") {
    $goBin = Join-Path $env:USERPROFILE "go\bin"
    New-Item -ItemType Directory -Force -Path $goBin | Out-Null

    if (-not (Test-Cmd "subfinder")) {
        Write-Host "    Installing subfinder..."
        go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
    } else {
        Write-Host "    [OK] subfinder already installed."
    }
    if (-not (Test-Cmd "ffuf")) {
        Write-Host "    Installing ffuf..."
        go install -v github.com/ffuf/ffuf/v2@latest
    } else {
        Write-Host "    [OK] ffuf already installed."
    }

    # Make it usable for the check_tools.py run at the end of THIS script,
    # not just in a future shell, AND persist it for future shells.
    $env:Path = "$env:Path;$goBin"
    if ($userPath -notlike "*$goBin*") {
        [System.Environment]::SetEnvironmentVariable("Path", "$userPath;$goBin", "User")
        Write-Host "    Added $goBin to your User PATH for future shells."
    }
} else {
    Write-Host "    [!] Go toolchain unavailable -- skipping subfinder/ffuf." -ForegroundColor Red
}

# ---------------------------------------------------------------------
# Cloud CLIs -- az (Azure) and aws (AWS), via their winget packages
# ---------------------------------------------------------------------
Write-Host "`n[*] Installing cloud CLIs (az, aws)..." -ForegroundColor Yellow
if (Test-Cmd "winget") {
    if (-not (Test-Cmd "az")) {
        Write-Host "    Installing Azure CLI via winget..."
        winget install --id Microsoft.AzureCLI --silent --accept-package-agreements --accept-source-agreements
    } else {
        Write-Host "    [OK] az (Azure CLI) already installed."
    }
    if (-not (Test-Cmd "aws")) {
        Write-Host "    Installing AWS CLI via winget..."
        winget install --id Amazon.AWSCLI --silent --accept-package-agreements --accept-source-agreements
    } else {
        Write-Host "    [OK] aws (AWS CLI) already installed."
    }
} else {
    Write-Host "    [!] winget not found -- install az/aws manually:" -ForegroundColor Red
    Write-Host "        az:  https://learn.microsoft.com/cli/azure/install-azure-cli-windows"
    Write-Host "        aws: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
}

# ---------------------------------------------------------------------
# Tools that are NOT auto-installed, on purpose
# ---------------------------------------------------------------------
Write-Host "`n[i] Not auto-installed -- see notes:" -ForegroundColor Cyan
Write-Host "    whatweb : Ruby-gem-only, no native Windows package. Use WSL instead,"
Write-Host "              or see https://github.com/urbanadventurer/WhatWeb/wiki/Installation"
Write-Host "    NetExec : needs Rust + the Visual C++ Build Tools before 'pipx install"
Write-Host "              git+https://github.com/Pennyw0rth/NetExec' will build successfully."
Write-Host "              This script won't silently pull in a multi-GB build-tools"
Write-Host "              installer on your behalf -- see netexec.wiki/getting-started"
Write-Host "              /installation/installation-on-windows for the manual steps."

Write-Host "`n[*] Running Assessment Tools Pre-Flight Check..." -ForegroundColor Yellow
python ..\check_tools.py

Write-Host "`nSetup complete." -ForegroundColor Green
Write-Host "(If nmap/Go were just installed for the first time via winget, open a NEW" -ForegroundColor Green
Write-Host " terminal to pick up PATH changes permanently -- this run already refreshed" -ForegroundColor Green
Write-Host " its own PATH for the check above.)" -ForegroundColor Green
