import shutil
import sys

# FIX (2026-11-14): two real bugs found while chasing down why a freshly
# installed NetExec never showed up here. (1) "netexec" was never the right
# binary to probe for -- the project's own install docs (both pipx and the
# Kali apt package) ship a command literally named `nxc`; "netexec"/"NetExec"
# are the project/package name, not anything you type. shutil.which("netexec")
# was always going to report MISSING no matter how correctly it was
# installed. (2) REQUIRED_TOOLS claimed 5 tools were "ABSOLUTELY REQUIRED",
# but a full grep of every agent's command_builders.py AND phases/*.py (the
# only two places anything actually shells out to a CLI tool) shows only
# nmap, subfinder, and ffuf are ever actually invoked by code today. httpx
# was especially misleading to list here unqualified -- PyPI's `httpx`
# (an async Python HTTP client that datasets/transformers pull in as a
# transitive dependency) ships its OWN unrelated `httpx` console script, so
# shutil.which("httpx") can report [OK] while pointing at a completely
# different program than ProjectDiscovery's recon CLI of the same name,
# without either one being wrong about being installed.
#
# Reclassified below to match what's actually wired into the codebase today;
# everything else moved to BENEFICIAL since nothing yet depends on it, but
# setup.sh/.ps1 still install what they safely can of it for future use.

# Absolutely required tools -- actually invoked by command_builders.py/phases
# today (confirmed by grep across every agent).
REQUIRED_TOOLS = [
    "nmap",
    "subfinder",
    "ffuf",
]

# Beneficial tools -- not yet wired into any registered action, but worth
# having for planned/future methodology coverage (Cloud, Web Vulns, AD, etc).
BENEFICIAL_TOOLS = [
    "whatweb",
    "sqlmap",
    "arjun",
    "curl",
    "az",      # Azure CLI
    "aws",     # AWS CLI
    "certipy", # certipy-ad
    "secretsdump.py",
    "nxc",     # NetExec's actual binary name (see note above) -- "netexec"
               # and "crackmapexec" are prior/alternate project names, never
               # the command you type.
    "httpx",   # NOTE: this may resolve to PyPI's unrelated async HTTP client
               # library's console script rather than ProjectDiscovery's
               # recon CLI of the same name -- see the note above. [OK] here
               # does not by itself confirm the recon tool is what you'll get.
]

def check_tools():
    print("="*50)
    print("    Assessment Tools Pre-Flight Check")
    print("="*50)
    print("\nNote: These tools are checked but NOT downloaded automatically.")
    print("You must install any missing tools manually via your package manager")
    print("(e.g., apt, apt-get, brew, choco, or python pip).\n")
    
    print("--- ABSOLUTELY REQUIRED TOOLS ---")
    missing_required = []
    for tool in REQUIRED_TOOLS:
        path = shutil.which(tool)
        if path:
            print(f"[OK] {tool} found at: {path}")
        else:
            print(f"[MISSING] {tool} is NOT installed or not in PATH.")
            missing_required.append(tool)
            
    print("\n--- BENEFICIAL TOOLS ---")
    missing_beneficial = []
    for tool in BENEFICIAL_TOOLS:
        path = shutil.which(tool)
        if path:
            print(f"[OK] {tool} found at: {path}")
        else:
            print(f"[MISSING] {tool} is NOT installed or not in PATH.")
            missing_beneficial.append(tool)
            
    print("\n" + "="*50)
    if missing_required:
        print("[!] WARNING: You are missing REQUIRED tools:")
        print("    " + ", ".join(missing_required))
        print("    The assessment CLI may fail to execute critical planned steps.")
    else:
        print("[+] All REQUIRED tools are installed.")
        
    if missing_beneficial:
        print("\n[i] Info: You are missing BENEFICIAL tools:")
        print("    " + ", ".join(missing_beneficial))
        print("    Consider installing these for full methodology coverage (Cloud, Web Vulns, etc).")
    else:
        print("\n[+] All BENEFICIAL tools are installed.")
    print("="*50)

if __name__ == "__main__":
    check_tools()
