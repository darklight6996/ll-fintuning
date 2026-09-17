import shutil
import sys

# Absolutely required tools
REQUIRED_TOOLS = [
    "nmap",
    "netexec", # or crackmapexec
    "subfinder",
    "httpx",
    "ffuf",
]

# Beneficial tools
BENEFICIAL_TOOLS = [
    "whatweb",
    "sqlmap",
    "arjun",
    "curl",
    "az",      # Azure CLI
    "aws",     # AWS CLI
    "certipy", # certipy-ad
    "secretsdump.py",
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
