"""
LoRA Adapter SHA-256 Hash Verification Script
Compares local adapter weights against the repository version hosted on Hugging Face Hub.
"""

import argparse
import hashlib
from pathlib import Path
from huggingface_hub import HfApi, hf_hub_download, model_info


def compute_sha256(filepath: Path) -> str:
    """Compute SHA-256 checksum of a local file."""
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8 * 1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def get_hf_adapter_sha256(repo_id: str, filename: str = "adapter_model.safetensors", token: str = None) -> str:
    """Fetch the SHA-256 hash of a file on Hugging Face Hub."""
    print(f"[*] Querying Hugging Face repository: {repo_id}")
    try:
        info = model_info(repo_id, token=token)
        for file in info.siblings:
            if file.rfilename == filename and getattr(file, "lfs", None):
                lfs_info = file.lfs
                sha256 = lfs_info.get("sha256") if isinstance(lfs_info, dict) else getattr(lfs_info, "sha256", None)
                if sha256:
                    return sha256
    except Exception as e:
        print(f"[!] Warning fetching metadata: {e}")

    print("[*] Downloading file from Hub to verify exact SHA-256...")
    downloaded_path = hf_hub_download(repo_id=repo_id, filename=filename, token=token)
    return compute_sha256(Path(downloaded_path))


def check_hashes(repo_id: str, search_roots: list = None, filename: str = "adapter_model.safetensors", token: str = None):
    print("=" * 70)
    print(f"  HUGGING FACE MODEL FINGERPRINT VERIFIER: {repo_id}")
    print("=" * 70)

    hf_hash = get_hf_adapter_sha256(repo_id, filename=filename, token=token)
    print(f"[+] Hugging Face Hub SHA-256 ({filename}):")
    print(f"    >> {hf_hash}")
    print()

    if not search_roots:
        script_dir = Path(__file__).resolve().parent
        project_root = script_dir.parent
        search_roots = [
            project_root / "scripts" / "lora-adapters",
            project_root / "scripts" / "lora-adapters-cpu",
            project_root / "scripts" / "lora-output",
            project_root.parent / "lora-adapters",
        ]

    print("-" * 70)
    print("  LOCAL FOLDERS CHECK")
    print("-" * 70)

    found_files = []
    for root in search_roots:
        p = Path(root)
        if p.is_file() and p.name == filename:
            found_files.append(p)
        elif p.is_dir():
            for sub_file in p.rglob(filename):
                found_files.append(sub_file)

    if not found_files:
        print(f"[!] No '{filename}' files found in specified paths.")
        return

    seen = set()
    unique_files = [f for f in found_files if not (f.resolve() in seen or seen.add(f.resolve()))]

    matches = []
    for fpath in unique_files:
        local_hash = compute_sha256(fpath)
        is_match = (local_hash.lower() == hf_hash.lower())
        status = "MATCHES HF! (Live Version)" if is_match else "NO MATCH"

        if is_match:
            matches.append(fpath)

        tag = "MATCH" if is_match else "DIFF "
        print(f"[{tag}] {fpath}")
        print(f"       Hash:   {local_hash}")
        print(f"       Status: {status}")
        print()

    print("=" * 70)
    if matches:
        print(f"[SUCCESS] {len(matches)} matching local file(s) found:")
        for m in matches:
            print(f"  -> {m}")
    else:
        print("[WARNING] No local adapter files match the uploaded Hugging Face version.")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify local LoRA adapter SHA-256 hashes against Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="DarkLight6996/CyberSec-Model-V2", help="Hugging Face repo ID")
    parser.add_argument("--filename", type=str, default="adapter_model.safetensors", help="Target filename to verify")
    parser.add_argument("--paths", nargs="*", default=None, help="Optional specific paths/directories to scan")
    parser.add_argument("--token", type=str, default=None, help="Optional HF user access token")

    args = parser.parse_args()
    check_hashes(
        repo_id=args.repo_id,
        search_roots=args.paths,
        filename=args.filename,
        token=args.token
    )
