import argparse
from pathlib import Path
from huggingface_hub import HfApi

def upload_model(repo_id: str, adapter_dir: str = None, private: bool = False, token: str = None):
    script_dir = Path(__file__).resolve().parent
    if adapter_dir is None:
        adapter_path = script_dir / "lora-adapters"
    else:
        adapter_path = Path(adapter_dir)

    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter folder not found at: {adapter_path}")

    api = HfApi(token=token)

    print(f"[*] Checking/Creating repository '{repo_id}' on Hugging Face...")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        private=private,
        exist_ok=True
    )

    print(f"[*] Uploading adapter files from '{adapter_path}' to '{repo_id}'...")
    api.upload_folder(
        folder_path=str(adapter_path),
        repo_id=repo_id,
        repo_type="model",
        commit_message="Upload fine-tuned LoRA adapters"
    )

    print(f"[+] Successfully uploaded model to: https://huggingface.co/{repo_id}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Upload fine-tuned LoRA adapters to Hugging Face Hub")
    parser.add_argument("--repo-id", type=str, default="DarkLight6996/CyberSec-Model-V2", help="Hugging Face repo ID (e.g. username/repo-name)")
    parser.add_argument("--adapter-dir", type=str, default=None, help="Local adapter directory path (default: scripts/lora-adapters)")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    parser.add_argument("--token", type=str, default=None, help="Optional HF Token")

    args = parser.parse_args()
    upload_model(
        repo_id=args.repo_id,
        adapter_dir=args.adapter_dir,
        private=args.private,
        token=args.token
    )
