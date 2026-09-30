# verify_bnb.py
import torch
import sys

print("[*] Testing PyTorch CUDA...")
if not torch.cuda.is_available():
    print("[!] ERROR: CUDA is not available in PyTorch.")
    sys.exit(1)

print(f"[+] CUDA device: {torch.cuda.get_device_name(0)}")

print("[*] Testing bitsandbytes Linear4bit CUDA layer...")
try:
    from bitsandbytes.nn import Linear4bit
    layer = Linear4bit(64, 64).cuda()
    x = torch.randn(1, 64).cuda()
    y = layer(x)
    print("bitsandbytes OK:", y.shape)
    if y.shape == torch.Size([1, 64]):
        print("[+] SUCCESS: bitsandbytes 4-bit CUDA operations verified!")
        sys.exit(0)
    else:
        print("[!] ERROR: Unexpected output shape from Linear4bit layer.")
        sys.exit(1)
except Exception as e:
    print(f"[!] ERROR testing bitsandbytes: {type(e).__name__}: {e}")
    sys.exit(1)
