import sys
import subprocess
import torch
import urllib.request

print("=== Checking Kaggle Verification & Environment ===")
print("Python:", sys.version)
print("Torch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU Device:", torch.cuda.get_device_name(0))

# Test Internet
try:
    with urllib.request.urlopen("https://www.google.com", timeout=10) as resp:
        print("Internet Status: CONNECTED (HTTP", resp.status, ")")
except Exception as e:
    print("Internet Status: FAILED (", e, ")")

# Test Pip Install
try:
    print("Testing pip install ultralytics...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "ultralytics"], check=True)
    import ultralytics
    print("Ultralytics installed successfully, version:", ultralytics.__version__)
except Exception as e:
    print("Pip install failed:", e)

print("=== Test Complete ===")
