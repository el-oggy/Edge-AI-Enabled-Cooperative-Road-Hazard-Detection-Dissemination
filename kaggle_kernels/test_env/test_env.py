
try:
    import ultralytics
    print("ultralytics version:", ultralytics.__version__)
except Exception as e:
    print("ultralytics import failed:", e)

try:
    import torch
    print("torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
except Exception as e:
    print("torch import failed:", e)

import urllib.request
try:
    urllib.request.urlopen("https://www.google.com", timeout=5)
    print("Internet access: YES")
except Exception as e:
    print("Internet access: NO (", e, ")")

