import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("Device count:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device() if torch.cuda.is_available() else "N/A")
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")

import torch
torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

import torch
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo

def print_gpu_memory():
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(handle)
    print(f"GPU Memory - Used: {info.used/1024**2:.2f}MB | Free: {info.free/1024**2:.2f}MB")

# Insert before critical operations
print_gpu_memory()