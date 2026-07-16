import torch


def resolve_device(device_name: str) -> torch.device:
    """Resolve a config device string to an available torch.device."""
    if device_name == "cuda":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("Warning: CUDA requested but not available, falling back to CPU")
        return torch.device("cpu")
    if device_name == "mps":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("Warning: MPS requested but not available, falling back to CPU")
        return torch.device("cpu")
    return torch.device(device_name)


def should_use_data_parallel(config) -> bool:
    """DataParallel is CUDA-only; skip on MPS/CPU even if config enables it."""
    if not config.hardware.data_parallel:
        return False
    if not torch.cuda.is_available():
        if config.hardware.data_parallel:
            print("Warning: data_parallel is enabled but CUDA is not available; using single device")
        return False
    return True
