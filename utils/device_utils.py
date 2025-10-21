"""
Device configuration utilities for supporting both CUDA and Apple Silicon MPS acceleration.
"""

import torch
import platform


def get_device(args=None):
    """
    Get the appropriate device for training/inference based on system capabilities and user preferences.
    
    Args:
        args: Configuration object with cuda and cuda_deterministic flags (optional)
        
    Returns:
        torch.device: The device to use for computation
        str: Device description for logging
    """
    
    # Check if user disabled GPU acceleration
    if args is not None and not getattr(args, 'cuda', True):
        device = torch.device("cpu")
        return device, f"CPU (用户指定), 线程数: {getattr(args, 'n_training_threads', 1)}"
    
    # For Apple Silicon Macs, prioritize MPS
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        device_description = "Apple Silicon MPS (Metal Performance Shaders)"
        
        # Set deterministic behavior if requested
        if args is not None and getattr(args, 'cuda_deterministic', True):
            # Note: MPS doesn't have the same deterministic controls as CUDA
            # We'll just ensure reproducible random state
            torch.manual_seed(getattr(args, 'seed', 1))
        
        return device, device_description
    
    # For systems with CUDA support
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_description = f"CUDA GPU: {torch.cuda.get_device_name(0)}"
        
        # Set number of threads for CPU operations
        torch.set_num_threads(getattr(args, 'n_training_threads', 1))
        
        # Configure CUDA deterministic behavior
        if getattr(args, 'cuda_deterministic', True):
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        
        return device, device_description
    
    # Fallback to CPU
    else:
        device = torch.device("cpu")
        torch.set_num_threads(getattr(args, 'n_training_threads', 1))
        device_description = f"CPU (无GPU可用), 线程数: {getattr(args, 'n_training_threads', 1)}"
        
        return device, device_description


def setup_seeds(seed_or_args, device=None):
    """
    Set up random seeds for reproducibility across different device types.
    
    Args:
        seed_or_args: Either an integer seed or configuration object with seed
        device: torch.device object (optional, will be auto-detected if not provided)
    """
    # Handle different input types
    if isinstance(seed_or_args, int):
        seed = seed_or_args
    else:
        seed = getattr(seed_or_args, 'seed', 1)
    
    # Get device if not provided
    if device is None:
        device, _ = get_device()
    elif isinstance(device, tuple):
        device = device[0]  # Extract device from tuple if needed
    
    # Set PyTorch seed
    torch.manual_seed(seed)
    
    # Set device-specific seeds
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    elif device.type == 'mps':
        # MPS uses the same random state as CPU PyTorch
        torch.manual_seed(seed)
    
    # Set NumPy seed
    import numpy as np
    np.random.seed(seed)


def get_device_info():
    """
    Get detailed information about available devices.
    
    Returns:
        dict: Device information
    """
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'pytorch_version': torch.__version__,
        'cpu_count': torch.get_num_threads(),
    }
    
    if torch.cuda.is_available():
        info['cuda_available'] = True
        info['cuda_version'] = torch.version.cuda
        info['cuda_device_count'] = torch.cuda.device_count()
        info['cuda_device_name'] = torch.cuda.get_device_name(0)
        info['cuda_memory'] = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        info['cuda_available'] = False
    
    if torch.backends.mps.is_available():
        info['mps_available'] = True
    else:
        info['mps_available'] = False
    
    return info


def print_device_info():
    """Print detailed device information."""
    info = get_device_info()
    
    print("=" * 60)
    print("Device Information:")
    print("=" * 60)
    print(f"Platform: {info['platform']}")
    print(f"Python: {info['python_version']}")
    print(f"PyTorch: {info['pytorch_version']}")
    print(f"CPU threads: {info['cpu_count']}")
    
    if info['cuda_available']:
        print("CUDA available: Yes")
        print(f"CUDA version: {info['cuda_version']}")
        print(f"GPU count: {info['cuda_device_count']}")
        print(f"GPU name: {info['cuda_device_name']}")
        print(f"GPU memory: {info['cuda_memory']:.1f} GB")
    else:
        print("CUDA available: No")
    
    if info['mps_available']:
        print("Apple MPS available: Yes")
    else:
        print("Apple MPS available: No")
    
    print("=" * 60)


def optimize_for_device(model, device):
    """
    Apply device-specific optimizations to a model.
    
    Args:
        model: PyTorch model
        device: torch.device object
        
    Returns:
        model: Optimized model
    """
    # Move model to device
    model = model.to(device)
    
    # Apply device-specific optimizations
    if device.type == 'cuda':
        # For CUDA, we can use mixed precision training if needed
        pass
    elif device.type == 'mps':
        # For MPS, ensure compatible operations
        # Note: Some operations might not be available on MPS
        pass
    
    return model


def tensor_to_device(tensor, device):
    """
    Safely move tensor to device with appropriate dtype.
    
    Args:
        tensor: Input tensor
        device: Target device
        
    Returns:
        tensor: Tensor on target device
    """
    # For MPS, ensure float32 precision as float64 is not well supported
    if device.type == 'mps' and tensor.dtype == torch.float64:
        tensor = tensor.float()
    
    return tensor.to(device)


def create_tpdv(device, dtype=torch.float32):
    """
    Create tensor parameters dictionary for device and dtype.
    
    Args:
        device: torch.device
        dtype: torch.dtype
        
    Returns:
        dict: Dictionary with dtype and device parameters
    """
    return dict(dtype=dtype, device=device)
