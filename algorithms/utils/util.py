import copy
import numpy as np

import torch
import torch.nn as nn

def init(module, weight_init, bias_init, gain=1):
    """MPS优化的权重初始化函数"""
    weight_init(module.weight.data, gain=gain)
    if module.bias is not None:
        bias_init(module.bias.data)
    return module

def get_clones(module, N):
    """MPS优化的模块复制函数"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def check(input):
    """Type-safe conversion to torch tensor with NaN/Inf sanitization (MPS friendly)."""
    if isinstance(input, np.ndarray):
        if input.dtype == np.float64:
            tensor = torch.from_numpy(input).to(torch.float32)
        elif input.dtype == np.int64:
            tensor = torch.from_numpy(input).to(torch.long)
        else:
            tensor = torch.from_numpy(input)
    elif torch.is_tensor(input):
        tensor = input
    else:
        tensor = torch.tensor(input)
    if torch.is_floating_point(tensor):
        tensor = torch.nan_to_num(tensor, nan=0.0, posinf=1e6, neginf=-1e6).clamp(-1e6, 1e6)
    return tensor

def check_mps_optimized(input, device=None):
    """专门为MPS优化的检查函数"""
    tensor = check(input)
    if device is not None and device.type == 'mps':
        # 确保张量在MPS设备上并且是连续的（MPS性能优化）
        tensor = tensor.to(device).contiguous()
    elif device is not None:
        tensor = tensor.to(device)
    return tensor
