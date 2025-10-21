import numpy as np
import math
import torch

def check(input):
    """将numpy数组转换为torch张量，自动匹配数据类型和设备"""
    if type(input) == np.ndarray:
        # 检查数据类型，选择最合适的torch类型
        if input.dtype == np.float64:
            tensor = torch.from_numpy(input).float()  # 转换为float32以提高MPS性能
        elif input.dtype == np.int64:
            tensor = torch.from_numpy(input).long()
        else:
            tensor = torch.from_numpy(input)
        return tensor
    elif torch.is_tensor(input):
        return input
    else:
        # 处理其他类型（标量等）
        return torch.tensor(input)

def check_device_optimized(input, device=None, dtype=None):
    """设备优化版本的check函数，直接转移到指定设备"""
    tensor = check(input)
    if device is not None:
        tensor = tensor.to(device)
    if dtype is not None:
        tensor = tensor.to(dtype)
    return tensor
        
def get_gard_norm(it):
    sum_grad = 0
    for x in it:
        if x.grad is None:
            continue
        sum_grad += x.grad.norm() ** 2
    return math.sqrt(sum_grad)

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)

def mse_loss(e):
    return e**2/2

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        obs_shape = obs_space.shape
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def get_shape_from_act_space(act_space):
    """Return the flattened action dimension for a (possibly composite) action space.
    Supports Box, Discrete, MultiBinary, MultiDiscrete, and Tuple combinations.
    For Tuple, returns the sum of subspace dimensions (Discrete counts as 1).
    """
    name = act_space.__class__.__name__
    if name == 'Discrete':
        return 1
    if name == 'MultiDiscrete':
        # number of categorical variables
        try:
            return int(len(act_space.nvec))
        except Exception:
            # fallback to shape if provided
            return int(act_space.shape)
    if name == 'Box':
        return int(act_space.shape[0])
    if name == 'MultiBinary':
        return int(act_space.shape[0])
    if name in ("Tuple", "tuple"):
        subspaces = getattr(act_space, "spaces", list(act_space))
        total = 0
        for sub in subspaces:
            subn = sub.__class__.__name__
            if subn == 'Discrete':
                total += 1
            elif subn in ('Box', 'MultiBinary'):
                total += int(sub.shape[0])
            elif subn == 'MultiDiscrete':
                total += int(len(sub.nvec))
            else:
                raise NotImplementedError(f"Unsupported subspace in Tuple: {subn}")
        return total
    # Fallback: try to infer as a list/iterable of subspaces
    try:
        sub0 = act_space[0]
        return int(sub0.shape[0]) + 1
    except Exception:
        raise NotImplementedError(f"Unsupported action space: {name}")


def tile_images(img_nhwc):
    """
    Tile N images into one big PxQ image
    (P,Q) are chosen to be as close as possible, and if N
    is square, then P=Q.
    input: img_nhwc, list or array of images, ndim=4 once turned into array
        n = batch index, h = height, w = width, c = channel
    returns:
        bigim_HWc, ndarray with ndim=3
    """
    img_nhwc = np.asarray(img_nhwc)
    N, h, w, c = img_nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    img_nhwc = np.array(list(img_nhwc) + [img_nhwc[0]*0 for _ in range(N, H*W)])
    img_HWhwc = img_nhwc.reshape(H, W, h, w, c)
    img_HhWwc = img_HWhwc.transpose(0, 2, 1, 3, 4)
    img_Hh_Ww_c = img_HhWwc.reshape(H*h, W*w, c)
    return img_Hh_Ww_c
