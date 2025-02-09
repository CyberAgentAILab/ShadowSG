import argparse
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union, get_origin

import cv2
import numpy as np
import torch
from jaxtyping import Float
from matplotlib import pyplot as plt
from PIL import Image
from pytorch3d.transforms import quaternion_to_matrix
from scipy.sparse import diags
from torch import Tensor
from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import \
    LearnedPerceptualImagePatchSimilarity as LPIPS


def parse_args(args_cls: type):
    parser = argparse.ArgumentParser()
    for field in fields(args_cls):
        option = f"--{field.name.replace('_', '-')}"
        if field.default is MISSING:
            parser.add_argument(option, type=field.type, required=True)
        elif field.type is bool:
            if field.default is True:
                parser.add_argument(option, action="store_false")
            else:
                parser.add_argument(option, action="store_true")
        else:
            parser.add_argument(option, type=field.type, default=field.default)
    return args_cls(**parser.parse_args().__dict__)

def erode3(img: Tensor) -> Tensor:
    if img.ndim == 2:
        img = img.unsqueeze(-1)
    img = img.permute(2, 0, 1).unsqueeze(0)

    kernel = torch.tensor([
        [0, 1, 0],
        [1, 1, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=img.device)
    kernel = kernel[None, None, ...]

    erosion = torch.nn.functional.conv2d(img.float(), kernel, padding=1)
    return erosion.squeeze(0).permute(1, 2, 0) >= 5

def laplacian_filter3(img: Tensor) -> Tensor:
    img = img.permute(2, 0, 1).unsqueeze(0)

    kernel = torch.tensor([
        [0, -1, 0],
        [-1, 4, -1],
        [0, -1, 0]
    ], dtype=torch.float32, device=img.device)
    kernel = kernel[None, None, ...]
    kernel = kernel.repeat(1, 3, 1, 1)

    laplacian = torch.nn.functional.conv2d(img.float(), kernel, padding=1)
    return laplacian.squeeze(0).permute(1, 2, 0)

def spatial_grad(img: Float[Tensor, "H W 3"]) -> Tuple[Float[Tensor, "H W 1"], Float[Tensor, "H W 1"]]:
    """Computing spatial gradients of an image.

    Parameters:
        img

    Returns:
        dx, dy
    """

    kernel_x = torch.tensor([[[
        [1., 0., -1.],
        [2., 0., -2.],
        [1., 0., -1.],
    ]]], device=img.device)
    kernel_y = kernel_x.transpose(-2, -1)

    gray = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
    gray = gray[None, None, ...]  # (1, 1, H, W)
    dx = torch.nn.functional.conv2d(
        input=gray,  # (1, 1, H, W)
        weight=kernel_x,  # (1, 1, 3, 3)
        padding="same"
    )
    dy = torch.nn.functional.conv2d(gray, kernel_y, padding="same")
    return dx[0].permute(1, 2, 0), dy[0].permute(1, 2, 0)


def luminance(img: Float[Tensor, "H W 3"]) -> Float[Tensor, "H W 1"]:
    triple = torch.tensor([[0.2126, 0.7152, 0.0722]], device=img.device)
    return torch.sum(img * triple, dim=-1, keepdim=True)

def bessel_fn(a: Tensor) -> Tensor:
    return torch.sqrt((0.7846 * a ** 3 + 3.185 * a ** 2 + 8.775 * a + 51.51) / (a ** 4 + 0.2126 * a ** 3 + 0.808 * a ** 2 + 1.523 * a + 1.305))

def rotation_between_vectors(vec1, vec2):
    ''' Retruns rotation matrix between two vectors (for Tensor object)
    https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/utils/geometry.py#L64C1-L64C2
    '''
    # https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d/476311#476311
    # vec1.shape = [N, 3]
    # vec2.shape = [N, 3]
    batch_size = vec1.shape[0]
    
    v = torch.cross(vec1, vec2, dim=-1)                                                     # [N, 3, 3]

    cos = torch.bmm(vec1.view(batch_size, 1, 3), vec2.view(batch_size, 3, 1))
    cos = cos.reshape(batch_size, 1, 1).repeat(1, 3, 3)                             # [N, 3, 3]
    
    skew_sym_mat = torch.zeros(batch_size, 3, 3, device=vec1.device)
    skew_sym_mat[:, 0, 1] = -v[:, 2]
    skew_sym_mat[:, 0, 2] = v[:, 1]
    skew_sym_mat[:, 1, 0] = v[:, 2]
    skew_sym_mat[:, 1, 2] = -v[:, 0]
    skew_sym_mat[:, 2, 0] = -v[:, 1]
    skew_sym_mat[:, 2, 1] = v[:, 0]

    identity_mat = torch.zeros(batch_size, 3, 3, device=vec1.device)
    identity_mat[:, 0, 0] = 1
    identity_mat[:, 1, 1] = 1
    identity_mat[:, 2, 2] = 1

    R = identity_mat + skew_sym_mat
    R = R + torch.bmm(skew_sym_mat, skew_sym_mat) / (1 + cos).clamp(min=1e-7)
    zero_cos_loc = (cos == -1).float()
    R_inverse = torch.zeros(batch_size, 3, 3, device=vec1.device)
    R_inverse[:, 0, 0] = -1
    R_inverse[:, 1, 1] = -1
    R_inverse[:, 2, 2] = -1
    R_out = R * (1 - zero_cos_loc) + R_inverse * zero_cos_loc                       # [N, 3, 3]

    return R_out

def build_covariance(rot: Float[Tensor, "* 4"], scale: Float[Tensor, "* 2"]) -> Float[Tensor, "* 3 3"]:
    if rot.shape[-1] == 4:
        R = quaternion_to_matrix(rot) # n_asg 3 3
    else:
        R = rot

    S = torch.zeros((*scale.shape[:-1], 3, 3), device=scale.device) # n_asg 3 3
    S[..., 0, 0] = scale[..., 0]
    S[..., 1, 1] = scale[..., 1]
    S[..., 2, 2] = 1.0
    cov = R @ S @ R.transpose(-2, -1) # n_asg 3 3
    return cov

    # x = R[..., :, 0] # * 3
    # y = R[..., :, 1] # * 3
    # lnc = torch.log(color)[..., None, None] * torch.eye(3, device=rot.device) # * C 3 3
    # cov = scale[..., 0:1, None] * (x[..., None] @ x[..., None, :]) + \
    #     scale[..., 1:2, None] * (y[..., None] @ y[..., None, :]) # * 3 3
    # cov = cov[..., None, :, :] - lnc

    # return cov

def fft(input: Float[Tensor, "H W 3"]) -> Float[Tensor, "H W 1"]:
    assert input.ndim == 3 and (input.shape[-1] == 3 or input.shape[-1] == 1)
    fft_input = input
    if input.shape[-1] == 3:
        fft_input = luminance(input)
    frequency = torch.fft.fftn(fft_input)
    frequency = torch.fft.fftshift(frequency)
    frequency = torch.sqrt(frequency.real**2 + frequency.imag**2)
    # frequency = torch.log10(frequency + 1e-8)
    return frequency

def scale_invarient_rmse(x: Float[Tensor, "*"], x_hat: Float[Tensor, "*"]) -> float:
    scale = torch.sum(x * x_hat) / torch.sum(x_hat ** 2)
    return torch.sqrt(torch.mean((x - scale * x_hat) ** 2)).item()

def angular_error(x: Float[Tensor, "* 3"], x_hat: Float[Tensor, "* 3"]) -> float:
    dot = torch.sum(x * x_hat, dim=-1)
    dot = dot / (torch.norm(x, dim=-1)*torch.norm(x_hat, dim=-1) + 1e-8)
    dot = torch.clamp(dot, -1 + 1e-8, 1 - 1e-8)
    angle = torch.rad2deg(torch.mean(torch.acos(dot)))
    return angle.item()

def laplacian_2d(N, M):
    main_diag = np.ones(N*M) * -4
    off_diag = np.ones(N*M-1)
    off_diag[np.arange(1, N*M) % M == 0] = 0

    diagonals = [main_diag, off_diag, off_diag, off_diag, off_diag]
    laplacian = diags(diagonals, [0, -1, 1, -M, M], shape=(N*M, N*M))

    return torch.from_numpy(laplacian.toarray())

def get_timestamp_path(path: Union[Path, str]):
    if isinstance(path, Path):
        path = str(path)
    pattern = Path(path.replace("{timestamp}", "*"))
    return sorted(list(pattern.parent.glob(pattern.stem)))[-1]

def fibonacci_sphere_sampling(sample_num, device="cuda"):
    '''https://github.com/apple/ml-neilf/blob/b1fd650f283b2207981596e0caba7419238599c9/code/model/ray_sampling.py#L17
    '''
    delta = np.pi * (3.0 - np.sqrt(5.0))

    # fibonacci sphere sample around z axis 
    idx = torch.arange(sample_num, device=device).float().unsqueeze(-1) # [S, 1]
    if sample_num == 1:
        z = torch.tensor([[1.0]], device=device)
    else:
        z = 1 - 2 * idx / (sample_num - 1) # [S, 1]
    rad = torch.sqrt(1 - z ** 2) # [S, 1]
    theta = delta * idx # [S, 1]
    y = torch.cos(theta) * rad # [S, 1]
    x = torch.sin(theta) * rad # [S, 1]
    z_samples = torch.cat([x, y, z], axis=-1) # [S, 3]

    return z_samples

def compute_metrics(img: Float[Tensor, "img_h img_w 3"],
    img_gt: Float[Tensor, "img_h img_w 3"],
    env: Float[Tensor, "env_h env_w 3"],
    env_gt: Optional[Float[Tensor, "env_h env_w 3"]] = None,
    plane_mask: Tensor = None) -> dict:
    device = img.device

    # Metrics
    psnr_fn = PSNR().to(device)
    ssim_fn = SSIM().to(device)
    lpips_fn = LPIPS(net_type="vgg", normalize=True)
    metrics_dict = {
        "img": {},
        "env": {},
    }

    # Image
    img = img.clamp_min(0.0)
    if plane_mask is not None:
        plane_mask = plane_mask.unsqueeze(-1)
        img = img * plane_mask
        img_gt = img_gt * plane_mask
    metrics_dict["img"]["psnr"] = psnr_fn(
        img[None, ...].permute(0, 3, 1, 2),
        img_gt[None, ...].permute(0, 3, 1, 2)
    ).item()
    metrics_dict["img"]["ssim"] = ssim_fn(
        img[None, ...].permute(0, 3, 1, 2),
        img_gt[None, ...].permute(0, 3, 1, 2)
    ).item()
    try:
        metrics_dict["img"]["lpips"] = lpips_fn(
            img[None, ...].permute(0, 3, 1, 2).detach().cpu().clamp(0.0, 1.0),
            img_gt[None, ...].permute(0, 3, 1, 2).detach().cpu().clamp(0.0, 1.0)
        ).item()
    except Exception as e:
        print("Compute image LPIPS failed:", e)
        metrics_dict["img"]["lpips"] = 0
    metrics_dict["img"]["scaled_rmse"] = scale_invarient_rmse(img_gt, img)
    metrics_dict["img"]["ang_error"] = angular_error(img_gt, img)

    # Envmap
    if env_gt is not None:
        env_h = env.shape[0]
        env_gt_h = env_gt.shape[0]
        if env_gt.shape[0] != env_h:
            # env_gt = torch.nn.functional.interpolate(env_gt.permute(2, 0, 1)[None, ...], size=(env_h, 2*env_h))[0].permute(1, 2, 0)
            env = torch.nn.functional.interpolate(env.permute(2, 0, 1)[None, ...], size=(env_gt_h, 2*env_gt_h))[0].permute(1, 2, 0)
        env_u = env[:env_h//2].clamp_min(0.0)
        env_gt_u = env_gt[:env_h//2, ..., :3]
        metrics_dict["env"]["psnr"] = psnr_fn(
            env_u[None, ...].permute(0, 3, 1, 2),
            env_gt_u[None, ...].permute(0, 3, 1, 2)
        ).item()
        try:
            metrics_dict["env"]["ssim"] = ssim_fn(
                env_u[None, ...].permute(0, 3, 1, 2),
                env_gt_u[None, ...].permute(0, 3, 1, 2)
            ).item()
        except Exception as e:
            print("Compute envmap SSIM failed:", e)
            metrics_dict["env"]["ssim"] = 0
        try:
            metrics_dict["env"]["lpips"] = lpips_fn(
                env_u[None, ...].permute(0, 3, 1, 2).detach().cpu().clamp(0.0, 1.0),
                env_gt_u[None, ...].permute(0, 3, 1, 2).detach().cpu().clamp(0.0, 1.0)
            ).item()
        except Exception as e:
            print("Compute envmap LPIPS failed:", e)
            metrics_dict["env"]["lpips"] = 0
        metrics_dict["env"]["scaled_rmse"] = scale_invarient_rmse(env_gt_u, env_u)
        metrics_dict["env"]["ang_error"] = angular_error(env_gt_u, env_u)
    else:
        metrics_dict["env"] = {"psnr": 0, "ssim": 0, "lpips": 0, "scaled_rmse": 0, "ang_error": 0}

    return metrics_dict

def transform_se3(T: Float[Tensor, "* 4 4"], p: Float[Tensor, "* 3"]) -> Float[Tensor, "* 3"]:
    R = T[:3, :3]
    t = T[:3, 3]
    p_ = (R @ p[..., None])[..., 0] + t
    return p_

def normalize_image(img: Tensor) -> Tensor:
    return (img - img.min()) / (img.max() - img.min())

def parse_args(args_cls: type) -> Any:
    parser = argparse.ArgumentParser()
    for field in fields(args_cls):
        option = f"--{field.name.replace('_', '-')}"
        if field.default is MISSING:
            parser.add_argument(option, type=field.type, required=True)
        elif field.type is bool:
            if field.default is True:
                parser.add_argument(option, action="store_false")
            else:
                parser.add_argument(option, action="store_true")
        elif get_origin(field.type) is Literal:
            parser.add_argument(option, choices=field.type.__args__, default=field.default)
        else:
            parser.add_argument(option, type=field.type, default=field.default)
    return args_cls(**parser.parse_args().__dict__)

def compute_error_map(img_err, mask = None, max_val = None):
    if mask is not None:
        img_err[~mask] = 0.0
    if max_val is None:
        max_val = img_err.max()
    img_err[img_err >= max_val] = max_val
    img_err = img_err / max_val
    # img_err_rescaled = -(img_err - 1) ** 4 + 1
    img_err_rescaled = img_err

    if isinstance(img_err_rescaled, Tensor):
        img_err_rescaled_np = (img_err_rescaled * 255).byte().detach().cpu().numpy()
    else:
        img_err_rescaled_np = (img_err_rescaled * 255).astype(np.uint8)
    img_err_rescaled_np = cv2.applyColorMap(img_err_rescaled_np, cv2.COLORMAP_JET)
    img_err_rescaled_np = cv2.cvtColor(img_err_rescaled_np, cv2.COLOR_BGR2RGB)
    if mask is not None:
        if isinstance(mask, Tensor):
            mask = mask.detach().cpu().numpy()
        img_err_rescaled_np[~mask] = 255
    img_err_rescaled_np = img_err_rescaled_np.astype(np.float32) / 255
    return img_err_rescaled_np

def draw_color_bar(min_val, max_val) -> np.ndarray:
    a = np.array([[min_val, max_val]])
    fig = plt.figure(figsize=(1, 5))
    plt.imshow(a, cmap="jet")
    plt.gca().set_visible(False)
    ax = plt.axes([0.1, 0.05, 0.4, 0.9])
    plt.colorbar(cax=ax)
    canvas = fig.canvas
    canvas.draw()
    img_cb = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img_cb = img_cb.reshape(*reversed(canvas.get_width_height()), 3)

    return img_cb
