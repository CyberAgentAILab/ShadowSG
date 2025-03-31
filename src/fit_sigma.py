import itertools
import shutil
from dataclasses import MISSING, dataclass
from pathlib import Path
from typing import Callable

import cv2
import mitsuba as mi
import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.transforms import quaternion_to_matrix
from scipy.optimize import curve_fit
from torch import Tensor, nn
from tqdm import tqdm

from shadow_sg.config import BaseConfig, Config, GaussianConfig, SceneConfig
from shadow_sg.gaussians import (AniSphericalGaussians, SphericalGaussians,
                                 create_gaussians)
from shadow_sg.io import load_img, load_json, save_img
from shadow_sg.scene import Scene
from shadow_sg.utils import parse_args


@dataclass
class Args:
    path_to_data: Path = MISSING
    path_to_output: Path = MISSING
    method: str = "curve_fit" # curve_fit, mlp_fit
    anistrophic: bool = False
    mu: float = 50.0
    tangent_n_samples: int = 1
    envmap_height: int = 512
    mlp_fit_n_iters: int = 10000
    mlp_fit_lr: float = 1e-3
    mlp_fit_n_hidden_layers: int = 16
    mlp_fit_n_neurons: int = 256
    mlp_fit_activation_fn: str = "softplus" # identity, relu, softplus, sigmoid
    mlp_fit_batch_size: int = 10000
    mlp_fit_scheduler_gamma: float = 0.9
    mlp_fit_scheduler_step_size: int = 2000
    device: str = "cuda"

def render_mitsuba(xml_path: str, device: str = "cuda") -> Tensor:
    mi.set_variant("llvm_ad_rgb")
    mi_scene = mi.load_file(xml_path)
    mi_img = mi.render(mi_scene)
    img_np = np.array(mi_img)
    img = torch.tensor(img_np, dtype=torch.float32, device=device)
    return img

def sigma_fn_k(k_coeffs):

    def sigma_fn(lambd, theta, tangent):
        k1, k2, k3, k4 = k_coeffs
        k = k1 * lambd ** 3 + k2 * lambd ** 2 + k3 * lambd + k4
        if isinstance(theta, Tensor):
            exp_fn = torch.exp
            abs_fn = torch.abs
        elif isinstance(theta, np.ndarray):
            exp_fn = np.exp
            abs_fn = np.abs
        else:
            raise NotImplementedError
        sigma = 1.0 / (1.0 + exp_fn(k * -theta))
        return sigma

    return sigma_fn

def sigma_fn_mlp(mlp: nn.Module):

    def sigma_fn(lambd, theta, tangent=None):
        if tangent is not None:
            mlp_input = torch.cat((lambd, theta, tangent), dim=-1)
        else:
            mlp_input = torch.cat((lambd, theta), dim=-1)
        sigma = mlp(mlp_input)
        return sigma
    
    return sigma_fn

def select_data(sigma_gt: Tensor, sharp: Tensor, ssdf: Tensor, tangent: Tensor, n_samples: int) -> dict:
    inds = torch.arange(ssdf.shape[0], device=sigma_gt.device)
    neg_mask = ssdf < 0
    nan_mask = torch.isinf(sigma_gt) | torch.isnan(sigma_gt)
    sample_inds_pos = torch.randperm(inds[~neg_mask & ~nan_mask].shape[0])[:n_samples]
    sample_inds_pos = inds[~neg_mask & ~nan_mask][sample_inds_pos]
    sample_inds_neg = torch.randperm(inds[neg_mask & ~nan_mask].shape[0])[:n_samples]
    sample_inds_neg = inds[neg_mask & ~nan_mask][sample_inds_neg]
    sample_inds = torch.cat((sample_inds_pos, sample_inds_neg))
    return {
        "sigma_gt": sigma_gt[sample_inds],
        "sharp": sharp[sample_inds],
        "ssdf": ssdf[sample_inds],
        "tangent": tangent[sample_inds]
    }

def sci_curve_fit(sigma_gt: np.ndarray, lambd: np.ndarray, theta: np.ndarray, p0: np.ndarray) -> np.ndarray:
    def fit(x, *p):
        return sigma_fn_k(p)(*x)

    x, cov = curve_fit(fit, (lambd, theta, theta), sigma_gt, p0=p0)
    err = np.sqrt(np.diag(cov))
    return x, err

def mlp_fit(sigma_gt: Tensor, lambd: Tensor, theta: Tensor, tangent: Tensor = None,
    n_hidden_layers: int = 2, n_neurons: int = 64, activation_fn: Callable = nn.ReLU,
    n_iters: int = 10000, lr: float = 1e-3, batch_size: int = 10000,
    scheduler_gamma: float = 0.9, scheduler_every_step: int = 2000,
    weight_decay: float = 1e-2) -> nn.Module:
    n_input_dim = 2
    if tangent is not None:
        n_input_dim = 3
    layers = [
        nn.Linear(n_input_dim, n_neurons),
    ]
    if activation_fn is not None:
        layers.append(activation_fn())
    for _ in range(n_hidden_layers):
        layers.append(nn.Linear(n_neurons, n_neurons))
        if activation_fn is not None:
            layers.append(activation_fn())
    layers.append(nn.Linear(n_neurons, 1))
    layers.append(nn.Sigmoid())
    mlp = nn.Sequential(*layers).to(sigma_gt.device)
    optimizer = torch.optim.Adam(mlp.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_gamma)
    for step in (pbar := tqdm(range(n_iters))):
        optimizer.zero_grad()
        inds = torch.randperm(sigma_gt.shape[0], device=sigma_gt.device)[:batch_size]
        mlp_input = torch.cat((lambd[inds], theta[inds], tangent[inds]) if tangent is not None else (lambd[inds], theta[inds]), dim=-1)
        sigma = mlp(mlp_input)
        loss = F.mse_loss(sigma, sigma_gt[inds])
        loss.backward()
        optimizer.step()

        if step % scheduler_every_step == 0:
            scheduler.step()

        if step % 10 == 0:
            pbar.set_postfix({
                "loss": f"{loss.item():.6f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.6f}"
            })

    return mlp

def get_lambd(sg: SphericalGaussians, normals: Tensor) -> Tensor:
    if isinstance(sg, AniSphericalGaussians):
        scale = sg.get_attr("scale")
        lambd = torch.sqrt(scale[:, 0:1] * scale[:, 1:2])
        lambd = lambd[None, None].broadcast_to(*normals.shape[:-1], 1, 1)
    else:
        lobe_axis = sg.get_attr("lobe_axis")
        sharpness = sg.get_attr("sharpness")
        lambd = torch.norm(sharpness * lobe_axis + 2.133 * normals[..., None, :], dim=-1, keepdim=True)
    return lambd

def get_sigma_gt(sg: SphericalGaussians, normals: Tensor, img_gt: Tensor) -> Tensor:
    assert sg.n_sg == 1
    lambd = get_lambd(sg, normals)
    if isinstance(sg, AniSphericalGaussians):
        z = quaternion_to_matrix(sg.get_attr("rotation"))[..., -1]
        irradiance = torch.pi / lambd
        smooth = torch.maximum(torch.sum(z * normals[..., None, :], dim=-1, keepdim=True), torch.tensor(0.0, device=normals.device))
        integral = smooth * irradiance
        sigma_gt = img_gt / integral[..., 0, :]
    else:
        sharpness = sg.get_attr("sharpness")
        amplitude = sg.get_attr("amplitude")
        lambda_m = sharpness + 2.133  # num_sgs 1
        a_jc = amplitude * 1.170 * torch.exp(lambd - lambda_m)
        sigma_gt = img_gt / (2 * torch.pi * a_jc * (1 - torch.exp(-lambd)) / lambd)[..., 0, :]
    return sigma_gt.clamp(0, 1), lambd[..., 0, :]

def sample_data(comb_data: dict, path_to_output: Path, sg: SphericalGaussians, ray_normals: Tensor, ssdf) -> dict:
    device = ray_normals.device
    sharps = np.linspace(comb_data["lambda_start"], comb_data["lambda_stop"], comb_data["lambda_n_samples"])
    if isinstance(sg, AniSphericalGaussians):
        tangents = np.linspace(comb_data["tangent_start"], comb_data["tangent_stop"], comb_data["tangent_n_samples"])
    else:
        tangents = [0.0]
    combinations = list(itertools.product(sharps, tangents))
    data = {
        "sigma_gt": [],
        "sharp": [],
        "ssdf": [],
        "tangent": [],
    }
    pbar = tqdm(combinations)
    for i, (sharp, tangent) in enumerate(pbar):
        pbar.set_description(f"Preprocessing sharpness {sharp:.2f}")
        if isinstance(sg, AniSphericalGaussians):
            sg.set_sharpness(torch.tensor([[sharp, comb_data["mu"]]], device=device))
            sg.set_tangent(torch.tensor([[tangent]], device=device))
        else:
            sg.set_sharpness(torch.tensor([[sharp]], device=device))
        sharp_act = sg.get_attr("sharpness")[0, 0].item()
        sg.set_amplitude(torch.tensor([[sharp_act / 10.0] * 3], device=device))
        if (path_to_output / "mitsuba" / f"img_{i:05d}.exr").exists():
            img_gt = load_img(path_to_output / "mitsuba" / f"img_{i:05d}.exr", device=device)
        else:
            envmap = sg.to_envmap()
            save_img(envmap, path_to_output, f"envmap", save_png=False)
            img_gt = render_mitsuba(str(path_to_output / "mitsuba.xml"), device=device)
            save_img(img_gt, path_to_output / "mitsuba", f"img_{i:05d}", save_png=False)
            save_img(img_gt, path_to_output / "mitsuba_vis", f"img_{i:05d}", save_exr=False)

        sigma_gt, lambd = get_sigma_gt(sg, ray_normals, img_gt)
        tang = torch.ones_like(lambd) * tangent
        data_pack = select_data(
            sigma_gt=sigma_gt[..., 0].view(-1),
            sharp=lambd[..., 0].view(-1),
            ssdf=ssdf[..., 0].view(-1),
            tangent=tang[..., 0].view(-1),
            n_samples=comb_data["n_samples"],
        )
        for k in data:
            data[k].append(data_pack[k])
    for k in data:
        data[k] = torch.cat(data[k])
    return data

def save_videos(metas: dict):
    videos = {}
    for k, v in metas.items():
        video = cv2.VideoWriter(
            str(v["path_to_output"]),
            cv2.VideoWriter_fourcc(*"mp4v"),
            v["fps"],
            (v["width"], v["height"]),
        )
        videos[k] = video
    
    def record_frame(frame: np.ndarray, name: str):
        if frame.dtype == np.float32:
            frame = ((frame ** (1 / 2.2)).clip(0, 1) * 255).astype(np.uint8)
        videos[name].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    def close():
        for v in videos.values():
            v.release()
    
    return record_frame, close

@torch.no_grad()
def visualization(comb_data: dict, path_to_output: Path, sg: SphericalGaussians,
    ray_normals: Tensor, envmap_height: int, width: int, height: int,
    sigma_fn_type_before, sigma_fn_type_after, ssdf, mu, device) -> None:
    aspect_ratio = width / height
    img_vis_width = int(envmap_height * aspect_ratio)
    save_video_metas = {
        "envmap_img": {
            "path_to_output": path_to_output / "envmap_img.mp4",
            "fps": 10,
            "width": envmap_height*2 + img_vis_width*3,
            "height": envmap_height,
        },
        "envmap_sigma": {
            "path_to_output": path_to_output / "envmap_sigma.mp4",
            "fps": 10,
            "width": envmap_height*2 + img_vis_width*2,
            "height": envmap_height,
        }
    }
    save_video_fn, close_videos_fn = save_videos(save_video_metas)

    sharps = np.linspace(comb_data["lambda_start"], comb_data["lambda_stop"], comb_data["lambda_n_samples"])
    if isinstance(sg, AniSphericalGaussians):
        tangents = np.linspace(comb_data["tangent_start"], comb_data["tangent_stop"], comb_data["tangent_n_samples"])
    else:
        tangents = [0.0]
    combinations = list(itertools.product(sharps, tangents))
    pbar = tqdm(combinations)
    for i, (sharp, tangent) in enumerate(pbar):
        pbar.set_description(f"Rendering image for sharpness {sharp:.2f}")
        img_gt = load_img(path_to_output / "mitsuba" / f"img_{i:05d}.exr", device=device)

        if isinstance(sg, AniSphericalGaussians):
            sg.set_sharpness(torch.tensor([[sharp, mu]], device=device))
            sg.set_tangent(torch.tensor([[tangent]], device=device))
        else:
            sg.set_sharpness(torch.tensor([[sharp]], device=device))
        sharp_act = sg.get_attr("sharpness")[0, 0].item()
        sg.set_amplitude(torch.tensor([[sharp_act / 10.0] * 3], device=device))
        envmap = sg.to_envmap()
        sigma_gt, lambd = get_sigma_gt(sg, ray_normals, img_gt)

        sg.cfg.gaussian.sigma_fn_type = sigma_fn_type_before
        img_before = sg.integrate(ray_normals, ssdf[..., None, :])["irradiance"]
        sg.cfg.gaussian.sigma_fn_type = sigma_fn_type_after
        output_dict = sg.integrate(ray_normals, ssdf[..., None, :])
        img_after = output_dict["irradiance"]
        sigma = output_dict["sigma"][..., 0, :]
        envmap_img_vis = np.concatenate((
            envmap.detach().cpu().numpy(),
            cv2.resize(img_gt.detach().cpu().numpy(), (img_vis_width, envmap_height)),
            cv2.resize(img_before.detach().cpu().numpy(), (img_vis_width, envmap_height)),
            cv2.resize(img_after.detach().cpu().numpy(), (img_vis_width, envmap_height)),
        ), axis=1)
        save_video_fn(envmap_img_vis, "envmap_img")
        envmap_sigma_vis = np.concatenate((
            envmap.detach().cpu().numpy(),
            cv2.resize(sigma_gt.detach().cpu().numpy(), (img_vis_width, envmap_height)),
            cv2.resize(sigma.broadcast_to(*sigma.shape[:-1], 3).detach().cpu().numpy(), (img_vis_width, envmap_height)),
        ), axis=1)
        save_video_fn(envmap_sigma_vis, "envmap_sigma")
    close_videos_fn()

def main():
    args: Args = parse_args(Args)
    (args.path_to_output / "mitsuba").mkdir(parents=True, exist_ok=True)
    (args.path_to_output / "mitsuba_vis").mkdir(parents=True, exist_ok=True)

    cfg = Config(
        base=BaseConfig(
            path_to_output=args.path_to_output,
            device=args.device,
        ),
        scene=SceneConfig(
            path_to_data=args.path_to_data,
        ),
        gaussian=GaussianConfig(
            anistropic=args.anistrophic,
            envmap_height=args.envmap_height,
            sigma_fn_type="k",
            n_gaussians=1,
            path_to_ckpt=args.path_to_data /  ("asg.txt" if args.anistrophic else "sg.txt"),
            path_to_sigma_ckpt="./data/k_coeffs_wang.txt",
            use_hemisphere=False,
        ),
    )

    scene = Scene(cfg)
    ray_pts, _, _, ray_normals = scene.ray_marching()
    sg = create_gaussians(cfg, is_eval=True)
    ssdf = scene.get_ssdf_sphs_level(ray_pts, sg.get_lobe_axis())[..., 0, :] # H W 1
    save_img(ssdf, args.path_to_output, "ssdf", save_png=False)
    save_img(-ssdf, args.path_to_output, "ssdf_neg", save_png=False)

    comb_data = load_json(args.path_to_data / "combinations.json")
    if (args.path_to_output / "sigma_gt.txt").exists() and (args.path_to_output / "sharp.txt").exists() and \
        (args.path_to_output / "ssdf.txt").exists() and (args.path_to_output / "tangent.txt").exists():
        data = {
            "sigma_gt": np.loadtxt(args.path_to_output / "sigma_gt.txt", dtype=np.float32),
            "sharp": np.loadtxt(args.path_to_output / "sharp.txt", dtype=np.float32),
            "ssdf": np.loadtxt(args.path_to_output / "ssdf.txt", dtype=np.float32),
            "tangent": np.loadtxt(args.path_to_output / "tangent.txt", dtype=np.float32),
        }
        data = {k: torch.tensor(v, device=args.device) for k, v in data.items()}
    else:
        shutil.copy(args.path_to_data / "mitsuba.xml", args.path_to_output / "mitsuba.xml")
        data = sample_data(comb_data, args.path_to_output, sg, ray_normals, ssdf)
        for k, v in data.items():
            np.savetxt(args.path_to_output / f"{k}.txt", v.detach().cpu().numpy())

    if args.method == "curve_fit":
        sigma_gt_np = data["sigma_gt"].detach().cpu().numpy()
        sharp_np = data["sharp"].detach().cpu().numpy()
        theta_np = data["ssdf"].detach().cpu().numpy()
        k_coeffs, fit_err = sci_curve_fit(sigma_gt_np, sharp_np, theta_np, p0=[1e-2, 1e-2, 1e-2, 1e-2])
        print("Optimal k_coeffs:", k_coeffs)
        print("Fit error:", fit_err)
        np.savetxt(args.path_to_output / "k_coeffs.txt", k_coeffs)
        sg.k_coeffs = k_coeffs
    elif args.method == "mlp_fit":
        if args.mlp_fit_activation_fn == "relu":
            activation_fn = nn.ReLU
        elif args.mlp_fit_activation_fn == "identity":
            activation_fn = None
        elif args.mlp_fit_activation_fn == "softplus":
            activation_fn = nn.Softplus
        elif args.mlp_fit_activation_fn == "sigmoid":
            activation_fn = nn.Sigmoid
        mlp = mlp_fit(data["sigma_gt"][..., None], data["sharp"][..., None], data["ssdf"][..., None],
            data["tangent"][..., None] if args.anistrophic else None,
            n_hidden_layers=args.mlp_fit_n_hidden_layers, n_neurons=args.mlp_fit_n_neurons,
            lr=args.mlp_fit_lr, n_iters=args.mlp_fit_n_iters, activation_fn=activation_fn,
            batch_size=args.mlp_fit_batch_size)
        torch.save(mlp, args.path_to_output / "sigma_mlp.pt")
        sg.mlp = mlp

    visualization(
        comb_data,
        args.path_to_output,
        sg,
        ray_normals,
        args.envmap_height,
        scene.camera.width,
        scene.camera.height,
        "k",
        "k" if args.method == "curve_fit" else "mlp_ssdf",
        ssdf,
        args.mu,
        args.device
    )

if __name__ == "__main__":
    main()
