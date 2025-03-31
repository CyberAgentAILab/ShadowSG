import csv
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pyembree
import pyexr
import torch
import trimesh
from jaxtyping import Float
from torch import Tensor


def load_sgs(sgs_path: Path, device: str = "cuda") -> Float[Tensor, "n_sgs 7"]:
    if sgs_path.suffix == ".txt":
        sgs = np.loadtxt(sgs_path)
    elif sgs_path.suffix == ".npy":
        sgs = np.load(sgs_path)
    else:
        raise RuntimeError(f"Invalid sg file to load: {sgs_path}")
    if sgs.ndim == 1:
        sgs = sgs[None, ...]
    return torch.tensor(sgs.astype(np.float32), device=device)

def save_img(img: Float[Tensor, "H W *"],
             fdir: Path, fname: Tuple[Path, str],
             gamma: float = 2.2,
             save_exr: bool = True,
             save_png: bool = True) -> None:
    assert img.ndim == 3 or img.ndim == 2
    if img.ndim == 2:
        img = img[..., None].broadcast_to(*img.shape, 3)
    if img.ndim == 3:
        assert img.shape[2] == 1 or img.shape[2] == 3
        if img.shape[2] == 1:
            img = img.broadcast_to(*img.shape[:-1], 3)

    img_np = img
    if isinstance(img, torch.Tensor):
        img_np = img.detach().cpu().numpy()

    if save_exr:
        pyexr.write(str(fdir / f"{fname}.exr"), img_np)

    if save_png:
        img_np = cv2.cvtColor(((img_np ** (1 / gamma)).clip(0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(fdir / f"{fname}.png"), img_np)

def load_spheretree(fpaths: List[Path],
                    device: torch.device = "cpu") -> Tuple[int, int, 
                    Float[Tensor, "n_s 3"], Float[Tensor, "n_s 1"]]:
    centers = []
    radii = []
    for fpath in fpaths:
        centers.append([])
        radii.append([])
        with open(fpath) as fp:
            L, B = (int(x) for x in fp.readline().rstrip().split(" "))
            n_spheres = (B ** L - 1) // (B - 1)
            lines = fp.readlines()[:n_spheres]
        for line in lines:
            line = [float(x) for x in line.rstrip().split(" ")]
            radius = line[3]
            if radius <= 1e-4:
                radius = 0
            centers[-1].append(torch.tensor([line[:3]], device=device))
            radii[-1].append(torch.tensor([[radius]], device=device))
        centers[-1] = torch.cat(centers[-1])
        radii[-1] = torch.cat(radii[-1])
    centers = torch.stack(centers)
    radii = torch.stack(radii)
    return {
        "L": L,
        "B": B,
        "centers": centers,
        "radii": radii
    }

def save_spheretree(fpath: Path, level: int, branch: int,
                    centers: Float[Tensor, "n_spheres 3"],
                    radii: Float[Tensor, "n_spheres 1"]) -> None:
    with open(fpath, "w+") as fp:
        fp.write(f"{level} {branch}\n")
        for i in range(centers.shape[0]):
            fp.write(
                f"{centers[i, 0]} {centers[i, 1]} {centers[i, 2]} {radii[i, 0]}\n")

def save_spheres_mesh(fpath: Path, centers: Float[Tensor, "n_spheres 3"], radii: Float[Tensor, "n_spheres 1"], subdivision: int = 4) -> None:
    spheres: trimesh.Trimesh = trimesh.creation.icosphere(subdivision, radii[0, 0].item())
    spheres.apply_translation(centers[0].detach().cpu().numpy())
    for idx, r in enumerate(radii[1:]):
        if r <= 1e-4:
            continue
        sphere = trimesh.creation.icosphere(subdivision, r[0].item())
        sphere.apply_translation(centers[idx+1].detach().cpu().numpy())
        spheres += sphere
    spheres.export(fpath)

def load_img(path: Path, device = "cuda", downscale: int = 1, numpy: bool = False) -> Tensor:
    if path.suffix == ".exr":
        img = pyexr.open(str(path)).get()
    else:
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pix_max = 256
        if img.dtype == np.uint16:
            pix_max = 65535
        img = img.astype(np.float32) / pix_max
    if downscale != 1:
        img = cv2.resize(img, (img.shape[1] // downscale, img.shape[0] // downscale))
    if not numpy:
        img = torch.from_numpy(img).to(device)
    return img

def load_plane(fpath: Path, device = "cuda") -> dict:
    with open(fpath) as fp:
        data = json.load(fp)
    for k, v in data.items():
        data[k] = torch.tensor(v, device=device)
    return data

def load_json(fpath: Path) -> dict:
    with open(fpath) as fp:
        data = json.load(fp)
    return data

def save_csv_dict(data: dict, fpath: Path):
    with open(fpath, "w") as fp:
        writer = csv.DictWriter(fp, data.keys())
        writer.writeheader()
        writer.writerow(data)

def load_csv_dict(fpath: Path) -> dict:
    with open(fpath) as fp:
        reader = csv.DictReader(fp)
        data = {k: v for k, v in reader}
    return data
