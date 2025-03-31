import os
from dataclasses import dataclass
from pathlib import Path

import torch
import trimesh

from shadow_sg.io import save_spheretree
from shadow_sg.utils import parse_args


@dataclass
class Args:
    path_to_obj: Path = ""
    save_objs: bool = False

def load_spheretree(path_to_sph) -> tuple:
    centers, radii = [], []
    with open(path_to_sph) as fp:
        L, B = (int(x) for x in fp.readline().rstrip().split(" "))
        n_spheres = (B ** L - 1) // (B - 1)
        lines = fp.readlines()[:n_spheres]
    for line in lines:
        line = [float(x) for x in line.rstrip().split(" ")]
        radius = 0.0 if line[3] <= 1e-6 else line[3]
        centers.append(torch.tensor([line[:3]]))
        radii.append(torch.tensor([[radius]]))
    centers = torch.cat(centers)
    radii = torch.cat(radii)
    return L, B, centers, radii

def get_spheres(B, centers, radii, level) -> tuple:
    n_spheres = B ** level
    start_idx = (1 - B ** level) // (1 - B)
    end_idx = start_idx + n_spheres
    return centers[start_idx:end_idx], radii[start_idx:end_idx]

def spheres_to_mesh(centers, radii) -> trimesh.Trimesh:
    spheres: trimesh.Trimesh = trimesh.creation.icosphere(4, radii[0, 0].item())
    spheres.apply_translation(centers[0].detach().cpu().numpy())
    for idx, r in enumerate(radii[1:]):
        if r <= 1e-4:
            continue
        sphere = trimesh.creation.icosphere(4, r[0].item())
        sphere.apply_translation(centers[idx+1].detach().cpu().numpy())
        spheres += sphere
    return spheres

def main():
    args: Args = parse_args(Args)

    os.system(f"./src/makeTreeMedial -branch 8 -depth 3 -testerLevels 2 -numCover 10000 -minCover 5 -initSpheres 1000 -minSpheres 200 -erFact 2 -nopause -expand -merge -verify -eval {args.path_to_obj}")

    sph_path = args.path_to_obj.parent / f"{args.path_to_obj.stem}-medial.sph"
    L, B, centers, radii = load_spheretree(sph_path)

    save_spheretree(args.path_to_obj.parent / f"{args.path_to_obj.stem}_sphere.sph", L, B, centers, radii)
    if args.save_objs:
        start_idx = (1 - B ** (2)) // (1 - B)
        end_idx = start_idx + B ** 2
        sphere_mesh = spheres_to_mesh(centers[start_idx:end_idx], radii[start_idx:end_idx])
        sphere_mesh.export(args.path_to_obj.parent / f"{args.path_to_obj.stem}_sphere.obj")

if __name__ == "__main__":
    main()
