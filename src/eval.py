from dataclasses import MISSING, dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

from shadow_sg.config import Config
from shadow_sg.gaussians import create_gaussians
from shadow_sg.io import load_img, save_img
from shadow_sg.scene import Scene
from shadow_sg.trainer import GaussiansTrainer, LSTSQTrainer
from shadow_sg.utils import parse_args


@dataclass
class Args:
    path_to_exp_dir: str = ""
    iteration: int = -1
    ssdf_method: str = "sph_level_min"
    load_exist_img: bool = False

    path_to_output: str = None
    path_to_ckpt: str = None

    render_individual_gaussians: bool = False
    render_all_objects: bool = False
    render_only_virtual_objs: bool = False

    composite_with_gt: bool = False
    composite_shadow_mask_weight: float = 0.82
    composite_shadow_weight: float = 0.4
    composite_shadow_comp_weight: float = 0.2

def composite(img, scene: Scene, obj_indices: list, shadow_mask_weight: float = 0.75, shadow_weight: float = 0.9, shadow_comp_weight: float = 0.1):
    img_gt = scene.camera.get_img(downscale=scene.cfg.scene.downscale_eval)

    _, ray_hits, ray_on_object, _ = scene.ray_marching(downscale=scene.cfg.scene.downscale_eval, obj_indices=obj_indices)
    # img = img[img.shape[0]//4:]
    # img_gt = img_gt[img_gt.shape[0]//4:]
    # ray_hits = ray_hits[ray_hits.shape[0]//4:]
    # ray_on_object = ray_on_object[ray_on_object.shape[0]//4:]
    plane_mask = ray_hits & (ray_on_object < 0)
    shadow_mask = torch.zeros_like(plane_mask)
    shadow_mask[plane_mask] = img[plane_mask].mean(dim=-1) < shadow_mask_weight * img[plane_mask].mean()

    result_img = img_gt.clone()
    result_img[ray_on_object >= 0] = img[ray_on_object >= 0]
    result_img[shadow_mask] = shadow_comp_weight * result_img[shadow_mask] + shadow_weight * img[shadow_mask]

    return result_img, shadow_mask

def main():
    args: Args = parse_args(Args)

    cfg = OmegaConf.load(Path(args.path_to_exp_dir) / "config.yaml")
    if "sharpness_activation_fn" not in cfg.gaussian:
        cfg.gaussian.sharpness_activation_fn = "exp"
    if "amplitude_activation_fn" not in cfg.gaussian:
        cfg.gaussian.amplitude_activation_fn = "abs"
    OmegaConf.save(cfg, Path(args.path_to_exp_dir) / "config.yaml")

    cfg = Config(**cfg)

    if args.path_to_output is not None:
        cfg.base.path_to_output = args.path_to_output

    if args.render_all_objects:
        cfg.path_to_img = Path(cfg.scene.path_to_data) / "image_virtual_objs.exr"
        if not cfg.path_to_img.exists():
            cfg.path_to_img = None

    scene = Scene(cfg)

    obj_indices = []
    if args.render_all_objects:
        obj_indices = list(range(len(scene.meshes)))

    if args.render_only_virtual_objs:
        obj_indices = [i for i in range(len(scene.meshes)) if not scene.obj_is_train[i]]

    if cfg.base.method == "gaussians":
        if args.iteration == -1:
            args.iteration = cfg.trainer.n_iters
        if args.ssdf_method is not None:
            cfg.scene.ssdf_method = args.ssdf_method
        if args.path_to_ckpt is None:
            cfg.gaussian.path_to_ckpt = str(Path(args.path_to_exp_dir) / "eval" / f"iter_{args.iteration:06d}" / "gaussians_opti.txt")
        else:
            cfg.gaussian.path_to_ckpt = args.path_to_ckpt
        if cfg.gaussian.optimizable_k:
            cfg.gaussian.path_to_sigma_ckpt = str(Path(args.path_to_exp_dir) / "eval" / f"iter_{args.iteration:06d}" / "k_coeffs.txt")
            cfg.gaussian.optimizable_k = False
        gaussians = create_gaussians(cfg, scene.plane_normal)
        trainer = GaussiansTrainer(cfg, gaussians)
        trainer.eval(args.iteration, scene, load_exist_img=args.load_exist_img, obj_indices=obj_indices)

        if args.composite_with_gt:
             img = load_img(Path(cfg.base.path_to_output) / "eval" / f"iter_{args.iteration:06d}" / "image.exr")
             img_composite, shadow_mask = composite(img, scene, obj_indices, args.composite_shadow_mask_weight, args.composite_shadow_weight, args.composite_shadow_comp_weight)
             save_img(shadow_mask, Path(cfg.base.path_to_output) / "eval" / f"iter_{args.iteration:06d}", "shadow_mask")
             save_img(img_composite, Path(cfg.base.path_to_output) / "eval" / f"iter_{args.iteration:06d}", "composite")
             save_img(gaussians.env_gt, Path(cfg.base.path_to_output) / "eval" / f"iter_{args.iteration:06d}", "envmap_gt")
             save_img(scene.camera.get_img(downscale=scene.cfg.scene.downscale_eval), Path(cfg.base.path_to_output) / "eval" / f"iter_{args.iteration:06d}", "image_gt")

        if args.render_individual_gaussians:
            path_to_individual_renders = Path(cfg.base.path_to_output) / "eval" / f"iter_{args.iteration:06d}" / "individual_renders"
            path_to_individual_renders.mkdir(parents=True, exist_ok=True)
            individual_images = []
            buffer = []
            # for i, individual_image in enumerate(gaussians.render_images_individual_gaussians(scene)):
            n_sg = len(list(path_to_individual_renders.glob("[0-9][0-9][0-9].png")))
            print("n_sg:", n_sg)
            for i in range(n_sg):
                if i > 0 and i % 8 == 0:
                    individual_images.append(np.concatenate(buffer, axis=1))
                    buffer = []
                # save_img(individual_image, path_to_individual_renders, f"{i:03d}", save_exr=False)
                # individual_image_np = ((individual_image.detach().cpu().numpy() ** (1 / 2.2)).clip(0, 1) * 255).astype(np.uint8)
                individual_image_np = np.array(Image.open(path_to_individual_renders / f"{i:03d}.png"))
                individual_image_scaled = individual_image_np.astype(np.float32)# + individual_image_np.astype(np.float32)
                individual_image_scaled = individual_image_scaled.clip(0, 255).astype(np.uint8)
                buffer.append(individual_image_scaled)
                # Image.fromarray(individual_image_scaled).save(path_to_individual_renders / f"{i:03d}_scaled.png")
            if len(buffer) != 8:
                for _ in range(8 - len(buffer)):
                    buffer.append(Image.fromarray(np.ones_like(buffer[0]) * 255))
            individual_images.append(np.concatenate(buffer, axis=1))
            if len(individual_images) > 1:
                individual_images = np.concatenate(individual_images, axis=0)
            else:
                individual_images = individual_images[0]
            Image.fromarray(individual_images).save(path_to_individual_renders / "all.png")

    elif cfg.base.method == "lstsq":
        envmap = load_img(Path(args.path_to_exp_dir) / "envmap" / "envmap.exr")
        envmap_nnls = None
        if (Path(args.path_to_exp_dir) / "envmap_nnls" / "envmap.exr").exists():
            envmap_nnls = load_img(Path(args.path_to_exp_dir) / "envmap_nnls" / "envmap.exr")
        trainer = LSTSQTrainer(cfg)

        if args.composite_with_gt:
            paths = [Path(cfg.base.path_to_output) / "envmap" / "image.exr", Path(cfg.base.path_to_output) / "envmap_nnls" / "image.exr"]
            for path in paths:
                if not path.exists():
                    continue
                img = load_img(path)
                img_composite, shadow_mask = composite(img, scene, obj_indices, args.composite_shadow_mask_weight, args.composite_shadow_weight, args.composite_shadow_comp_weight)
                save_img(shadow_mask, path.parent, "shadow_mask")
                save_img(img_composite, path.parent, "composite")
                save_img(scene.camera.get_img(downscale=scene.cfg.scene.downscale_eval), path.parent, "image_gt")
    else:
        raise ValueError(f"Unknown method: {cfg.method}")

if __name__ == "__main__":
    main()
