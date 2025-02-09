import json
import timeit
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torchvision.transforms
from matplotlib import pyplot as plt
from scipy.optimize import nnls
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from shadow_sg.config import Config
from shadow_sg.gaussians import AniSphericalGaussians, SphericalGaussians
from shadow_sg.io import load_img, save_img
from shadow_sg.scene import Scene
from shadow_sg.utils import (compute_error_map, compute_metrics, erode3,
                             laplacian_2d, laplacian_filter3, normalize_image)


class GaussiansTrainer:
    def __init__(self,
        cfg: Config,
        gaussians: SphericalGaussians | AniSphericalGaussians,
    ) -> None:
        self.cfg = cfg
        self.device = self.cfg.base.device
        self.gaussians = gaussians
        self.path_to_output = Path(self.cfg.base.path_to_output)

    def setup_optimizers(self) -> None:
        attrs = self.gaussians.get_optimizable_dict()
        if self.cfg.gaussian.anistropic:
            self.optimizer = torch.optim.Adam([
                {"params": attrs["rotation"], "name": "rotation", "lr": self.cfg.trainer.lr_asg_rotation},
                {"params": attrs["scale"], "name": "scale", "lr": self.cfg.trainer.lr_asg_scale},
                {"params": attrs["amplitude"], "name": "amplitude", "lr": self.cfg.trainer.lr_asg_amplitude},
            ])
        else:
            self.optimizer = torch.optim.Adam([
                {"params": attrs["lobe_axis"], "name": "lobe_axis", "lr": self.cfg.trainer.lr_sg_lobe_axis},
                {"params": attrs["sharpness"], "name": "sharpness", "lr": self.cfg.trainer.lr_sg_sharpness},
                {"params": attrs["amplitude"], "name": "amplitude", "lr": self.cfg.trainer.lr_sg_amplitude},
            ])
        if self.cfg.gaussian.sigma_fn_type == "mlp":
            self.optimizer.add_param_group({
                "params": self.gaussians.mlp.parameters(), "name": "sigma_mlp", "lr": self.cfg.trainer.lr_sigma_mlp
        })

        if self.cfg.gaussian.sigma_fn_type == "k" and self.cfg.gaussian.optimizable_k:
            self.optimizer.add_param_group({
                "params": self.gaussians.k_coeffs, "name": "k_coeffs", "lr": self.cfg.trainer.lr_k_coeffs
            })

        if self.cfg.trainer.lr_scheduler_step > 0:
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=self.cfg.trainer.lr_scheduler_gamma,
            )

    def train_report(self,
        step: int,
        writer: SummaryWriter,
        loss_dict: dict,
        time_iter: float
    ) -> None:
        if step % self.cfg.trainer.log_every_n == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train_loss/{k}", v, step)
            writer.add_scalar(f"step_time", time_iter, step)
            writer.add_scalar(f"n_sg", self.gaussians.n_sg, step)

            # Log gradients
            if self.cfg.trainer.log_grads:
                for k, v in self.gaussians.get_optimizable_dict().items():
                    grad = v.grad
                    for i in range(grad.shape[1]):
                        fig, ax = plt.subplots()
                        ax.scatter(np.arange(grad.shape[0]), grad[:, i].detach().cpu().numpy(), s=1)
                        ax.scatter(np.arange(grad.shape[0]), v[:, i].detach().cpu().numpy(), s=1, color="red")
                        writer.add_figure(f"train_grad/{k}_{i}", fig, step)
                        fig.clear()
                        plt.close(fig)

    @torch.no_grad()
    def eval(self, step: int, scene: Scene, writer: Optional[SummaryWriter] = None, load_exist_img: bool = False, obj_indices: list = []) -> None:
        if step % self.cfg.trainer.eval_every_n == 0:

            _, ray_hits, ray_on_object, _ = scene.ray_marching(downscale=self.cfg.scene.downscale_eval, obj_indices=obj_indices)
            # if self.cfg.path_to_mask_plane is not None:
            #     ray_hits = load_img(self.cfg.path_to_mask_plane, downscale=self.cfg.scene.downscale_eval)[..., 0].bool()
            plane_mask = ray_hits & (ray_on_object < 0)

            iter_save_dir = self.path_to_output / f"eval/iter_{step:06d}"
            if not iter_save_dir.exists():
                iter_save_dir.mkdir(parents=True)

            env = self.gaussians.to_envmap()
            save_img(env, iter_save_dir, "envmap")

            if load_exist_img:
                img = load_img(iter_save_dir / "image.exr")
            else:
                img = self.gaussians.render_image(scene, obj_indices=obj_indices)
            img_gt = None
            if self.cfg.path_to_img is not None:
                img_gt = scene.camera.get_img(downscale=self.cfg.scene.downscale_eval)

            if img_gt is not None:
                metrics_dict = compute_metrics(img, img_gt, env, self.gaussians.env_gt)
                with open(iter_save_dir / "eval.json", "w+") as fp:
                    json.dump(metrics_dict, fp, indent=2)
                metrics_dict_plane = compute_metrics(img, img_gt, env, self.gaussians.env_gt, plane_mask)
                with open(iter_save_dir / "eval_plane.json", "w+") as fp:
                    json.dump(metrics_dict_plane, fp, indent=2)

            save_img(img, iter_save_dir, "image")
            self.save(iter_save_dir)

            if img_gt is not None:
                img_err = torch.sum(torch.abs(img - img_gt), dim=-1)
                if plane_mask is not None:
                    img_err[~plane_mask] = -1.0
                save_img(img_err, iter_save_dir, "image_err", save_png=False)

            if self.cfg.trainer.log_sigma and self.cfg.gaussian.sigma_fn_type == "mlp":
                ray_pts, _, _, ray_normals = scene.ray_marching()
                sharpness = self.gaussians.get_attr("sharpness")
                lobe_axis = self.gaussians.get_attr("lobe_axis")
                lambda_jc = torch.norm(sharpness * lobe_axis + 2.133 * ray_normals.view(-1, 3)[..., None, :], dim=-1, keepdim=True)
                sigma = self.gaussians.get_sigma(
                    lambda_jc,
                    None,
                    lobe_axis,
                    ray_pts.view(-1, 3)
                ) # (n_pts, n_sg, 1)
                sigma_first = sigma[:, 0, 0].view(ray_pts.shape[:-1]) 
                save_img(sigma_first, iter_save_dir, "sigma")

                visibility = scene.get_visibility_sphere(ray_pts, lobe_axis)[..., 0, 0] # (n_pts, n_sg, 1)
                save_img(visibility, iter_save_dir, "visibility")

            if self.cfg.trainer.patch_based_sampling and self.cfg.trainer.reg_laplacian > 0:
                drgb = laplacian_filter3(img)
                drgb_gt = laplacian_filter3(img_gt)
                save_img(drgb, iter_save_dir, "drgb")
                save_img(drgb_gt, iter_save_dir, "drgb_gt")

            if writer is not None:
                writer.add_image("eval/image", img, step, dataformats="HWC")
                writer.add_image("eval/envmap", env, step, dataformats="HWC")
                if img_gt is not None:
                    writer.add_image("eval/image_err", img_err, step, dataformats="HW")
                    for name, metrics in metrics_dict.items():
                        for k, v in metrics.items():
                            writer.add_scalar(f"eval_{name}/{k}", v, step)
                if self.cfg.trainer.log_sigma and self.cfg.gaussian.sigma_fn_type == "mlp":
                    writer.add_image("eval/sigma", sigma_first, step, dataformats="HW")
                    writer.add_image("eval/visibility", visibility, step, dataformats="HW")
                if self.cfg.trainer.patch_based_sampling and self.cfg.trainer.reg_laplacian > 0:
                    writer.add_image("eval/drgb", drgb, step, dataformats="HWC")
                    writer.add_image("eval/drgb_gt", drgb_gt, step, dataformats="HWC")

    def save(self, save_dir: Path) -> None:
        self.gaussians.savetxt(save_dir / "gaussians_opti.txt")
        if self.cfg.gaussian.sigma_fn_type == "mlp":
            torch.save(self.gaussians.mlp.state_dict(), save_dir / "sigma_mlp.pt")

    def train(self, scene: Scene) -> None:
        self.setup_optimizers()
        writer = SummaryWriter(self.path_to_output)

        img_gt = scene.camera.get_img(downscale=self.cfg.scene.downscale)
        writer.add_image("eval/rgb_gt", img_gt, 0, dataformats="HWC")
        if self.gaussians.env_gt is not None:
            writer.add_image("eval/env_gt", self.gaussians.env_gt, 0, dataformats="HWC")
        init_envmap = self.gaussians.to_envmap()
        writer.add_image("eval/env", init_envmap, 0, dataformats="HWC")

        ray_pts, ray_hits, ray_on_object, _ = scene.ray_marching()
        height, width = ray_pts.shape[:2]
        if self.cfg.path_to_mask_plane is not None:
            ray_hits = load_img(self.cfg.path_to_mask_plane, downscale=self.cfg.scene.downscale)[..., 0].bool()
        plane_mask = ray_hits & (ray_on_object < 0)
        if self.cfg.trainer.patch_based_sampling:
            plane_mask_eroded = erode3(plane_mask)
        ray_pts_plane = ray_pts[plane_mask]
        img_gt_plane = img_gt[plane_mask]

        n_rays = ray_pts_plane.shape[0]
        batch_size = self.cfg.trainer.batch_size
        if batch_size == -1:
            batch_size = n_rays

        n = scene.plane_normal.repeat(batch_size, 1)

        for step in (pbar := tqdm(range(1, self.cfg.trainer.n_iters + 1))):
            pbar.set_description("Training: ")

            tstart_iter = timeit.default_timer()

            self.optimizer.zero_grad()

            if self.cfg.trainer.patch_based_sampling:
                patch_size = int(np.sqrt(batch_size))
                u = np.random.randint(0, width - patch_size)
                v = np.random.randint(0, height - patch_size)
                plane_m = plane_mask[v:v + patch_size, u:u + patch_size]
                plane_m_eroded = plane_mask_eroded[v:v + patch_size, u:u + patch_size]
                pts = ray_pts[v:v + patch_size, u:u + patch_size]
                gt = img_gt[v:v + patch_size, u:u + patch_size]
                n = n.view(patch_size, patch_size, 3)
            else:
                ray_indices = torch.randperm(n_rays, device=self.device)[:batch_size]
                pts = ray_pts_plane[ray_indices]
                gt = img_gt_plane[ray_indices]

            if self.cfg.gaussian.sigma_fn_type != "mlp":
                thetas = scene.get_ssdf(pts, self.gaussians.get_lobe_axis())
            else:
                thetas = None

            render_dict = self.gaussians.integrate(n, thetas, pts)
            rgb = scene.get_plane_diffuse() * render_dict["irradiance"]

            if self.cfg.trainer.patch_based_sampling:
                rgb = rgb * plane_m.unsqueeze(-1)

            loss_dict = {}
            loss_dict["rgb_loss"] = torch.nn.functional.mse_loss(rgb, gt)

            if self.cfg.trainer.patch_based_sampling and self.cfg.trainer.reg_laplacian > 0:
                drgb = laplacian_filter3(rgb)
                drgb = drgb * plane_m_eroded
                drgb_gt = laplacian_filter3(gt)
                drgb_gt = drgb_gt * plane_m_eroded
                loss_dict["reg_laplacian"] = self.cfg.trainer.reg_laplacian * torch.nn.functional.mse_loss(drgb, drgb_gt)

            # Regularization of visibility when using MLP
            if self.cfg.gaussian.sigma_fn_type == "mlp":
                sigma = render_dict["sigma"] # (n_pts, n_sg, 1)
                visibility = scene.get_visibility_sphere(pts, self.gaussians.get_lobe_axis(), chunk_size=batch_size) # (n_pts, n_sg, 1)
                loss_dict["reg_visbility"] = 0.01 * torch.nn.functional.mse_loss(sigma, visibility)

            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()
            if self.cfg.trainer.lr_scheduler_step > 0 and step % self.cfg.trainer.lr_scheduler_step == 0:
                self.scheduler.step()

            tend_iter = timeit.default_timer()

            pbar.set_postfix({"loss": f"{loss.item():.6f}"})
            self.train_report(step, writer, loss_dict, tend_iter-tstart_iter)
            self.eval(step, scene, writer=writer)
            if step >= self.cfg.trainer.adapt_control_from_iter and step < self.cfg.trainer.adapt_control_to_iter and step % self.cfg.trainer.adapt_control_every_iter == 0:
                self.gaussians.adaptive_control(self.optimizer)

        writer.close()

    def fit_envmap(self, scene: Scene) -> None:
        self.setup_optimizers()
        writer = SummaryWriter(self.path_to_output)
        envmap_gt = self.gaussians.env_gt
        H, W = envmap_gt.shape[:2]
        assert envmap_gt is not None
        writer.add_image("eval/env_gt", envmap_gt, dataformats="HWC")

        theta, phi = torch.meshgrid([torch.linspace(-np.pi, np.pi, W), torch.linspace(0., np.pi, H)], indexing="xy")
        viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)
        viewdirs = viewdirs.view(-1, 3).to(self.device) # H W 3
        envmap_gt_flat = envmap_gt.view(-1, 3)

        for step in (pbar := tqdm(range(1, self.cfg.trainer.n_iters + 1))):
            pbar.set_description("Fitting to envmap")

            tstart_iter = timeit.default_timer()
            self.optimizer.zero_grad()
            viewdir_inds = torch.randperm(H * W, device=self.device)[:self.cfg.trainer.batch_size]
            envmap = torch.sum(self.gaussians.eval(viewdirs[viewdir_inds]), dim=-2)
            loss_dict = {
                "mse_err": torch.mean((envmap - envmap_gt_flat[viewdir_inds]) ** 2),
            }
            loss = sum(loss_dict.values())
            loss.backward()
            self.optimizer.step()
            t_end_iter = timeit.default_timer()

            self.train_report(step, writer, loss_dict, t_end_iter - tstart_iter)
            self.eval(step, scene, writer=writer)

        writer.close()

class LSTSQTrainer:
    def __init__(self, config: Config) -> None:
        self.cfg = config
        self.device = config.base.device
        self.path_to_output = Path(config.base.path_to_output)
        self.env_gt = None
        if config.path_to_env is not None:
            self.env_gt = load_img(config.path_to_env, self.device)

    def __sample_directions(self, height: int) -> Tensor:
        width = height * 2
        theta, phi = torch.meshgrid([torch.linspace(-np.pi, np.pi, width), torch.linspace(0., np.pi/2, height//2)],
            indexing="xy")
        theta = theta.to(self.device)
        phi = phi.to(self.device)
        samples_d = torch.stack([torch.cos(theta) * torch.sin(phi),
                                torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)
        samples_d = samples_d.to(self.device)  # H/2 W 3
        return samples_d, theta, phi

    def train(self, scene: Scene) -> None:
        writer = SummaryWriter(self.path_to_output)

        img_gt = scene.camera.get_img(downscale=self.cfg.scene.downscale)
        writer.add_image("eval/rgb_gt", img_gt, 0, dataformats="HWC")
        if self.env_gt is not None:
            writer.add_image("eval/env_gt", self.env_gt, 0, dataformats="HWC")
        
        # Direction sampling
        env_height = self.cfg.gaussian.envmap_height
        env_width = env_height * 2
        samples_d, theta, phi = self.__sample_directions(env_height)
        samples_d_flat = samples_d.view(-1, 3)  # n_d 3
        sine = torch.sin(phi[..., None].reshape(-1, 1)) * (2*np.pi/env_width) * (np.pi/env_height)  # n_d 1

        ray_pts, ray_hits, ray_on_object, ray_normals = scene.ray_marching()
        if self.cfg.path_to_mask_plane is not None:
            ray_hits = load_img(self.cfg.path_to_mask_plane, downscale=self.cfg.scene.downscale)[..., 0].bool()
        plane_mask = ray_hits & (ray_on_object < 0)
        ray_pts = ray_pts[plane_mask]
        ray_normals = ray_normals[plane_mask]
        ray_vis = scene.get_visibility_mesh(ray_pts, samples_d_flat)[..., None]

        # LSTSQ
        time_start = timeit.default_timer()
        cosine = torch.sum(ray_normals.unsqueeze(-2) * samples_d_flat, dim=-1, keepdim=True).clamp_min(0.0)  # n_p n_d 1
        diffuse = torch.ones(*cosine.shape[:-1], 3).to(cosine)
        A = diffuse * ray_vis * cosine * sine  # n_p n_d 3
        A = A.permute(2, 0, 1)  # 3 n_p n_d
        D = laplacian_2d(env_height//2, env_width).to(A)
        B = img_gt[plane_mask] # n_p 3
        B = B.T.unsqueeze(-1)  # 3 n_p 1
        if self.cfg.trainer.reg_lstsq_1 > 0 or self.cfg.trainer.reg_lstsq_2 > 0:
            left = A.permute(0, 2, 1) @ A + self.cfg.trainer.reg_lstsq_1 * D.T @ D + self.cfg.trainer.reg_lstsq_2 * torch.eye(A.shape[-1], device=self.device)
            right = A.permute(0, 2, 1) @ B
        else:
            left = A
            right = B
        try:
            X_nnls = []
            for ch in range(left.shape[0]):
                x = nnls(left[ch].cpu().numpy(), right[ch, :, 0].cpu().numpy(), atol=0.1)[0]
                X_nnls.append(torch.from_numpy(x).to(self.device))
            X_nnls = torch.cat(X_nnls).to(torch.float32)
            envmap_nnls = X_nnls.view(3, env_height//2, env_width).permute(1, 2, 0)
        except Exception as e:
            print("Error when computing nnls:", e)
            envmap_nnls = None
        X = torch.linalg.lstsq(left.cpu(), right.cpu(), driver="gelsd")
        X = X.solution.to(self.device)
        envmap = X.view(3, env_height//2, env_width).permute(1, 2, 0)
        time_end = timeit.default_timer()

        self.eval(scene, envmap, envmap_nnls, time_end - time_start, writer)

    def __eval_envmap(self, path_to_output: Path, scene: Scene, envmap: Tensor, writer: SummaryWriter = None,
        plane_mask: Tensor = None, load_exist_img: bool = False, obj_indices: list = []) -> None:
        if not path_to_output.exists():
            path_to_output.mkdir(parents=True)

        save_img(envmap, path_to_output, "envmap")

        if load_exist_img:
            img = load_img(path_to_output / "image.exr")
        else:
            img = self.render(scene, envmap, obj_indices=obj_indices)
        img_gt = None
        if self.cfg.path_to_img is not None:
            img_gt = scene.camera.get_img(downscale=self.cfg.scene.downscale_eval)

        save_img(img, path_to_output, "image")

        if img_gt is not None:
            env_gt_eval = self.env_gt
            envmap_eval = torch.cat([envmap, torch.zeros_like(envmap)], dim=0)
            if env_gt_eval is not None:
                envmap_eval = torchvision.transforms.Resize((self.env_gt.shape[0], self.env_gt.shape[1]))(
                    envmap_eval.permute([2, 0, 1])).permute([1, 2, 0])
            metrics_dict = compute_metrics(img, img_gt, envmap_eval, env_gt_eval)
            save_img(envmap_eval, path_to_output, "envmap_eval")
            with open(path_to_output / "eval.json", "w") as fp:
                json.dump(metrics_dict, fp, indent=2)
            if plane_mask is not None:
                metrics_dict_plane = compute_metrics(img, img_gt, envmap_eval, env_gt_eval, plane_mask)
                with open(path_to_output / "eval_plane.json", "w+") as fp:
                    json.dump(metrics_dict_plane, fp, indent=2)

        if img_gt is not None:
            img_err = torch.sum(torch.abs(img - img_gt), dim=-1)
            if plane_mask is not None:
                img_err[~plane_mask] = -1.0
            save_img(img_err, path_to_output, "image_err", save_png=False)

        if writer is not None:
            writer.add_image(f"{path_to_output.stem}/image", img, 0, dataformats="HWC")
            writer.add_image(f"{path_to_output.stem}/envmap", envmap, 0, dataformats="HWC")
            if img_gt is not None:
                writer.add_image(f"{path_to_output.stem}/image_err", img_err, 0, dataformats="HW")
                for name, metrics in metrics_dict.items():
                    for k, v in metrics.items():
                        if v is not None:
                            writer.add_scalar(f"{path_to_output.stem}_{name}/{k}", v, 0)

    def eval(self, scene: Scene, envmap: Tensor, envmap_nnls: Tensor, elapsed: float = None,
        writer: Optional[SummaryWriter] = None, load_exist_img: bool = False, obj_indices: list = []) -> None:
        if elapsed is not None:
            with open(self.path_to_output / "time.txt", "w") as fp:
                fp.write(f"{elapsed}")
        _, ray_hits, ray_on_object, _ = scene.ray_marching(downscale=self.cfg.scene.downscale_eval, obj_indices=obj_indices)
        # if self.cfg.path_to_mask_plane is not None:
        #     ray_hits = load_img(self.cfg.path_to_mask_plane, downscale=self.cfg.scene.downscale_eval)[..., 0].bool()
        plane_mask = ray_hits & (ray_on_object < 0)
        if envmap is not None:
            self.__eval_envmap(self.path_to_output / "envmap", scene, envmap, writer, plane_mask, load_exist_img, obj_indices)
        if envmap_nnls is not None:
            self.__eval_envmap(self.path_to_output / "envmap_nnls", scene, envmap_nnls, writer, plane_mask, load_exist_img, obj_indices)

    def render(self, scene: Scene, envmap: Tensor, obj_indices: list = []) -> Tensor:
        ray_pts, ray_hits, ray_on_objects, ray_normals = scene.ray_marching(downscale=self.cfg.scene.downscale_eval,
            obj_indices=obj_indices)
        ray_pts_flat = ray_pts.view(-1, 3)
        ray_hits_flat = ray_hits.view(-1)
        ray_on_objects_flat = ray_on_objects.view(-1)
        ray_normals_flat = ray_normals.view(-1, 3)

        radiances = []

        env_height = self.cfg.gaussian.envmap_height
        env_width = env_height * 2
        samples_d, theta, phi = self.__sample_directions(env_height)
        samples_d_flat = samples_d.view(-1, 3)
        sine = torch.sin(phi[..., None].reshape(-1, 1)) * (2*np.pi/env_width) * (np.pi/env_height)  # n_d 1

        bs = self.cfg.gaussian.render_chunk_size
        for i in tqdm(range(0, ray_pts_flat.shape[0], bs), desc="Rendering"):
            hits = ray_hits_flat[i:i+bs]
            r = torch.zeros(hits.shape[0], 3, device=self.device)
            if not hits.any():
                radiances.append(r)
                continue
            on_objects = ray_on_objects_flat[i:i+bs][hits]
            pts = ray_pts_flat[i:i+bs][hits]
            n = ray_normals_flat[i:i+bs][hits]
            vis = scene.get_visibility_sphere(pts, samples_d_flat, obj_indices=obj_indices)
            cosine = torch.sum(n.unsqueeze(-2) * samples_d_flat, dim=-1, keepdim=True).clamp_min(0.0)
            diffuse = scene.get_diffuse(on_objects)
            diffuse = diffuse[..., None, :].broadcast_to(*cosine.shape[:-1], 3)
            r[hits] = torch.sum(diffuse * vis * cosine * sine * envmap.view(-1, 3), dim=-2)
            radiances.append(r)
        
        radiances = torch.cat(radiances).view(ray_pts.shape)

        return radiances
