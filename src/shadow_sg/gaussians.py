from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from jaxtyping import Float
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from torch import Tensor, nn
from tqdm import tqdm

from shadow_sg import EPS
from shadow_sg.config import Config
from shadow_sg.io import load_img
from shadow_sg.scene import Scene
from shadow_sg.utils import (build_covariance, fibonacci_sphere_sampling,
                             rotation_between_vectors)


class SphericalGaussians():
    def __init__(self,
        cfg: Config,
        hemisph_dir: Optional[Float[Tensor, "H W 3"]] = None,
        is_eval: bool = False
    ) -> None:
        self.cfg = cfg
        self.device = self.cfg.base.device
        self.env_gt = None
        if self.cfg.path_to_env is not None:
            self.env_gt = load_img(self.cfg.path_to_env, self.device)
        self.hemisph_dir = hemisph_dir.to(self.device) if isinstance(hemisph_dir, Tensor) else None
        self.is_eval = is_eval

        if self.cfg.gaussian.sigma_fn_type == "k":
            self.k_coeffs = np.loadtxt(self.cfg.gaussian.path_to_sigma_ckpt, dtype=np.float32)
            self.k_coeffs = torch.tensor(self.k_coeffs, device=self.device)
        elif self.cfg.gaussian.sigma_fn_type == "mlp_ssdf":
            # self.mlp = tcnn.Network(
            #     n_input_dims=3 if self.cfg.gaussian.anistropic else 2, # (lambd, theta[, tangent])
            #     n_output_dims=1,
            #     network_config={
            #         "otype": "FullyFusedMLP",
            #         "activation": "ReLU",
            #         "output_activation": "Sigmoid",
            #         "n_neurons": 128,
            #         "n_hidden_layers": 4,
            #     }
            # )
            # self.mlp.load_state_dict(torch.load(self.cfg.gaussian.path_to_sigma_ckpt))
            assert self.cfg.gaussian.path_to_sigma_ckpt != "", "Path to mlp checkpoint is not specified for mlp-based ssdf"
            self.mlp = torch.load(self.cfg.gaussian.path_to_sigma_ckpt).to(self.device)
        elif self.cfg.gaussian.sigma_fn_type == "mlp":
            # self.mlp = tcnn.NetworkWithInputEncoding(
            #     n_input_dims=8 if self.cfg.gaussian.anistropic else 7, # (xyz, mu, lambd[, tangent])
            #     n_output_dims=1,
            #     network_config={
            #         "otype": "CutlassMLP",
            #         "activation": "ReLU",
            #         "output_activation": "Sigmoid",
            #         "n_neurons": 128,
            #         "n_hidden_layers": 4,
            #     }
            # )
            self.mlp = nn.Sequential(
                nn.Linear(8 if self.cfg.gaussian.anistropic else 7, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, 1),
                nn.Sigmoid()
            ).to(self.device)
            if self.cfg.gaussian.path_to_sigma_ckpt:
                self.mlp.load_state_dict(torch.load(self.cfg.gaussian.path_to_sigma_ckpt))

        self.init()

    def get_sigma(self, lambd: Tensor, theta: Tensor, mu: Tensor, xyz: Tensor) -> Tensor:
        if self.cfg.gaussian.sigma_fn_type == "k":
            k = self.k_coeffs[0] * lambd ** 3 + self.k_coeffs[1] * lambd ** 2 + self.k_coeffs[2] * lambd + self.k_coeffs[3]
            # k = self.k_coeffs[1] * lambd ** 2 + self.k_coeffs[2] * lambd + self.k_coeffs[3]
            sigma = torch.sigmoid(k * theta)
        if self.cfg.gaussian.sigma_fn_type == "mlp_ssdf":
            mlp_input = torch.cat((lambd, theta), dim=-1)
            mlp_input_flat = mlp_input.view(-1, mlp_input.shape[-1])
            sigma = self.mlp(mlp_input_flat)
            sigma = sigma.view(*mlp_input.shape[:-1], -1)
        if self.cfg.gaussian.sigma_fn_type == "mlp":
            xyz = xyz[:, None, :].broadcast_to(xyz.shape[0], mu.shape[0], xyz.shape[1]) # n_pts n_sg 3
            mu = mu[None, ...].broadcast_to(xyz.shape[0], *mu.shape) # n_pts n_sg 3
            mlp_input = torch.cat((xyz, mu, lambd), dim=-1)
            mlp_input_flat = mlp_input.view(-1, mlp_input.shape[-1])
            sigma = self.mlp(mlp_input_flat.detach())
            sigma = sigma.view(*mlp_input.shape[:-1], -1)
        return sigma

    @torch.no_grad()
    def init(self) -> None:
        if self.cfg.gaussian.path_to_ckpt:
            sgs = torch.tensor(np.loadtxt(self.cfg.gaussian.path_to_ckpt, ndmin=2), dtype=torch.float32, device=self.device)
        else:
            n_sg = self.cfg.gaussian.n_gaussians
            lobe_axis = fibonacci_sphere_sampling(n_sg, device=self.device) # n_sg 3
            if self.hemisph_dir is not None:
                dot = torch.sum(lobe_axis * self.hemisph_dir, dim=-1, keepdim=True) # n_sg 1
                out_mask = dot[..., 0] < 0 # n_sg
                lobe_axis[out_mask] = lobe_axis[out_mask] - 2 * dot[out_mask] * self.hemisph_dir
            if self.cfg.gaussian.sharpness_activation_fn == "exp":
                sharpness = -torch.rand((n_sg, 1), device=self.device) * 3.0
            elif self.cfg.gaussian.sharpness_activation_fn == "identity":
                sharpness = torch.rand((n_sg, 1), device=self.device) * (np.exp(1.0) - 1) + 1
            else:
                raise ValueError(f"Invalid sharpness activation function: {self.cfg.gaussian.sharpness_activation_fn}")
            amplitude = torch.rand((n_sg, 1), device=self.device).repeat(1, 3) # monochrome
            sgs = torch.cat([lobe_axis, sharpness, amplitude], dim=-1)

        if self.cfg.gaussian.sigma_fn_type == "k" and self.cfg.gaussian.optimizable_k:
            self.k_coeffs = nn.Parameter(self.k_coeffs.detach().clone(), requires_grad=True)
            # self.k_coeffs = nn.Parameter(torch.rand((4,), device=self.device), requires_grad=True)
            # self.k_coeffs = nn.Parameter(torch.zeros((4,), device=self.device), requires_grad=True)
            # self.k_coeffs = nn.Parameter(torch.ones((4,), device=self.device), requires_grad=True)

        self._attrs = {
            "lobe_axis": nn.Parameter(sgs[:, :3].detach().clone(), requires_grad=not self.is_eval),
            "sharpness": nn.Parameter(sgs[:, 3:4].detach().clone(), requires_grad=not self.is_eval),
            "amplitude": nn.Parameter(sgs[:, 4:].detach().clone(), requires_grad=not self.is_eval),
        }

    @property
    def n_sg(self) -> int:
        return list(self._attrs.values())[0].shape[0]

    def get_optimizable_dict(self) -> dict:
        return self._attrs

    def get_lobe_axis(self) -> Float[Tensor, "n_sg 3"]:
        return self.get_attr("lobe_axis")

    @torch.no_grad()
    def set_sharpness(self, sharpness: Tensor) -> None:
        self._attrs["sharpness"].data.copy_(sharpness)

    @torch.no_grad()
    def set_amplitude(self, amplitude: Tensor) -> None:
        self._attrs["amplitude"].data.copy_(amplitude)

    @torch.no_grad()
    def savetxt(self, fname: Tuple[Path, str]) -> None:
        np.savetxt(fname, torch.cat(list(self._attrs.values()), dim=-1).detach().cpu().numpy())
        if self.cfg.gaussian.sigma_fn_type == "k" and self.cfg.gaussian.optimizable_k:
            np.savetxt(fname.parent / "k_coeffs.txt", self.k_coeffs.detach().cpu().numpy())

    def get_attr(self, name: str) -> Tensor:
        attr = self._attrs[name]
        if name == "lobe_axis":
            if self.cfg.gaussian.lobe_axis_activation_fn == "normalize":
                attr = nn.functional.normalize(attr, dim=-1)
            elif self.cfg.gaussian.lobe_axis_activation_fn == "identity":
                pass
            else:
                raise ValueError(f"Invalid lobe_axis_activation_fn: {self.cfg.gaussian.lobe_axis_activation_fn}")
        if name == "amplitude":
            if self.cfg.gaussian.amplitude_activation_fn == "abs":
                attr = torch.abs(attr)
            elif self.cfg.gaussian.amplitude_activation_fn == "identity":
                pass
            else:
                raise ValueError(f"Invalid amplitude_activation_fn: {self.cfg.gaussian.amplitude_activation_fn}")
        if name == "sharpness":
            if self.cfg.gaussian.sharpness_activation_fn == "exp":
                attr = torch.exp(-attr)
            elif self.cfg.gaussian.sharpness_activation_fn == "identity":
                pass
            else:
                raise ValueError(f"Invalid sharpness_activation_fn: {self.cfg.gaussian.sharpness_activation_fn}")
        return attr

    @torch.no_grad()
    def normalize(self) -> None:
        self._attrs["lobe_axis"].data = nn.functional.normalize(self._attrs["lobe_axis"].data, dim=-1)

    def eval(self, v: Float[Tensor, "n_dirs 3"]):
        lobes = self.get_attr("lobe_axis") # n_sg 3
        sharpness = self.get_attr("sharpness") # n_sg 1
        amplitude = self.get_attr("amplitude") # n_sg 3
        vals = amplitude * torch.exp(sharpness * (torch.sum(v[..., None, :] * lobes, dim=-1, keepdim=True) - 1.)) # n_dirs n_sg 3
        return vals

    @torch.no_grad()
    def adaptive_control(self, optimizer: torch.optim.Optimizer) -> None:
        amplitude = self._attrs["amplitude"] # n_sg 3
        amplitude_grad = amplitude.grad # n_sg 3
        prune_mask = (amplitude.abs().amax(dim=-1) < self.cfg.gaussian.prune_by_amplitude_threshold) & \
            (amplitude_grad.abs().amax(dim=-1) < self.cfg.gaussian.prune_by_amplitude_grad_threshold)
        if prune_mask.any():
            self._prune_by_mask(prune_mask, optimizer)

    @torch.no_grad()
    def _prune_by_mask(self, mask: Tensor, optimizer: torch.optim.Optimizer) -> None:
        optimizable_dict = {}
        for group in optimizer.param_groups:
            if group["name"] not in self._attrs:
                continue
            stored_state = optimizer.state.get(group["params"][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][~mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][~mask]
                del optimizer.state[group["params"][0]]
                optimizer.state[group["params"][0]] = stored_state
            group["params"][0] = nn.Parameter(group["params"][0][~mask].requires_grad_(True))
            optimizable_dict[group["name"]] = group["params"][0]
        self._attrs = optimizable_dict

    @torch.no_grad()
    def to_envmap(self, draw_inds: bool = False) -> Float[Tensor, "height 2*height 3"]:
        if self.cfg.gaussian.envmap_height == -1 and self.env_gt is not None:
            height = self.env_gt.shape[0]
        else:
            height = self.cfg.gaussian.envmap_height
        assert height > 0
        H = height
        W = height * 2
        theta, phi = torch.meshgrid([torch.linspace(-np.pi, np.pi, W), torch.linspace(0., np.pi, H)], indexing="xy")
        viewdirs = torch.stack([torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)], dim=-1)
        viewdirs = viewdirs.to(self.device) # H W 3

        env = self.eval(viewdirs) # H W n_sg 3
        if self.hemisph_dir is not None:
            z = self.get_lobe_axis() # n_sg 3
            env = (torch.sum(z * self.hemisph_dir, dim=-1, keepdim=True) > 0) * env
        env = torch.sum(env, dim=-2)

        if draw_inds:
            lobe_axis = self.get_lobe_axis() # n_sg 3
            theta = torch.atan2(lobe_axis[..., 1], lobe_axis[..., 0]) # n_sg
            phi = torch.acos(lobe_axis[..., 2]) # n_sg
            u = W * (theta + np.pi) / (2 * np.pi) # n_sg
            v = H * (phi / np.pi) # n_sg
            def draw_text(img, text,
                pos=(0, 0),
                font=cv2.FONT_HERSHEY_PLAIN,
                font_scale=1,
                font_thickness=1,
                text_color=(0, 255, 0),
                text_color_bg=(0, 0, 0)
            ):
                x, y = pos
                text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
                text_w, text_h = text_size
                cv2.rectangle(img, pos, (x + text_w, y + text_h), text_color_bg, -1)
                cv2.putText(img, text, (x, y + text_h + font_scale - 1), font, font_scale, text_color, font_thickness)

            env_np = env.detach().cpu().numpy()
            for i in range(u.shape[0]):
                draw_text(env_np, str(i), (int(u[i].item()), int(v[i].item())))
            env = torch.tensor(env_np, dtype=env.dtype, device=env.device)

        return env

    def integrate(self,
        normals: Float[Tensor, "num_rays 3"],
        thetas: Float[Tensor, "num_rays num_sgs 1"] = None,
        xyz: Float[Tensor, "num_rays 3"] = None
    ) -> Float[Tensor, "num_rays 3"]:

        lobe_axis = self.get_attr("lobe_axis")
        sharpness = self.get_attr("sharpness")
        amplitude = self.get_attr("amplitude")

        lambda_m = sharpness + 2.133  # num_sgs 1
        lambda_jc = torch.norm(sharpness * lobe_axis + 2.133 * normals[..., None, :], dim=-1, keepdim=True)
        a_jc = amplitude * 1.170 * torch.exp(lambda_jc - lambda_m)

        sigma = self.get_sigma(lambda_jc, thetas, lobe_axis, xyz) # num_rays num_sgs 1
        f = 2 * torch.pi * (1 - torch.exp(-lambda_jc)) / lambda_jc  # num_rays num_sgs 1
        irradiance = a_jc * sigma * f  # num_rays num_sgs 3
        if self.hemisph_dir is not None:
            irradiance = (torch.sum(lobe_axis * self.hemisph_dir, dim=-1, keepdim=True) > 0) * irradiance
        irradiance = torch.sum(irradiance, dim=-2)  # num_rays 3

        return {
            "irradiance": irradiance,
            "sigma": sigma
        }

    @torch.no_grad()
    def render_image(self,
        scene: Scene,
        obj_indices: list = [],
    ) -> Float[Tensor, "H W 3"]:

        ray_pts, ray_hits, ray_on_obj, ray_normals = scene.ray_marching(obj_indices=obj_indices,
            downscale=self.cfg.scene.downscale_eval)
        # if self.cfg.path_to_mask_plane is not None:
        #     ray_hits = load_img(self.cfg.path_to_mask_plane, downscale=self.cfg.scene.downscale_eval)[..., 0].bool()
        ray_pts_flat = ray_pts.view(-1, 3)
        ray_hits_flat = ray_hits.view(-1)
        ray_on_obj_flat = ray_on_obj.view(-1)
        ray_normals_flat = ray_normals.view(-1, 3)
        rgb = []

        for i in (pbar := tqdm(range(0, ray_pts_flat.shape[0], self.cfg.gaussian.render_chunk_size))):
            pbar.set_description("Rendering")

            pts = ray_pts_flat[i:i+self.cfg.gaussian.render_chunk_size]
            n = ray_normals_flat[i:i+self.cfg.gaussian.render_chunk_size]
            on_obj = ray_on_obj_flat[i:i+self.cfg.gaussian.render_chunk_size]

            if self.cfg.gaussian.sigma_fn_type != "mlp":
                thetas = scene.get_ssdf(pts, self.get_lobe_axis(), obj_indices=obj_indices)
            else:
                thetas = None

            if self.cfg.scene.ssdf_method == "mesh" and on_obj.any():
                thetas[on_obj] = scene.get_ssdf_sphs_level_min(pts[on_obj], self.get_lobe_axis(), obj_indices=obj_indices)

            render_dict = self.integrate(n, thetas, pts)
            irradiance = render_dict["irradiance"]
            diffuse = scene.get_diffuse(on_obj)
            irradiance *= diffuse
            rgb.append(irradiance)

        rgb = torch.cat(rgb)
        rgb[~ray_hits_flat] = 0
        rgb = rgb.view(*ray_pts.shape)

        return rgb

    @torch.no_grad()
    def render_images_individual_gaussians(self, scene: Scene, obj_indices: list = []):
        attrs = {k: v.data.clone() for k, v in self._attrs.items()}
        n_gaussians = list(attrs.values())[0].shape[0]
        for i in range(n_gaussians):
            self._attrs = {k: v[i:i+1] for k, v in attrs.items()}
            if torch.sum(self.get_lobe_axis() * self.hemisph_dir, dim=-1).item() > 0:
                yield self.render_image(scene, obj_indices=obj_indices)
        self._attrs = {k: nn.Parameter(v.detach().clone(), requires_grad=not self.is_eval) for k, v in attrs.items()}

class AniSphericalGaussians(SphericalGaussians):
    def __init__(self,
        cfg: Config,
        hemisph_dir: Optional[Float[Tensor, "H W 3"]] = None,
        is_eval: bool = False
    ) -> None:
        super(AniSphericalGaussians, self).__init__(cfg, hemisph_dir, is_eval)

    def init(self) -> int:
        if self.cfg.gaussian.path_to_ckpt:
            sg = torch.tensor(np.loadtxt(self.cfg.gaussian.path_to_ckpt, ndmin=2), dtype=torch.float32, device=self.device)
        else:
            n_sg = self.cfg.gaussian.n_gaussians
            z = fibonacci_sphere_sampling(n_sg, device=self.device)
            if self.hemisph_dir is not None:
                dot = torch.sum(z * self.hemisph_dir, dim=-1, keepdim=True) # n_sg 1
                out_mask = dot[..., 0] < 0 # n_sg
                z[out_mask] = z[out_mask] - 2 * dot[out_mask] * self.hemisph_dir
            R = rotation_between_vectors(torch.tensor([0, 0, 1.0], device=self.device)[None].repeat(n_sg, 1), z)
            quat = matrix_to_quaternion(R) # n_sg 4
            scale = torch.rand((n_sg, 2), device=self.device)
            c = torch.torch.rand((n_sg, 3), device=self.device)
            sg = torch.cat([quat, scale, c], dim=-1)

        self._attrs = {
            "rotation": nn.Parameter(sg[:, :4].detach().clone(), requires_grad=not self.is_eval),
            "scale": nn.Parameter(sg[:, 4:6].detach().clone(), requires_grad=not self.is_eval),
            "amplitude": nn.Parameter(sg[:, 6:9].detach().clone(), requires_grad=not self.is_eval),
        }

    @torch.no_grad()
    def normalize(self) -> None:
        self._attrs["rotation"] = nn.functional.normalize(self._attrs["rotation"], dim=-1)

    @torch.no_grad()
    def savetxt(self, fname: Tuple[Path, str]) -> None:
        np.savetxt(fname, torch.cat(list(self._attrs.values()), dim=-1).detach().cpu().numpy())

    def get_lobe_axis(self) -> Float[Tensor, "n_asg 3"]:
        quat = self.get_attr("rotation")
        rot = quaternion_to_matrix(quat)
        return rot[..., -1]

    @torch.no_grad()
    def set_sharpness(self, sharpness: Float[Tensor, "n_sg 1"]) -> None:
        if sharpness.shape[-1] == 1:
            sharp = torch.cat((sharpness, sharpness), dim=-1)
        else:
            sharp = sharpness
        self._attrs["scale"].data.copy_(sharp)

    @torch.no_grad()
    def set_tangent(self, tangent: Float[Tensor, "n_sg 1"]) -> None:
        z_axis = torch.tensor([[0, 0, 1.0]], device=self.device).broadcast_to(self.n_sg, 3)
        R = rotation_between_vectors(z_axis, self.get_lobe_axis())
        delta_R = torch.zeros((self.n_sg, 3, 3), device=self.device)
        delta_R[..., 0, 0] = torch.cos(tangent)[..., 0]
        delta_R[..., 0, 1] = -torch.sin(tangent)[..., 0]
        delta_R[..., 1, 0] = torch.sin(tangent)[..., 0]
        delta_R[..., 1, 1] = torch.cos(tangent)[..., 0]
        delta_R[..., 2, 2] = 1.0
        R = R @ delta_R
        self._attrs["rotation"].data.copy_(matrix_to_quaternion(R))

    def get_optimizable_dict(self) -> dict:
        return self._attrs

    def get_attr(self, name: str) -> Tensor:
        if name == "tangent":
            R = quaternion_to_matrix(self.get_attr("rotation"))
            z_axis = torch.tensor([[0, 0, 1.0]], device=self.device).broadcast_to(self.n_sg, 3)
            R_z = rotation_between_vectors(z_axis, self.get_lobe_axis())
            attr = torch.acos(torch.clamp(torch.sum(R_z[..., 0] * R[..., 0], dim=-1, keepdim=True), -1.0, 1.0))
            return attr
        attr = self._attrs[name]
        if name == "amplitude":
            attr = torch.abs(attr)
        if name == "rotation":
            attr = nn.functional.normalize(attr)
        if name == "scale":
            attr = torch.reciprocal(torch.exp(attr))
        return attr

    def get_sigma(self, lambd: Tensor, theta: Tensor, mu: Tensor, xyz: Tensor, tangent: Tensor) -> Tensor:
        if self.cfg.gaussian.sigma_fn_type == "k":
            k1, k2, k3, k4 = self.k_coeffs
            k = k1 * lambd ** 3 + k2 * lambd ** 2 + k3 * lambd + k4 # n_asg 1
            sigma = 1.0 / (1.0 + torch.exp(k * -theta)) # n_rays n_asg 1
        if self.cfg.gaussian.sigma_fn_type == "mlp_ssdf":
            lambd = lambd[None, ...].broadcast_to(theta.shape[0], *lambd.shape)
            tangent = tangent[None, ...].broadcast_to(theta.shape[0], *tangent.shape)
            mlp_input = torch.cat((lambd, theta, tangent), dim=-1)
            mlp_input_flat = mlp_input.view(-1, mlp_input.shape[-1])
            sigma = self.mlp(mlp_input_flat)
            sigma = sigma.view(*mlp_input.shape[:-1], -1)
        if self.cfg.gaussian.sigma_fn_type == "mlp":
            xyz = xyz[:, None, :].broadcast_to(xyz.shape[0], lambd.shape[0], xyz.shape[1]) # n_pts n_sg 3
            mu = mu[None, ...].broadcast_to(xyz.shape[0], *mu.shape) # n_pts n_sg 3
            lambd = lambd[None, ...].broadcast_to(xyz.shape[0], *lambd.shape) # n_pts n_sg 1
            mlp_input = torch.cat((xyz, mu, lambd), dim=-1)
            mlp_input_flat = mlp_input.view(-1, mlp_input.shape[-1])
            sigma = self.mlp(mlp_input_flat)
            sigma = sigma.view(*mlp_input.shape[:-1], -1)
        return sigma

    def get_cov(self) -> Float[Tensor, "n_asg 3 3"]:
        quat = self.get_attr("rotation") # n_asg 4
        scale = self.get_attr("scale") # n_asg 2
        return build_covariance(quat, scale)

    def get_cov_clamped_cos(self, n: Float[Tensor, "* 3"]) -> Float[Tensor, "* 3 3"]:
        R = rotation_between_vectors(torch.tensor([0, 0, 1.0], device=self.device)[None].repeat(*n.shape[:-1], 1), n)
        scale = torch.ones((*n.shape[:-1], 2), device=self.device) * 2.133
        return build_covariance(R, scale)

    def eval(self, v: Float[Tensor, "n_dirs 3"]) -> Float[Tensor, "n_dirs n_asg C"]:
        c = self.get_attr("amplitude") # n_asg 3
        R = quaternion_to_matrix(self.get_attr("rotation"))
        scale = self.get_attr("scale")
        x, y, z = R[..., :, 0], R[..., :, 1], R[..., :, 2]
        lambd, mu = scale[..., 0:1], scale[..., 1:2]
        v = v[..., None, :]
        smooth = torch.maximum(torch.sum(v * z, dim=-1, keepdim=True), torch.tensor(0.0, device=self.device)) # n_dirs n_asg 1
        lambda_term = lambd * (torch.sum(v * x, dim=-1, keepdim=True) ** 2) # n_dirs n_asg 1
        eta_term = mu * (torch.sum(v * y, dim=-1, keepdim=True) ** 2) # n_dirs n_asg 1
        exp = torch.exp(-lambda_term - eta_term) # n_dirs n_asg 1
        return c * smooth * exp

    def integrate(self,
        normals: Float[Tensor, "n_rays 3"],
        thetas: Float[Tensor, "n_rays n_asg 1"] = None,
        xyz: Float[Tensor, "n_rays 3"] = None,
        ) -> Float[Tensor, "n_rays 3"]:

        # cov_c = self.get_cov_clamped_cos(normals) # n_rays 3 3
        # cov_l = self.get_cov() # n_asg 3 3

        # cov = cov_c[..., None, :, :] + cov_l # n_rays n_asg 3 3
        # s, u = torch.linalg.eigh(cov) # n_rays n_asg 3, n_rays n_asg 3 3
        # lambd = s[..., -1:] - s[..., 0:1] # n_rays n_asg 1
        # mu = s[..., -2:-1] - s[..., 0:1] # n_rays n_asg 1

        # z_l = quaternion_to_matrix(self.get_attr("rotation"))[..., -1] # n_asg 3
        # z_c = normals # n_rays 3
        # z = u[..., 0] # n_rays n_asg 3
        # smooth_l = torch.maximum(torch.sum(z * z_l, dim=-1, keepdim=True), torch.tensor(0.0, device=self.device)) # n_rays n_asg 1
        # smooth_c = torch.maximum(torch.sum(z * z_c[..., None, :], dim=-1, keepdim=True), torch.tensor(0.0, device=self.device)) # n_rays n_asg 1
        # sharp = torch.sqrt(lambd * mu) # n_rays n_asg 1
        # # nu = lambd - mu # n_rays n_asg 1
        # # integral = torch.pi / sharp - (torch.exp(-mu) / (2 * lambd)) * (bessel_fn(nu) + nu / mu * bessel_fn(nu + nu / mu)) # n_rays n_asg 1
        # integral = torch.pi / sharp
        # integral = smooth_l * smooth_c * integral # n_rays n_asg 1

        # color_c = torch.ones((*normals.shape[:-1], 3), device=self.device) * 1.170 # n_rays C
        # color_l = self.get_attr("amplitude") # n_asg C
        # color = (color_c[..., None, :] * color_l) # n_rays n_asg C

        scale = self.get_attr("scale") # n_asg 2
        sharp = torch.sqrt(scale[:, 0:1] * scale[:, 1:2]) # n_asg 1
        integral = torch.pi / (sharp + EPS) # n_asg 1
        z = self.get_lobe_axis() # n_asg 3
        smooth = torch.maximum(torch.sum(z * normals[..., None, :], dim=-1, keepdim=True), torch.tensor(0.0, device=self.device)) # n_rays n_asg 1
        color = self.get_attr("amplitude") # n_asg C

        tangent = self.get_attr("tangent").detach() # n_asg 1
        sigma = self.get_sigma(sharp, thetas, z, xyz, tangent) # n_rays n_asg 1
        irradiance = sigma * color * smooth * integral
        if self.hemisph_dir is not None:
            irradiance = (torch.sum(z * self.hemisph_dir, dim=-1, keepdim=True) > 0) * irradiance
        irradiance = torch.sum(irradiance, dim=-2)

        return irradiance

def create_gaussians(
    cfg: Config,
    hemisph_dir: Optional[Tensor] = None,
    is_eval: bool = False
    ) -> SphericalGaussians:

    if not cfg.gaussian.use_hemisphere:
        hemisph_dir = None
    else:
        assert hemisph_dir is not None, "hemisph_dir must be provided when use_hemisphere is True"

    if cfg.gaussian.anistropic:
        return AniSphericalGaussians(cfg, hemisph_dir, is_eval)
    else:
        return SphericalGaussians(cfg, hemisph_dir, is_eval)
