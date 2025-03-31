from pathlib import Path
from typing import List, Tuple

import numpy as np
import pyembree
import torch
import trimesh
from jaxtyping import Bool, Float
from torch import Tensor
from tqdm import tqdm
from trimesh import Trimesh

from shadow_sg import EPS
from shadow_sg.camera import Camera, OrthographicCamera, PerspectiveCamera
from shadow_sg.config import Config
from shadow_sg.io import load_img, load_json
from shadow_sg.utils import rotation_between_vectors, transform_se3


class Scene:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.device = self.cfg.base.device

        scene_data = load_json(self.cfg.path_to_scene_data)
        self.camera = self.load_camera(scene_data["camera"])

        self.plane_xyz = torch.tensor(scene_data["plane"]["xyz"], device=self.device)
        self.plane_normal = torch.tensor(scene_data["plane"]["normal"], device=self.device)
        self.plane_scale = torch.tensor(scene_data["plane"]["scale"], device=self.device)
        self.plane_diffuse = torch.tensor([scene_data["plane"]["diffuse"]], device=self.device)

        self.meshes, self.obj_is_train = self.load_meshes(scene_data["objects"])
        self.L, self.B, self.centers, self.radii = self.load_spheretree(scene_data["objects"])

        # Load diffuse
        self.obj_diffuse = []
        for obj in scene_data["objects"]:
            self.obj_diffuse.append(obj["diffuse"])
        self.obj_diffuse = torch.tensor(self.obj_diffuse, device=self.device)

        self.diffuse = torch.cat([self.plane_diffuse, self.obj_diffuse])

    @property
    def w2p(self) -> Tensor:
        w2p_R = rotation_between_vectors(self.plane_normal[None], torch.tensor([[0.0, 0.0, 1.0]], device=self.device))[0]
        w2p_t = (-w2p_R @ self.plane_xyz[..., None])[..., 0]
        w2p = torch.eye(4, device=self.device)
        w2p[:3, :3] = w2p_R
        w2p[:3, 3] = w2p_t
        return w2p

    def get_plane_diffuse(self) -> Tensor:
        return self.plane_diffuse

    def get_obj_diffuse(self, obj_indices: Tensor) -> Tensor:
        return self.diffuse_colors[obj_indices]

    def get_diffuse(self, obj_indices: Tensor) -> Tensor:
        return self.diffuse[obj_indices + 1]

    def get_spheres(self, obj_indices: list = [], level: int = -1) -> Tuple[Tensor, Tensor]:
        if len(obj_indices) == 0:
            obj_indices = torch.tensor([i for i in range(len(self.meshes)) if self.obj_is_train[i]], device=self.device)
        if level < 0:
            level = self.L + level
        n_spheres = self.B ** level
        start_idx = (1 - self.B ** level) // (1 - self.B)
        end_idx = start_idx + n_spheres
        centers = self.centers[obj_indices, start_idx:end_idx]
        radii = self.radii[obj_indices, start_idx:end_idx]
        return centers, radii
    
    @torch.no_grad()
    def get_sphere_normals(self, p: Float[Tensor, "* 3"], obj_indices: list = [], level: int = -1) -> Float[Tensor, "* 3"]:
        centers, radii = self.get_spheres(obj_indices=obj_indices, level=level)
        centers = centers.view(-1, 3)
        radii = radii.view(-1, 1)
        p_on_sphere = (torch.abs(torch.norm(p[..., None, :] - centers, dim=-1,
                       keepdim=True)**2 - radii**2) <= 1e-2)[..., 0]  # * n_s
        normals = 2 * (p[..., None, :] - centers)  # * n_s 3
        normals[~p_on_sphere] = 0.0
        normals = torch.sum(normals, dim=-2)  # * 3
        normals = torch.nn.functional.normalize(normals, dim=-1)
        return normals

    @torch.no_grad()
    def ray_marching_sphere(self, ray_o: Tensor, ray_d: Tensor, obj_indices: list = [], level: int = -1) -> Tuple[Tensor, Tensor, Tensor]:
        centers, radii = self.get_spheres(obj_indices=obj_indices, level=level)
        centers = centers.view(-1, 3)
        radii = radii.view(-1, 1)
        d2centers = ray_o[..., None, :] - centers  # * n_spheres 3
        dir_dot = torch.sum(ray_d[..., None, :] * d2centers, dim=-1,
                            keepdim=True)  # * n_spheres 1
        nabla = dir_dot ** 2 - \
            (torch.norm(d2centers, dim=-1, keepdim=True) ** 2 -
            radii ** 2)  # * n_spheres 1
        ray_hits = (nabla >= -EPS)[..., 0] # * n_spheres
        t_obj = torch.zeros((*ray_hits.shape, 1), device=ray_hits.device) # * n_spheres 1
        t_obj[ray_hits] = torch.minimum(
            -dir_dot[ray_hits] + torch.sqrt(nabla[ray_hits]),
            -dir_dot[ray_hits] - torch.sqrt(nabla[ray_hits]),
        )
        t_obj[~ray_hits] = float("inf")
        t_obj = torch.min(t_obj, dim=-2)[0]  # * 1
        ray_hits = torch.any(ray_hits, dim=-1)
        t_obj[~ray_hits] = 0
        ray_pts = ray_o + t_obj * ray_d # * 3
        ray_normals = self.get_sphere_normals(ray_pts, level=level)

        return ray_pts, ray_hits.long(), ray_normals

    @torch.no_grad()
    def ray_marching_plane(self, ray_o: Tensor, ray_d: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # Ray-plane intersection
        n = self.plane_normal[None, ...]  # 1 3
        p = self.plane_xyz[None, ...]  # 1 3
        plane_scale = self.plane_scale

        delta = torch.sum(ray_d * n, dim=-1)
        ray_hits = torch.abs(delta) >= EPS
        t = torch.zeros((*ray_hits.shape, 1), device=ray_hits.device)
        t[ray_hits] = torch.sum((p - ray_o[ray_hits]) * n, dim=-1, keepdim=True) / delta[ray_hits][..., None]
        ray_hits = ray_hits & (t[..., 0] > EPS)
        pts = torch.zeros_like(ray_o)
        pts[ray_hits] = ray_o[ray_hits] + t[ray_hits] * ray_d[ray_hits]
        pts_plane = transform_se3(self.w2p, pts)
        pts_on_plane = ((torch.abs(pts_plane) - plane_scale) <= EPS).all(dim=-1)  # *
        ray_hits = ray_hits & pts_on_plane
        return pts, ray_hits, n.repeat(*ray_hits.shape, 1)

    @torch.no_grad()
    def ray_marching_mesh(self, ray_o: Tensor, ray_d: Tensor, obj_indices: list = []) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        assert len(self.meshes) > 0
        o = ray_o.view(-1, 3).detach().cpu().numpy()
        d = ray_d.view(-1, 3).detach().cpu().numpy()
        ray_hits = np.ones(o.shape[0], dtype=int) * -1
        pts = np.zeros_like(o)
        normals = np.zeros_like(o)
        mesh_inds = obj_indices
        if len(mesh_inds) == 0:
            mesh_inds = [i for i in range(len(self.meshes)) if self.obj_is_train[i]]
        for mesh_idx in mesh_inds:
            mesh = self.meshes[mesh_idx]
            intersects, index_ray, index_tri = mesh.ray.intersects_location(
                ray_origins=o, ray_directions=d, multiple_hits=False)
            ray_hits[index_ray] = mesh_idx
            pts[index_ray] = intersects
            normals[index_ray] = mesh.face_normals[index_tri]
        pts = torch.tensor(pts, device=ray_o.device).view(*ray_o.shape)
        ray_hits = torch.tensor(ray_hits, device=ray_o.device).view(*ray_o.shape[:-1])
        normals = torch.tensor(normals, device=ray_o.device).view(*ray_o.shape)
        return pts, ray_hits, normals

    @torch.no_grad()
    def ray_marching(self, obj_indices: list = [], level: int = -1, use_sphere: bool = False, downscale: int = None) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        downscale = self.cfg.scene.downscale if downscale is None else downscale
        ray_o, ray_d = self.camera.generate_rays(downscale=downscale)
        if use_sphere or len(self.meshes) == 0:
            pts_obj, p_on_obj, normals_obj = self.ray_marching_sphere(ray_o, ray_d, obj_indices, level=level)
        else:
            pts_obj, p_on_obj, normals_obj = self.ray_marching_mesh(ray_o, ray_d, obj_indices)
        pts_plane, p_on_plane, normals_plane = self.ray_marching_plane(ray_o, ray_d)

        t_obj = torch.norm((pts_obj - ray_o), dim=-1, keepdim=True)
        t = torch.norm((pts_plane - ray_o), dim=-1, keepdim=True)
        p_on_obj[(t_obj >= t)[..., 0] & (p_on_obj >= 0) & p_on_plane] = -1
        ray_hits = (p_on_obj >= 0) | p_on_plane
        t[(p_on_obj >= 0) & p_on_plane] = torch.minimum(
            t_obj[(p_on_obj >= 0) & p_on_plane],
            t[(p_on_obj >= 0) & p_on_plane])
        t[(p_on_obj >= 0) & ~p_on_plane] = t_obj[(p_on_obj >= 0) & ~p_on_plane]

        ray_pts = ray_o + ray_d * t

        ray_normals = normals_plane
        ray_normals[p_on_obj >= 0] = normals_obj[p_on_obj >= 0]

        return ray_pts, ray_hits, p_on_obj, ray_normals

    @torch.no_grad()
    def get_visibility_sphere(self,
        p: Float[Tensor, "*num_points 3"],
        d: Float[Tensor, "num_dirs 3"],
        obj_indices: list = [],
        level: int = -1,
        chunk_size: int = 512,
        ) -> Float[Tensor, "*num_points num_dirs 1"]:
        p = p[..., None, :]  # *n_p 1 3
        p_flat = p.view(-1, 1, 3)
        centers, radii = self.get_spheres(obj_indices=obj_indices, level=level)
        centers = centers.view(-1, 3)
        radii = radii.view(-1, 1)

        # Ray-sphere intersection
        occlude_list = []
        for idx in range(0, p_flat.shape[0], chunk_size):
            d2centers = p_flat[idx:idx+chunk_size] - centers  # n_p n_s 3
            d2centers = d2centers.unsqueeze(-2)  # n_p n_s 1 3
            dir_dot = torch.sum(d * d2centers, dim=-1,
                                keepdim=True)  # n_p n_s n_d 1
            nabla = dir_dot ** 2 - \
                (torch.norm(d2centers, dim=-1, keepdim=True) **
                 2 - radii.unsqueeze(-2) ** 2)  # n_p n_s n_d 1
            t_plus = -dir_dot + torch.sqrt(nabla)
            t_minus = -dir_dot - torch.sqrt(nabla)
            t = torch.minimum(t_plus, t_minus)  # n_p n_s n_d 1
            t = torch.nan_to_num(t, nan=-1.0)  # n_p n_s n_d 1
            occlude_list.append(torch.any(t > 0, dim=-3))  # n_p n_d 1
        occlude = torch.cat(occlude_list).view(*p.shape[:-2], d.shape[0], 1)

        return (~occlude).float()

    @torch.no_grad()
    def get_visibility_mesh(self,
        p: Float[Tensor, "n_pts 3"],
        d: Float[Tensor, "n_dirs 3"],
        ret_numpy: bool = False,
        mesh_idx: int = 0) -> Bool[Tensor, "n_pts n_dirs"]:
        p_numpy = np.broadcast_to(p.detach().cpu().numpy()[:, None, :], (p.shape[0], d.shape[0], 3)).reshape(-1, 3)
        d_numpy = np.broadcast_to(d.detach().cpu().numpy()[None, ...], (p.shape[0], d.shape[0], 3)).reshape(-1, 3)
        vis = np.zeros((p.shape[0]*d.shape[0]), dtype=bool)
        _, index_ray, _ = self.meshes[mesh_idx].ray.intersects_location(p_numpy, d_numpy, multiple_hits=False)
        vis[index_ray] = True
        vis = vis.reshape(p.shape[0], d.shape[0])
        vis = ~vis
        if ret_numpy:
            return vis
        vis = torch.tensor(vis, dtype=bool, device=p.device)
        return vis
    
    @torch.no_grad()
    def get_ssdf_mesh(self,
                      p: Float[Tensor, "n_pts 3"],
                      d: Float[Tensor, "n_dirs 3"],
                      obj_indices: list = []) -> Float[Tensor, "n_pts n_dirs 1"]:
        assert len(self.meshes) > 0

        if len(obj_indices) == 0:
            meshes = [self.meshes[i] for i in range(len(self.meshes)) if self.obj_is_train[i]]
        else:
            meshes = [self.meshes[idx] for idx in obj_indices]
        theta, phi = torch.meshgrid(
            [
                torch.linspace(0, 2*np.pi, self.cfg.scene.ssdf_mesh_n_thetas),
                torch.linspace(0., np.pi/2, self.cfg.scene.ssdf_mesh_n_phis)
            ], indexing="xy")
        sampled_dirs = torch.stack([torch.cos(theta) * torch.sin(phi),
                                    torch.sin(theta) * torch.sin(phi),
                                    torch.cos(phi)], dim=-1)
        sampled_dirs = sampled_dirs.view(-1, 3).to(p.device) # n_sampled 3
        sampled_vis = torch.ones((p.shape[0], sampled_dirs.shape[0]), device=p.device).bool() # n_pts n_sampled_dirs
        light_vis = torch.ones((p.shape[0], d.shape[0]), device=p.device).bool() # n_pts n_dirs
        for i in range(len(meshes)):
            sampled_vis &= self.get_visibility_mesh(p, sampled_dirs, mesh_idx=i) # n_pts n_sampled_dirs
            light_vis &= self.get_visibility_mesh(p, d, mesh_idx=i) # n_pts n_dirs
        sampled_vis = sampled_vis[..., None].repeat(1, 1, d.shape[0]) # n_pts n_sampled_dirs n_dirs
        light_vis = light_vis[:, None, :].repeat(1, sampled_dirs.shape[0], 1) # n_pts n_sampled_dirs n_dirs

        ang = torch.acos(torch.sum(sampled_dirs[:, None, ...] * d,
            keepdim=True, dim=-1).clamp(-1.0 + EPS, 1.0 - EPS)) # n_sampled_dirs ndirs 1
        ang = ang[None].repeat(p.shape[0], 1, 1, 1) # n_pts n_samp n_dirs 1
        ang = ang.masked_fill(light_vis[..., None] & sampled_vis[..., None], float("inf"))
        thetas = torch.min(ang, dim=-3)[0] # n_pts n_dirs 1
        ang = ang.masked_fill(~light_vis[..., None] & ~sampled_vis[..., None], float("-inf"))
        ang[~light_vis & sampled_vis] *= -1
        thetas[~light_vis[:, 0, :]] = torch.max(ang, dim=-3)[0][~light_vis[:, 0, :]]

        return thetas

    def get_ssdf_sphs_level(self,
                            p: Float[Tensor, "num_points 3"],
                            d: Float[Tensor, "num_dirs 3"],
                            obj_indices: list = [],
                            ) -> Float[Tensor, "num_points num_dirs 1"]:
        r"""Given a set of points and directions, computes the ssdf angle
        at each point by the spheres defined by the geometry.

        Args:
            p (Float[Tensor, "num_points 3"]): the points
            d (Float[Tensor, "num_dirs 3"]): the directions
            obj_indices (list, optional): which objects to use.
                Defaults to [].
            level (int, optional): which level of detail to use.
                Defaults to -1 (the finest level).

        Returns:
            Float[Tensor, "num_points num_dirs 1"]: the ssdf angle
                at each point in each direction.
        """
        centers, radii = self.get_spheres(obj_indices=obj_indices, level=self.cfg.scene.sphere_level)  # n_obj n_sph 3, n_obj n_sph 1
        centers = centers.view(-1, 3)
        radii = radii.view(-1, 1)
        centers_p = centers - p[..., None, :]  # n_pts n_sph 3
        centers_p_norm = torch.norm(centers_p, dim=-1, keepdim=True)  # n_pts n_sph 1
        centers_p = centers_p / centers_p_norm  # n_pts n_sph 3
        centers_p_dot_d = torch.sum(centers_p[..., None, :] * d, dim=-1, keepdim=True)  # n_pts n_sph n_dirs 1
        ang_centers_p_d = torch.acos(centers_p_dot_d.clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_sph n_dirs 1
        ang_centers_p_tangent = torch.asin((radii[None] / centers_p_norm).clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_sph 1
        ang = ang_centers_p_d - ang_centers_p_tangent[..., None, :]  # n_pts n_sph n_dirs 1
        minus = torch.any(ang < 0, dim=-3)  # n_pts n_dirs 1
        thetas = torch.min(torch.abs(ang), dim=-3)[0]  # n_pts n_dirs 1
        ang_minus = ang.masked_fill(ang >= 0, float("-inf"))
        thetas_minus = torch.max(ang_minus, dim=-3)[0]  # n_pts n_dirs 1
        thetas[minus] = thetas_minus[minus]
        return thetas

    def get_ssdf_sphs_level_min(self,
        p: Float[Tensor, "num_points 3"],
        d: Float[Tensor, "num_dirs 3"],
        obj_indices: list = [],
        ) -> Float[Tensor, "num_points num_dirs 1"]:
        
        centers, radii = self.get_spheres(obj_indices=obj_indices, level=self.cfg.scene.sphere_level)  # n_obj n_sph 3, n_obj n_sph 1
        centers = centers.view(-1, 3)
        radii = radii.view(-1, 1)
        centers_p = centers - p[..., None, :]  # n_pts n_sph 3
        centers_p_norm = torch.norm(centers_p, dim=-1, keepdim=True)  # n_pts n_sph 1
        centers_p = centers_p / centers_p_norm  # n_pts n_sph 3
        centers_p_dot_d = torch.sum(centers_p[..., None, :] * d, dim=-1, keepdim=True)  # n_pts n_sph n_dirs 1
        ang_centers_p_d = torch.acos(centers_p_dot_d.clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_sph n_dirs 1
        ang_centers_p_tangent = torch.asin((radii[None] / centers_p_norm).clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_sph 1
        ang = ang_centers_p_d - ang_centers_p_tangent[..., None, :]  # n_pts n_sph n_dirs 1
        thetas = torch.min(ang, dim=-3)[0]
        return thetas

    def get_ssdf_sphs_coarse2fine(self,
        p: Float[Tensor, "num_points 3"],
        d: Float[Tensor, "num_dirs 3"],
        obj_indices: list = [],
    ) -> Float[Tensor, "num_points num_dirs 1"]:
        r"""Given a set of points and directions, computes the ssdf angle
        at each point by the spheres defined by the geometry.
        This function expands spheres from the root to the leaves level by level.
        At each level, expand k octaves.

        Args:
            p (Float[Tensor, "num_points 3"]): the points
            d (Float[Tensor, "num_dirs 3"]): the directions
            obj_indices (list, optional): which objects to use.
                Defaults to [], which means use all.
            k (int, optional): the number of octaves to expand. Defaults to 3.

        Returns:
            Float[Tensor, "num_points num_dirs 1"]: the ssdf angle at each point
                in each direction.
        """
        k = self.cfg.scene.ssdf_sph_coarse2fine_k
        if len(obj_indices) == 0:
            obj_indices = torch.tensor([i for i in range(len(self.meshes)) if self.obj_is_train[i]], device=self.device)
        else:
            obj_indices = torch.tensor(obj_indices, device=self.device)  # n_obj
        centers_all = self.centers[obj_indices].view(*([1]*len(p.shape[:-1])), 1, -1, 1, 3).broadcast_to(*p.shape[:-1], len(obj_indices), -1, d.shape[0], 3)  # n_pts n_obj n_sph n_dirs 3
        radii_all = self.radii[obj_indices].view(*([1]*len(p.shape[:-1])), 1, -1, 1, 1).broadcast_to(*p.shape[:-1], len(obj_indices), -1, d.shape[0], 1)  # n_pts n_obj n_sph n_dirs 1
        inds = torch.arange(1, self.B+1, device=self.device).view(*([1]*len(p.shape[:-1])), 1, self.B, 1, 1).broadcast_to(*p.shape[:-1], len(obj_indices), self.B, d.shape[0], 1)  # n_pts n_obj B n_dirs 1
        for _ in range(1, self.L):
            centers = torch.gather(centers_all, dim=-3, index=inds.repeat(*([1]*len(p.shape[:-1])), 1, 1, 1, 3))  # n_pts n_obj n_sph n_dirs 3
            radii = torch.gather(radii_all, dim=-3, index=inds)  # n_pts n_obj n_sph n_dirs 1
            centers_p = centers - p[..., None, None, None, :]  # n_pts n_obj n_sph n_dirs 3
            centers_p_norm = torch.norm(centers_p, dim=-1, keepdim=True)  # n_pts n_obj n_sph n_dirs 1
            centers_p = centers_p / centers_p_norm
            centers_p_dot_d = torch.sum(centers_p * d, dim=-1, keepdim=True)  # n_pts n_obj n_sph n_dirs 1
            ang_centers_p_d = torch.acos(centers_p_dot_d.clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_obj n_sph n_dirs 1
            ang_centers_p_tangent = torch.asin((radii / centers_p_norm).clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_obj n_sph n_dirs 1
            ang = ang_centers_p_d - ang_centers_p_tangent  # n_pts n_obj n_sph n_dirs 1
            minus = torch.any(ang < 0, dim=-3, keepdim=True).broadcast_to(*p.shape[:-1], len(obj_indices), k, d.shape[0], 1)  # n_pts n_obj k n_dirs 1
            inds_k = torch.argsort(ang, dim=-3)[..., :k, :, :]  # n_pts n_obj k n_dirs 1
            ang_minus = ang.masked_fill(ang >= 0, float("-inf"))
            inds_minus_k = torch.argsort(ang_minus, dim=-3, descending=True)[..., :k, :, :]  # n_pts n_obj k n_dirs 1
            inds_k[minus] = inds_minus_k[minus]
            inds = torch.gather(inds, dim=-3, index=inds_k)  # n_pts n_obj k n_dirs 1
            inds = self.B * inds + 1
            inds = inds[..., None, :, :] + torch.arange(0, self.B, device=self.device).view(*([1]*len(p.shape[:-1])), 1, 1, self.B, 1, 1)
            inds = inds.view(*inds.shape[:-4], -1, d.shape[0], 1)
        thetas = torch.gather(ang, dim=-3, index=inds_k[..., 0:1, :, :])  # n_pts n_obj 1 n_dirs 1
        thetas = torch.min(thetas[..., 0, :, :], dim=-3)[0] # n_pts n_dirs 1
        minus = torch.any(minus[..., 0, :, :], dim=-3)  # n_pts n_dirs 1
        thetas_minus = torch.gather(ang_minus, dim=-3, index=inds_minus_k[..., 0:1, :, :])  # n_pts n_obj 1 n_dirs 1
        thetas[minus] = torch.max(thetas_minus[..., 0, :, :], dim=-3)[0][minus]

        return thetas

    def get_ssdf_sphs_coarse2fine_min(self,
        p: Float[Tensor, "num_points 3"],
        d: Float[Tensor, "num_dirs 3"],
        obj_indices: list = [],
    ) -> Float[Tensor, "num_points num_dirs 1"]:
        r"""Given a set of points and directions, computes the ssdf angle
        at each point by the spheres defined by the geometry.
        This function expands spheres from the root to the leaves level by level.
        At each level, expand k octaves.

        Args:
            p (Float[Tensor, "num_points 3"]): the points
            d (Float[Tensor, "num_dirs 3"]): the directions
            obj_indices (list, optional): which objects to use.
                Defaults to [], which means use all.
            k (int, optional): the number of octaves to expand. Defaults to 3.

        Returns:
            Float[Tensor, "num_points num_dirs 1"]: the ssdf angle at each point
                in each direction.
        """
        k = self.cfg.scene.ssdf_sph_coarse2fine_k
        if len(obj_indices) == 0:
            obj_indices = torch.tensor([i for i in range(len(self.meshes)) if self.obj_is_train[i]], device=self.device)
        else:
            obj_indices = torch.tensor(obj_indices, device=self.device)  # n_obj
        centers_all = self.centers[obj_indices].view(*([1]*len(p.shape[:-1])), 1, -1, 1, 3).broadcast_to(*p.shape[:-1], len(obj_indices), -1, d.shape[0], 3)  # n_pts n_obj n_sph n_dirs 3
        radii_all = self.radii[obj_indices].view(*([1]*len(p.shape[:-1])), 1, -1, 1, 1).broadcast_to(*p.shape[:-1], len(obj_indices), -1, d.shape[0], 1)  # n_pts n_obj n_sph n_dirs 1
        inds = torch.arange(1, self.B+1, device=self.device).view(*([1]*len(p.shape[:-1])), 1, self.B, 1, 1).broadcast_to(*p.shape[:-1], len(obj_indices), self.B, d.shape[0], 1)  # n_pts n_obj B n_dirs 1
        for _ in range(1, self.L):
            centers = torch.gather(centers_all, dim=-3, index=inds.repeat(*([1]*len(p.shape[:-1])), 1, 1, 1, 3))  # n_pts n_obj n_sph n_dirs 3
            radii = torch.gather(radii_all, dim=-3, index=inds)  # n_pts n_obj n_sph n_dirs 1
            centers_p = centers - p[..., None, None, None, :]  # n_pts n_obj n_sph n_dirs 3
            centers_p_norm = torch.norm(centers_p, dim=-1, keepdim=True)  # n_pts n_obj n_sph n_dirs 1
            centers_p = centers_p / centers_p_norm
            centers_p_dot_d = torch.sum(centers_p * d, dim=-1, keepdim=True)  # n_pts n_obj n_sph n_dirs 1
            ang_centers_p_d = torch.acos(centers_p_dot_d.clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_obj n_sph n_dirs 1
            ang_centers_p_tangent = torch.asin((radii / centers_p_norm).clamp(-1.0 + EPS, 1.0 - EPS))  # n_pts n_obj n_sph n_dirs 1
            ang = ang_centers_p_d - ang_centers_p_tangent  # n_pts n_obj n_sph n_dirs 1
            inds_k = torch.argsort(ang, dim=-3)[..., :k, :, :]  # n_pts n_obj k n_dirs 1
            inds = torch.gather(inds, dim=-3, index=inds_k)  # n_pts n_obj k n_dirs 1
            inds = self.B * inds + 1
            inds = inds[..., None, :, :] + torch.arange(0, self.B, device=self.device).view(*([1]*len(p.shape[:-1])), 1, 1, self.B, 1, 1)
            inds = inds.view(*inds.shape[:-4], -1, d.shape[0], 1)
        thetas = torch.gather(ang, dim=-3, index=inds_k[..., 0:1, :, :])  # n_pts n_obj 1 n_dirs 1
        thetas = torch.min(thetas[..., 0, :, :], dim=-3)[0] # n_pts n_dirs 1

        return thetas

    def get_ssdf(self,
        p: Float[Tensor, "num_points 3"],
        d: Float[Tensor, "num_dirs 3"],
        chunk_size: int = -1,
        obj_indices: list = [],
    ) -> Float[Tensor, "num_points num_dirs 1"]:

        methods = {
            "mesh": self.get_ssdf_mesh,
            "sph_level": self.get_ssdf_sphs_level,
            "sph_coarse2fine": self.get_ssdf_sphs_coarse2fine,
            "sph_level_min": self.get_ssdf_sphs_level_min,
            "sph_coarse2fine_min": self.get_ssdf_sphs_coarse2fine_min,
        }
        if self.cfg.scene.ssdf_method not in methods:
            raise NotImplementedError(f"Unknown ssdf method {self.cfg.scene.ssdf_method}, please select from {methods.keys()}")
        if chunk_size == -1:
            return methods[self.cfg.scene.ssdf_method](p, d, obj_indices)
        elif chunk_size > 0:
            p_flat = p.view(-1, 3)
            ssdf = []
            for i in (pbar := tqdm(range(0, p_flat.shape[0], chunk_size))):
                pbar.set_description(f"Computing SSDF with {self.cfg.scene.ssdf_method}")
                ssdf.append(methods[self.cfg.scene.ssdf_method](p_flat[i:i+chunk_size], d, obj_indices))
            ssdf = torch.cat(ssdf, dim=0)
            return ssdf.view(p.shape[:-1] + ssdf.shape[1:])
        else:
            raise ValueError(f"Invalid chunk size {chunk_size}")

    def load_spheretree(self, data: dict) -> Tuple[int, int, Tensor, Tensor]:
        centers, radii = [], []
        for obj in data:
            path_to_sph = Path(self.cfg.scene.path_to_data) / obj["path_to_sph"]
            o2w = torch.tensor(obj["o2w"], device=self.device)
            scale = o2w[0, :3].norm()
            centers.append([])
            radii.append([])
            with open(path_to_sph) as fp:
                L, B = (int(x) for x in fp.readline().rstrip().split(" "))
                n_spheres = (B ** L - 1) // (B - 1)
                lines = fp.readlines()[:n_spheres]
            for line in lines:
                line = [float(x) for x in line.rstrip().split(" ")]
                radius = 0.0 if line[3] <= EPS else line[3]
                centers[-1].append(torch.tensor([line[:3]], device=self.device))
                radii[-1].append(torch.tensor([[radius]], device=self.device))
            centers[-1] = transform_se3(o2w, torch.cat(centers[-1]))
            radii[-1] = scale * torch.cat(radii[-1])
        centers = torch.stack(centers)
        radii = torch.stack(radii)
        return L, B, centers, radii

    def load_meshes(self, data: dict) -> List[Trimesh]:
        meshes = []
        obj_is_train = []
        for obj in data:
            if obj["path_to_mesh"] is not None:
                path_to_mesh = Path(self.cfg.scene.path_to_data) / obj["path_to_mesh"]
                mesh = trimesh.load_mesh(path_to_mesh)
                o2w = np.array(obj["o2w"])
                mesh.apply_transform(o2w)
                meshes.append(mesh)
            obj_is_train.append(obj["is_train"])
        return meshes, obj_is_train

    def load_camera(self, data: dict) -> Camera:
        img_gt = None
        if self.cfg.path_to_img is not None:
            img_gt = load_img(self.cfg.path_to_img, self.device)
        if data["proj_type"] == "perspective":
            camera = PerspectiveCamera(
                width=data["width"],
                height=data["height"],
                c2w=torch.tensor(data["c2w"], device=self.device),
                img=img_gt,
                fx=data["fx"],
                fy=data["fy"],
                cx=data["cx"],
                cy=data["cy"],
                k1=data["k1"] if "k1" in data else 0.0,
                k2=data["k2"] if "k2" in data else 0.0,
                p1=data["p1"] if "p1" in data else 0.0,
                p2=data["p2"] if "p2" in data else 0.0,
                k3=data["k3"] if "k3" in data else 0.0,
            )
        elif data["proj_type"] == "orthographic":
            camera = OrthographicCamera(
                width=data["width"],
                height=data["height"],
                c2w=torch.tensor(data["c2w"], device=self.device),
                img=img_gt,
                znear=data["znear"],
                zfar=data["zfar"],
                max_y=data["max_y"],
                min_y=data["min_y"],
                max_x=data["max_x"],
                min_x=data["min_x"],
                scale=data["scale"],
            )
        else:
            raise ValueError(f"Unknown camera type {data['proj_type']}")
        return camera
