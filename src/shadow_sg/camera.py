from abc import abstractmethod
from dataclasses import dataclass
from typing import Tuple

import torch
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms

from shadow_sg.utils import transform_se3


@dataclass
class Camera:
    width: int
    height: int
    c2w: Float[Tensor, "4 4"]
    img: Float[Tensor, "H W 3"]

    @abstractmethod
    def generate_rays(self, downscale: int = 1) -> Tuple[Float[Tensor, "H W 3"], Float[Tensor, "H W 3"]]:
        pass

    def get_wh(self, downscale: int = 1) -> Tuple[int, int]:
        return self.width // downscale, self.height // downscale

    def get_uvs(self, in_ndc: bool = False, z_plane: float = 1.0, downscale: int = 1) -> Float[Tensor, "H W 3"]:
        w, h = self.get_wh(downscale)
        u, v = torch.meshgrid([torch.arange(w), torch.arange(h)], indexing="xy")
        u = u.to(self.c2w)
        v = v.to(self.c2w)
        if in_ndc:
            u = torch.flip(u / w * 2 - 1, dims=(1,))
            v = torch.flip(v / h * 2 - 1, dims=(0,))

        z = torch.ones_like(u) * z_plane
        uvs = torch.stack([u, v, z], dim=-1) # H W 3
        return uvs

    def get_img(self, downscale: int = 1) -> Tensor:
        img = self.img.clone()
        if downscale > 1:
            w, h = self.get_wh(downscale)
            transform = transforms.Resize((h, w))
            img = transform(img.permute(2, 0, 1)).permute(1, 2, 0)
        return img

@dataclass
class PerspectiveCamera(Camera):
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0
    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0

    def _get_K(self, downscale: int = 1) -> Float[Tensor, "3 3"]:
        fx, fy = self.fx / downscale, self.fy / downscale
        cx, cy = self.cx / downscale, self.cy / downscale
        K = torch.tensor([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ], device=self.c2w.device)
        return K

    def _unproj(self, uvs: Float[Tensor, "* 3"], downscale: int = 1) -> Float[Tensor, "* 3"]:
        K = self._get_K(downscale=downscale)
        xys = (K.inverse() @ uvs[..., None])[..., 0] # * 3
        k1, k2, p1, p2, k3 = self.k1, self.k2, self.p1, self.p2, self.k3
        x, y = xys[..., 0], xys[..., 1]
        r = x**2 + y**2
        x_distorted = x * (1 + k1*r + k2*r**2 + k3*r**3) + 2*p1*x*y + p2*(r + 2*x**2)
        y_distorted = y * (1 + k1*r + k2*r**2 + k3*r**3) + p1*(r + 2*y**2) + 2*p2*x*y
        unprojected_points = torch.stack([x_distorted, y_distorted,
                                          torch.ones_like(x_distorted)], dim=-1)

        return unprojected_points

    def generate_rays(self, downscale: int = 1) -> Tuple[Float[Tensor, "H W 3"], Float[Tensor, "H W 3"]]:
        uvs = self.get_uvs(downscale=downscale)
        c2w_R = self.c2w[:3, :3]
        c2w_t = self.c2w[:3, 3]
        ray_o = c2w_t.view([1,]*len(uvs.shape[:-1]) +
                           [3]).broadcast_to(*uvs.shape).clone()
        ray_d = torch.nn.functional.normalize(
            (c2w_R[None, ...] @ self._unproj(uvs, downscale=downscale)[..., None])[..., 0] + \
                c2w_t[None, ...] - ray_o, dim=-1)

        return ray_o, ray_d

@dataclass
class OrthographicCamera(Camera):
    znear: float = 0.0
    zfar: float = 0.0
    max_y: float = 0.0
    min_y: float = 0.0
    max_x: float = 0.0
    min_x: float = 0.0
    scale: float = 0.0

    def _get_proj_mat(self) -> Float[Tensor, "4 4"]:
        # https://en.wikipedia.org/wiki/Orthographic_projection
        proj_mat = torch.tensor([
            [-2 / (self.max_x - self.min_x) * self.scale, 0, 0, (self.max_x + self.min_x) / (self.max_x - self.min_x)],
            [0, -2 / (self.max_y - self.min_y) * self.scale, 0, (self.max_y + self.min_y) / (self.max_y - self.min_y)],
            [0, 0, -2 / (self.zfar - self.znear) * self.scale, (self.zfar + self.znear) / (self.zfar - self.znear)],
            [0, 0, 0, 1.0]
        ], device=self.c2w.device)
        return proj_mat

    def _unproj(self, uvs: Float[Tensor, "* 3"]) -> Float[Tensor, "* 3"]:
        unproj_mat = self._get_proj_mat().inverse()
        xyz = transform_se3(unproj_mat, uvs)
        return xyz

    def generate_rays(self, downscale=1) -> Tuple[Float[Tensor, "3"], Float[Tensor, "H W 3"]]:
        uvs = self.get_uvs(in_ndc=True, z_plane=0.0, downscale=downscale)
        ray_o = self._unproj(uvs)

        ray_o = transform_se3(self.c2w, ray_o)
        ray_d = torch.zeros_like(ray_o)
        ray_d[..., 2] = 1.0
        ray_d = (self.c2w[:3, :3] @ ray_d[..., None])[..., 0]

        return ray_o, ray_d

if __name__ == "__main__":
    cam = OrthographicCamera(
        width=651,
        height=835,
        c2w=torch.tensor([
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0, 100.0],
            [0.0, 0.0, 0.0, 1.0]
        ], device="cuda"),
        img=torch.tensor([], device="cuda"),
        znear=-100.0,
        zfar=100.0,
        max_y=100.0,
        min_y=-100.0,
        max_x=100.0,
        min_x=-100.0,
        scale=100.0
    )

    ray_o, ray_d = cam.generate_rays()
