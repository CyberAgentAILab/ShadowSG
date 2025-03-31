from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Union

from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf


@dataclass
class SceneConfig:
    path_to_data: str = ""
    sphere_level: int = -1
    downscale: int = 1
    downscale_eval: int = 1
    diffuse: list = field(default_factory=lambda: [1.0, 1.0, 1.0])
    ssdf_method: str = "sph_coarse2fine_min" # ["mesh", "sph_level", "sph_coarse2fine", "sph_level_min", "sph_coarse2fine_min"]
    ssdf_mesh_n_thetas: int = 36
    ssdf_mesh_n_phis: int = 9
    ssdf_sph_coarse2fine_k: int = 3 

@dataclass
class GaussianConfig:
    anistropic: bool = False
    envmap_height: int = 512
    n_gaussians: int = 256
    path_to_ckpt: str = ""
    path_to_sigma_ckpt: str = "./data/k_coeffs_wang.txt"
    prune_by_amplitude_grad_threshold: float = 0.001
    prune_by_amplitude_threshold: float = 0.005
    render_chunk_size: int = 1024
    sigma_fn_type: str = "k" # ["k", "mlp", "mlp_ssdf"]
    optimizable_k: bool = False
    use_hemisphere: bool = True

    lobe_axis_activation_fn: str = "normalize"
    sharpness_activation_fn: str = "exp"
    amplitude_activation_fn: str = "abs"

@dataclass
class TrainerConfig:
    n_iters: int = 20000

    lr_sg_lobe_axis: float = 1e-3
    lr_sg_sharpness: float = 1e-3
    lr_sg_amplitude: float = 1e-3
    lr_asg_rotation: float = 1e-3
    lr_asg_scale: float = 1e-3
    lr_asg_amplitude: float = 1e-3
    lr_sigma_mlp: float = 1e-3
    lr_k_coeffs: float = 1e-4
    lr_scheduler_step: int = 5000
    lr_scheduler_gamma: float = 0.9

    batch_size: int = 1024
    patch_based_sampling: bool = False

    log_every_n: int = 50
    log_grads: bool = False
    log_sigma: bool = False
    eval_every_n: int = 5000

    adapt_control_from_iter: int = 500
    adapt_control_every_iter: int = 500
    adapt_control_to_iter: int = 15000

    reg_laplacian: float = 0.0
    reg_lstsq_1: float = 0.1
    reg_lstsq_2: float = 0.1

@dataclass
class BaseConfig:
    path_to_output: str = ""
    device: str = "cuda"
    method: str = "gaussians" # ["gaussians", "fit_envmap", "lstsq"]

@dataclass
class Config:
    base: BaseConfig = field(default_factory=BaseConfig)
    scene: SceneConfig = field(default_factory=SceneConfig)
    gaussian: GaussianConfig = field(default_factory=GaussianConfig)
    trainer: TrainerConfig = field(default_factory=TrainerConfig)

    def __post_init__(self):
        if isinstance(self.base, dict):
            self.base = BaseConfig(**self.base)
        if isinstance(self.scene, dict):
            self.scene = SceneConfig(**self.scene)
        if isinstance(self.gaussian, dict):
            self.gaussian = GaussianConfig(**self.gaussian)
        if isinstance(self.trainer, dict):
            self.trainer = TrainerConfig(**self.trainer)

        exts = ["exr", "png", "jpg", "jpeg"]
        for ext in exts:
            for path in Path(self.scene.path_to_data).glob(f"image.{ext}"):
                self.path_to_img = path

    @property
    def path_to_env(self) -> Union[Path, None]:
        exts = ["exr", "png", "jpg", "jpeg"]
        for ext in exts:
            for path in Path(self.scene.path_to_data).glob(f"envmap.{ext}"):
                return path
        return None

    @property
    def path_to_mask_obj(self) -> Union[Path, None]:
        path = Path(self.scene.path_to_data) / f"mask_obj.png"
        if path.exists():
            return path
        return None

    @property
    def path_to_mask_plane(self) -> Union[Path, None]:
        path = Path(self.scene.path_to_data) / f"mask_plane.png"
        if path.exists():
            return path
        return None

    @property
    def path_to_scene_data(self) -> Path:
        return Path(self.scene.path_to_data) / "scene.json"

if __name__ == "__main__":
    cfg = Config()
    OmegaConf.save(cfg.base, "configs/base/default.yaml")
    OmegaConf.save(cfg.scene, "configs/scene/default.yaml")
    OmegaConf.save(cfg.gaussian, "configs/gaussian/default.yaml")
    OmegaConf.save(cfg.trainer, "configs/trainer/default.yaml")
    OmegaConf.save({
        "defaults": [
            {"base": "default"},
            {"scene": "default"},
            {"gaussian": "default"},
            {"trainer": "default"},
            "_self_",
            {"override hydra/hydra_logging": "disabled"},
            {"override hydra/job_logging": "disabled"},
        ],
        "hydra": {"output_subdir": None, "run": {"dir": "."}},
    }, "configs/default.yaml")
