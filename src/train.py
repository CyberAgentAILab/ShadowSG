import os
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf

from shadow_sg.config import Config
from shadow_sg.gaussians import create_gaussians
from shadow_sg.scene import Scene
from shadow_sg.trainer import GaussiansTrainer, LSTSQTrainer


@hydra.main(version_base=None, config_path="../configs", config_name="default")
def main(cfg: DictConfig) -> None:
    cfg = Config(**cfg)
    os.makedirs(cfg.base.path_to_output, exist_ok=True)
    for path_to_event_file in Path(cfg.base.path_to_output).glob("events.*"):
        os.remove(path_to_event_file)
    OmegaConf.save(cfg, Path(cfg.base.path_to_output) / "config.yaml")  
    scene = Scene(cfg)
    if cfg.base.method == "gaussians":
        gaussians = create_gaussians(cfg, scene.plane_normal)
        trainer = GaussiansTrainer(cfg, gaussians)
        trainer.train(scene)
    elif cfg.base.method == "fit_envmap":
        gaussians = create_gaussians(cfg, scene.plane_normal)
        trainer = GaussiansTrainer(cfg, gaussians)
        trainer.fit_envmap(scene)
    elif cfg.base.method == "lstsq":
        trainer = LSTSQTrainer(cfg)
        trainer.train(scene)
    else:
        raise ValueError(f"Unknown method: {cfg.method}")
    

if __name__ == "__main__":
    main()
