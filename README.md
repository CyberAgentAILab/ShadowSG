# ShadowSG

This is the repository for the 3DV2025 paper:

Hanwei Zhang, Xu Cao, Hiroshi Kawasaki, and Takafumi Taketomi, "*ShadowSG: Spherical Gaussian Illumination from Shadows,*" 2025 International Conference on 3D Vision (3DV), 2025.

## Clone the Repository

```bash
git clone --recursive https://github.com/CyberAgentAILab/ShadowSG.git
```

## Install Dependencies

1. pytorch==2.3.1, torchvision==0.18.1, pytorch3d.

    ```bash
    pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu118
    pip install "git+https://github.com/facebookresearch/pytorch3d.git"
    ```

    > Note: You can specify your own cuda version by changing the index url link.

2. Other packages

    ```bash
    pip install -r requirements.txt
    ```

## Data Preparation

We use the following folder structure for a specific scene containing a single image.
A minimal scene should include an image file, a `scene.json` file to define the geometry, the mesh of the shadow caster object, and the sphere tree approximation of the mesh.

```
data
│   image[.exr][.png]
│   scene.json
│
└───geometry
       mesh.obj
       mesh_sphere.sph
```

The synthetic and real-world data used in the paper can be found via [Google Drive](https://drive.google.com/drive/folders/1cn9Y0F1Chib4OQaB4lcypIfsmr0dRgG9?usp=sharing).
We provide a script to automatically download our data in `scripts/download_data.sh`.
Simply running the following command and the data will be downloaded to `./data`.

```bash
sh ./data/download_data.sh
```

### Generate sphere trees for custom models

We provide a compiled binary `./src/makeTreeMedial` of [spheretree](https://github.com/r0bertr/spheretree).
This binary is tested on Ubuntu 22.04.

Run the following command to generate a sphere-tree approximation for an custom `.obj` file.
Note that the model must be watertight.

```bash
chmod +x ./src/makeTreeMedial
python ./src/obj2sph.py --path-to-obj ./data/synthetic/bunny_point1/geometry/mesh.obj
```

Replace `--path-to-obj` by any custom watertight `.obj` model.
A `.sph` text file will be written to the same directory as your `.obj` file.

This script executes the `makeTreeMedial` binary and you may be required to compiled the binary manually in submodule `./external/spheretree` if you use a different OS other than Ubuntu.

## Train

To run an SG optimization for a specific scene, run the following command.

```bash
python src/train.py base.path_to_output=./experiments/bunny scene.path_to_data=./data/synthetic/bunny_point1/
```

Replace `base.path_to_output` and `scene.path_to_data` by any custom output path and any scene data folder.

## Citation

Please consider citating our work using the following bibtex:

```
@inproceedings{shadowsg2025zhang,
  title={{ShadowSG: S}pherical Gaussian Illumination from Shadows},
  author={Zhang, Hanwei and Cao, Xu and Kawasaki, Hiroshi and Taketomi, Takafumi},
  booktitle= International Conference on 3D Vision (3DV),
  year={2025}
}
```
