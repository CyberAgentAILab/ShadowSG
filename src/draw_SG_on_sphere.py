from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R

path_to_sgs = []
sg_vals = []
sg_max = 0
    
for path_to_sg in path_to_sgs:
    print(path_to_sg)
    sg = np.loadtxt(path_to_sg)
    lobe_x, lobe_y, lobe_z = sg[:3]
    lobe_axis = np.array([lobe_x, lobe_y, lobe_z])
    lobe_axis /= np.linalg.norm(lobe_axis, keepdims=True)
    lobe_amplitude = sg[4]
    lobe_sharpness = np.exp(-sg[3])
    print(lobe_axis, lobe_amplitude, lobe_sharpness)

    img_size = 1024
    # lobe_axis = np.array([1., 1., 1.])
    # lobe_amplitude = 0.5
    # lobe_sharpness = 3

    # create a sphere
    xx, zz = np.meshgrid(np.linspace(-1, 1, img_size), np.linspace(-1, 1, img_size))
    # revert yy to make the sphere right-handed
    zz = -zz
    yy = -np.sqrt(1 - xx**2 - zz**2)
    sphere_normal = np.stack([xx, yy, zz], axis=-1)
    r = R.from_rotvec(np.radians(-45) * np.array([0, 0, 1]))
    sphere_normal = r.apply(sphere_normal.reshape(-1, 3)).reshape(img_size, img_size, 3)
    # normalize the normal
    sphere_normal /= np.linalg.norm(sphere_normal, axis=-1, keepdims=True)
    nan_mask = np.isnan(sphere_normal).any(axis=-1)

    # calculate the SG value
    lobe_value = np.exp(lobe_sharpness * (np.dot(sphere_normal, lobe_axis)-1))
    sg_val = lobe_amplitude * lobe_value
    sg_mean = np.mean(sg_val[~nan_mask])
    sg_std = np.std(sg_val[~nan_mask])
    sg_max = max(sg_max, sg_mean + 2 * sg_std)
    sg_vals.append(sg_val)

fnames = ["original.png", "alter_lobe_axis.png", "alter_sharpness.png", "alter_amplitude.png"]

for i, sg_val in enumerate(sg_vals):

    # plot the SG map
    plt.imshow(sg_val, cmap='inferno', vmin=0, vmax=sg_max)
    # set no boundary and no axis
    plt.axis('off')
    # save the figure with transparent background
    plt.savefig(Path(path_to_sgs[i]).parent / Path(path_to_sgs[i]).stem / "envsph.png", transparent=True, bbox_inches='tight', pad_inches=0)

    envmap = Image.open(Path(path_to_sgs[i]).parent / Path(path_to_sgs[i]).stem / "envsph.png")
    img = Image.open(Path(path_to_sgs[i]).parent / Path(path_to_sgs[i]).stem / "eval/iter_020000/image.png")
    envmap_size = img.width // 4
    envmap = envmap.resize((envmap_size, envmap_size))
    img.paste(envmap, (0, img.height - envmap_size), envmap)
    img.save(Path(path_to_sgs[i]).parent / fnames[i])
