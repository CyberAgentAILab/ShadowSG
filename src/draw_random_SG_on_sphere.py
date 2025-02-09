import matplotlib.pyplot as plt
import numpy as np

n_sgs = 6
sg_vals = []
    
for sg_idx in range(n_sgs):
    lobe_x = np.random.rand() * 2 - 1
    lobe_y = -np.random.rand()
    lobe_z = np.random.rand() * 2 - 1
    lobe_axis = np.array([lobe_x, lobe_y, lobe_z])
    lobe_axis /= np.linalg.norm(lobe_axis, keepdims=True)
    lobe_amplitude = np.random.rand()
    lobe_sharpness = np.random.rand() * 5
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
    # normalize the normal
    sphere_normal /= np.linalg.norm(sphere_normal, axis=-1, keepdims=True)
    nan_mask = np.isnan(sphere_normal).any(axis=-1)

    # calculate the SG value
    lobe_value = np.exp(lobe_sharpness * (np.dot(sphere_normal, lobe_axis)-1))
    sg_val = lobe_amplitude * lobe_value
    # sg_mean = np.mean(sg_val[~nan_mask])
    # sg_std = np.std(sg_val[~nan_mask])
    # sg_max = max(sg_max, sg_mean + 2 * sg_std)
    sg_vals.append(sg_val)

fnames = ["original.png", "alter_lobe_axis.png", "alter_sharpness.png", "alter_amplitude.png"]

for i, sg_val in enumerate(sg_vals):

    # plot the SG map
    plt.imshow(sg_val, cmap='inferno')
    # set no boundary and no axis
    plt.axis('off')
    # save the figure with transparent background
    plt.savefig("random_sg_sph_" + str(i) + ".png", transparent=True, bbox_inches='tight', pad_inches=0)
