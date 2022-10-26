#%%
import numpy as np

h_max = [3.60, 7.20]


for h in h_max:

    data = np.load(
        f"data/dataset_2nn/train_unet_periodic_2nn_16_l_{h:.2f}_h_15000_n.npz"
    )

    z = data["density"]
    xx = data["density_F"]

    z = np.append(z, -1 * z, axis=0)
    xx = np.append(xx, xx, axis=0)

    p = np.random.permutation(z.shape[0])

    np.savez(
        f"data/dataset_2nn/train_unet_periodic_2nn_augmentation_16_l_{h:.2f}_h_30000_n.npz",
        density=z[p],
        density_F=xx[p],
    )

# %%
