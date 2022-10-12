#%%
import numpy as np
import matplotlib.pyplot as plt

l = 14
h_max = 2.7
ndata = 150000
data = np.load(
    f"data/correlation_1nn/train_1nn_correlation_map_h_{h_max}_n_{ndata}_l_{l}_pbc_j_1.0.npz"
)

xx = data["correlation"]
z = data["density"]


# we save it as a C_i i+j term
r = np.arange(z.shape[-1])
# j=np.arange(z.shape[-1])
for j in range(1, int(z.shape[-1] / 2)):

    m = xx[:, r, (r + j) % z.shape[-1]]
    if j == 1:
        ms = m.reshape(-1, 1, z.shape[-1])
    else:
        ms = np.append(ms, m.reshape(-1, 1, z.shape[-1]), axis=-2)

plt.imshow(xx[0])
plt.colorbar()
plt.show()
for i in range(int(z.shape[-1] / 2) - 1):
    plt.plot(ms[0, i, :])
    plt.show()
#%%
print(ms.shape)
np.savez(
    f"data/correlation_1nn_rebuilt/test_1nn_correlation_map_h_{h_max}_{ndata}_l_{l}_pbc_j_1.0.npz",
    density=z,
    correlation=ms,
)
# %% create a mixed dataset

import numpy as np
import matplotlib.pyplot as plt

l = 15
hs_max = [1.0, 2.7, 4.5]
ndata = 150000

for h_max in hs_max:
    data = np.load(
        f"data/correlation_1nn_rebuilt/train_1nn_correlation_map_h_{h_max}_{ndata}_l_{l}_pbc_j_1.0.npz"
    )

    xx = data["correlation"]
    z = data["density"]

    if h_max == 1.0:
        xxs = xx
        zs = z
    else:
        xxs = np.append(xxs, xx, axis=0)
        zs = np.append(zs, z, axis=0)


r = np.random.permutation(zs.shape[0])
zs = zs[r]
xxs = xxs[r]

np.savez(
    f"data/correlation_1nn_rebuilt/train_1nn_correlation_map_h_mixed_n_{ndata}_l_{l}_pbc_j_1.0.npz",
    density=zs,
    correlation=xxs,
)

# %%
