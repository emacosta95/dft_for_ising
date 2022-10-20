# %%
import matplotlib.pyplot as plt
import numpy as np

l = 16
h_max = 1.0
ndata = 1000
data = np.load(
    f"data/correlation_1nn/test_1nn_correlation_map_h_{h_max}_n_{ndata}_l_{l}_pbc_j_1.0.npz"
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
# for i in range(int(z.shape[-1] / 2) - 1):
#     plt.plot(ms[0, i, :])
#     plt.show()
# %%
print(ms.shape)
np.savez(
    f"data/correlation_1nn_rebuilt/test_1nn_correlation_map_h_{h_max}_{ndata}_l_{l}_pbc_j_1.0.npz",
    density=z,
    correlation=ms,
)

# %%
