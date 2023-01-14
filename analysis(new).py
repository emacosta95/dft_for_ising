#%% imports
from cmath import log
import os
import time
from ast import increment_lineno
from turtle import position

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from numpy.lib.mixins import _inplace_binary_method
from pyrsistent import l
from scipy import fft, ifft
from tqdm.notebook import tqdm, trange

from src.training.utils_analysis import dataloader, nuv_representability_check


# CORRELATION MAP from density to correlation (TEST)

#%% PART I: loading the data
batch = 1000
nbatch = 10
minibatch = int(batch / nbatch)
ls = [16, 32, 64, 128]
h_max = [2.7]
device = "cpu"
torch.set_num_threads(3)


for l in ls:
    op = []
    op_ml = []
    for i in range(len(h_max)):
        data = np.load(
            f"data/correlation_1nn_rebuilt/test_1nn_correlation_map_h_{h_max[i]}_1000_l_{l}_pbc_j_1.0.npz"
        )
        z = data["density"][:batch]
        xx = data["correlation"][:batch]

        if i != 0:
            z_torch = torch.tensor(z, dtype=torch.double, device=device)
            print(z.shape)
            # model=torch.load(f'model_rep/1nn_den2cor/h_{h_max}_150k_unet_periodic_den2corRESNET_[40, 40, 40, 40]_hc_5_ks_1_ps_4_nconv_0_nblock',map_location='cpu')
            model = torch.load(
                f"model_rep/1nn_den2cor/h_{h_max[i]}_150k_unet_periodic_den2corLSTM_scalable_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock",
                map_location=device,
            )

            model.eval()

            for j in trange(nbatch):
                print(i)
                x = (
                    model(z_torch[minibatch * j : minibatch * (j + 1)])
                    .cpu()
                    .detach()
                    .numpy()
                )
                if j == 0:
                    xx_ml = x
                else:
                    xx_ml = np.append(xx_ml, x, axis=0)

        op.append(np.average(xx))

        if i == 0:
            op_ml.append(np.average(xx))
        else:
            op_ml.append(np.average(xx_ml))

    dxx = np.sqrt(np.average((xx - xx_ml) ** 2) / np.average((xx) ** 2))

    # plt.plot(h_max,op,label=f'exact l={l}')
    plt.plot(h_max, op_ml, label=f"l={l}")
    if l == 128:
        plt.plot(
            h_max,
            op,
            label="exact l=128",
            linestyle="--",
            linewidth=2,
            color="black",
            marker="o",
        )
plt.legend(fontsize=20)
plt.show()

#%% PART II(a): Distribution of ln(C)
orderparameter_ml = np.average(xx_ml)
orderparameter = np.average(xx)
print(np.abs(orderparameter - orderparameter_ml) / np.abs(orderparameter))

#%%
from scipy import stats

ls = [7, 15, 31, 63]
for i in range(4):
    plt.plot(np.sqrt(np.arange(ls[i])), logc_l[i], label=f"{ls[i]}")
plt.legend()
plt.show()

for i in range(4):
    plt.plot(np.sqrt(np.arange(ls[i])), logc_ml_l[i], label=f"{ls[i]}")
plt.legend()
plt.show()


slope_0, intercept, r_value, p_value, std_err = stats.linregress(
    np.sqrt(np.arange(ls[-1]))[7:35], logc_ml_l[-1][7:35]
)
print(slope_0, intercept, r_value)
slope_0, intercept, r_value, p_value, std_err = stats.linregress(
    np.sqrt(np.arange(ls[-1]))[7:35], logc_l[-1][7:35]
)
print(slope_0, intercept, r_value)


#%% PART II(b): accuracy analysis of the average value
xx_ml = np.average(xx_ml, axis=0)
xx = np.average(xx, axis=0)


plt.plot(xx[1, :])
plt.plot(xx_ml[1, :])
plt.show()
plt.plot(z[0, :])
plt.show()

# accuracy measure L2
plt.imshow(xx)
plt.colorbar()
plt.show()
plt.imshow(xx_ml)
plt.colorbar()
plt.show()
dxx = np.sqrt(np.average((xx - xx_ml) ** 2) / np.average((xx) ** 2))
print(dxx)

# for i in range(xx.shape[-1]):
#     plt.plot(xx[:,i],label='exact')
#     plt.plot(xx_ml[:,i],label='ml')
#     plt.legend()
#     plt.show()


g = np.average(xx, axis=-1)
g_ml = np.average(xx_ml, axis=-1)

#%%
x = np.arange(int(l / 2) - 1)

plt.plot(x, (g), label=f"g")
plt.plot((x), (g_ml), label=f"g_ml")
# plt.axvline(x=2)
# plt.axvline(x=45)
plt.legend()
plt.loglog()
plt.show()

nu = np.average((np.log(g[7:35])) / np.log(x[7:35]))
print(nu)

nu_ml = np.average((np.log(g_ml[7:35])) / np.log(x[7:35]))
print(nu_ml)


#%% compute the slope of the curve
from scipy import stats

slope_0, intercept, r_value, p_value, std_err = stats.linregress(
    np.log(x[2:45]), np.log(g)[2:45]
)

print(slope_0, intercept, r_value)

slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(
    np.log(x[2:45]), np.log(g_ml)[2:45]
)

print(slope_1, intercept, r_value)

#%%


#%%
for i in range(xx.shape[-1]):
    plt.plot(z[0])
    plt.plot(xx[0, i])
    plt.plot(xx_ml[0, i])
    plt.title(f"{i}-th component")
    plt.show()


# %% Den2Magn Analysis

#%% Testing the neural network by using DMRG dataset
h_max = 1.8
ls = [16, 32, 64]
ns = [100, 100, 100]


xs_ml = {}
xs = {}
zs = {}
for i in range(len(ls)):

    data = np.load(
        f"data/den2magn_dataset_1nn/test_unet_periodic_1nn_l_{ls[i]}_h_{h_max}_ndata_{ns[i]}.npz"
    )
    # data=np.load('data/den2magn_dataset_1nn/train_unet_periodic_1nn_8_l_4.50_h_150000_n.npz')
    x = torch.tensor(data["magnetization_x"], dtype=torch.double)
    z = torch.tensor(data["density"], dtype=torch.double)
    model = torch.load(
        f"model_rep/1nn_den2magn/h_{h_max:.1f}_15k_unet_periodic_den2magn_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock",
        map_location="cpu",
    )

    x_ml = model(z).detach().numpy()
    xs_ml[ls[i]] = np.abs(x_ml)
    xs[ls[i]] = np.abs(x.detach().numpy())  # we analyze
    # xs_ml[ls[i]]=(x_ml)
    # xs[ls[i]]=(x.detach().numpy()) #we analyze
    # x**2 instead of x
    zs[ls[i]] = z.detach().numpy()


# %% I part: have a look to the profiles
l = 64

for i in range(10):
    plt.plot(xs_ml[l][i], label="ml")
    plt.plot(xs[l][i], label="exact")
    plt.legend()
    plt.xlabel("l", fontsize=20)
    plt.ylabel("x", fontsize=20)
    plt.show()


# metrics absolute relative error
dx = np.average(
    np.sqrt(np.average((xs[l] - xs_ml[l]) ** 2, axis=-1))
    / np.sqrt(np.average((xs[l]) ** 2, axis=-1))
)

print(dx)

#%% binder cumulator
u_ml = 0.5 * (
    3 - np.average(xs_ml[l] ** 4, axis=-1) / np.average(xs_ml[l] ** 2, axis=-1) ** 2
)
u_ml_av = np.average(u_ml)
print(u_ml_av)

u = 0.5 * (3 - np.average(xs[l] ** 4, axis=-1) / np.average(xs[l] ** 2, axis=-1) ** 2)
u_av = np.average(u)
print(u_av)

# %% average over the disorder

x_ml_ave = np.average(xs_ml[l], axis=0)
x_ave = np.average(xs[l], axis=0)

# %%
# plt.plot(x_ml_ave)
# plt.plot(x_ave)
# plt.show()

print(np.average(x_ml_ave))
print(np.average(x_ave))
# %% augmentation
# data=np.load(f'data/den2magn_dataset_1nn/train_unet_periodic_1nn_16_l_2.71_h_15000_n.npz')

# x=data['magnetization_x']
# z=data['density']

# z=np.append(z,-1*z,axis=0)
# x=np.append(x,x,axis=0)
# np.savez('data/den2magn_dataset_1nn/train_unet_periodic_1nn_augmentation_16_l_2.71_h_30000_n.npz',density=z,magnetization_x=x)
# %% Losses analysis
import torch

data = torch.load("")
