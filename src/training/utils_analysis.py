# Libraries
from typing import List, Dict, Tuple
import numpy as np
import argparse
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
from torchmetrics import R2Score
from tqdm.notebook import tqdm, trange
from torch.utils.data import Dataset, TensorDataset, DataLoader
import matplotlib.pyplot as plt
from src.training.models import Energy_unet
from src.training.utils import count_parameters


def dataloader(
    type: str,
    session_name: str,
    n_instances: int,
    lr: int,
    diff_soglia: int,
    n_ensambles: int,
    epochs: int,
    early_stopping: bool,
    variable_lr: bool,
):

    session_name = session_name

    name_istances = f"number_istances_{n_instances}"
    session_name = session_name + "_" + name_istances

    n_initial_name = f"n_ensamble_{n_ensambles}_different_initial"
    session_name = session_name + "_" + n_initial_name

    epochs_name = f"epochs_{epochs}"
    session_name = session_name + "_" + epochs_name

    lr_name = f"lr_{lr}"
    session_name = session_name + "_" + lr_name

    if variable_lr:
        variable_name = "variable_lr"
        session_name = session_name + "_" + variable_name

    if early_stopping:
        diff_name = f"diff_soglia_{diff_soglia}"
        session_name = session_name + "_" + diff_name

    if type == "density":

        data = np.load(
            "data/gd_data/density_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_n = data["min_density"]
        gs_n = data["gs_density"]
        return min_n, gs_n

    elif type == "energy":

        data = np.load(
            "data/gd_data/eng_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_eng = data["min_energy"]
        gs_eng = data["gs_energy"]
        return min_eng, gs_eng

    elif type == "history":

        data = np.load(
            "data/gd_data/history/history_" + session_name + ".npz",
            allow_pickle=True,
        )

        history = data["history"]

        history_n = data["history_n"]

        return history, history_n


def test_models_unet(models_name: List, data_path: List):
    r_square_list = []
    accuracy_prediction_energy_average = []
    accuracy_prediction_energy_std = []
    r2 = R2Score()

    for i, model_name in enumerate(models_name):

        data = np.load(data_path[i])
        n_std = data["density"]
        F_std = data["F"]
        e_std = data["energy"]
        v_std = data["potential"]
        ds = TensorDataset(
            pt.tensor(n_std), pt.tensor(F_std), pt.tensor(v_std), pt.tensor(e_std)
        )
        dl = DataLoader(ds, batch_size=100)

        model = pt.load("model_rep/" + model_name, map_location="cpu")
        model.eval()
        model = model.to(dtype=pt.double)

        dde = []
        ddevde = []
        for n, f, v, e_std in dl:
            n = n.to(dtype=pt.double)
            f = f.to(dtype=pt.double)
            v = v.to(dtype=pt.double)
            e_std = e_std.numpy()
            model.eval()
            energy = Energy_unet(model, pt.tensor(v, dtype=pt.double))

            output = model(n)
            # print(f.shape)
            r2.update(output.mean(dim=-1), f)
            # print(f[0],pt.mean(output,dim=-1)[0])
            eng = energy.batch_calculation(n.squeeze())
            de = np.average(np.abs(eng.detach().numpy() - e_std) / np.abs(e_std))
            devde = np.std(np.abs(eng.detach().numpy() - e_std) / np.abs(e_std))
            dde.append(de)
            ddevde.append(devde)
        print(model)
        print(f"# parameters={count_parameters(model)}")
        print(f"R_square_test={r2.compute()} for {model_name} \n")
        print(f"de={de} std(de)={devde}")
        r_square_list.append(r2.compute().detach().numpy())
        r2.reset()
        accuracy_prediction_energy_average.append(np.average(dde))
        accuracy_prediction_energy_std.append(np.average(ddevde))
    return (
        np.asarray(r_square_list),
        np.asarray(accuracy_prediction_energy_average),
        np.asarray(accuracy_prediction_energy_std),
    )


def parallel_nambu_diagonalization_ising_model(
    nbatch, l: int, j_coupling: float, hs: np.array, device: str, pbc: bool
):
    """Compute the correlation <\sigma_x \sigma_x>(ij) of the transverse quantum ising model using the Nambu Mapping

    Args:
        l (int): length of the chain
        j_coupling (float): coupling costant of the spin interaction
        hs (np.array): realizations of the magnetic field [batch,l]
        checkpoint (bool): if True, it creates a npz version of the dataset
        train (bool): if True, it labels the file as train, valid otherwise
        device(str): the device used for the computation. Can be either 'cuda' or 'cpu' (standard is 'cpu').
    Returns:
        e,f,m_z (Tuple[np.array]): a triple of energies, H-K functional values and transverse magnetizations for each hs realizations
    """

    n_dataset = hs.shape[0]

    batch = int(n_dataset / nbatch)
    # uniform means h_ave=0
    hs = pt.tensor(hs, dtype=pt.double, device=device)

    # obc
    j_vec = j_coupling * pt.ones(l, device=device)
    # the 0-th component is null in OBC
    j_vec_l = j_vec.clone()
    if not (pbc):
        j_vec_l[0] = 0
    if pbc:
        j_vec_l[0] = -1 * j_vec_l[0]

    # the l-th component is null in OBC
    j_vec_r = j_vec.clone()
    if not (pbc):
        j_vec_r[-1] = 0
    if pbc:
        j_vec_r[-1] = -1 * j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l = pt.einsum("ij,j->ij", pt.eye(l, device=device), j_vec_l)
    j_l = pt.roll(j_l, shifts=-1, dims=1)
    j_r = pt.einsum("ij,j->ij", pt.eye(l, device=device), j_vec_r)
    j_r = pt.roll(j_r, shifts=1, dims=1)
    # the coupling part for a
    j = -0.5 * (j_r + j_l)
    # the coupling part for b
    j_b = -0.5 * (j_r - j_l)
    # the b matrix of the nambu matrix
    b = j_b

    for i in trange(nbatch):
        # the external field
        h = hs[i * batch : (i + 1) * batch]
        h_matrix = pt.einsum("ij,aj->aij", pt.eye(l, device=device), h)
        # the a matrix of the nambu matrix
        a = j + h_matrix

        # create the nambu matrix
        h_nambu = pt.zeros((batch, 2 * l, 2 * l), device=device)
        h_nambu[:, :l, :l] = a
        h_nambu[:, :l, l:] = b
        h_nambu[:, l:, :l] = -1 * pt.conj(b)
        h_nambu[:, l:, l:] = -1 * pt.conj(a)

        e, w = pt.linalg.eigh(h_nambu)

        # the v coefficients
        v = w.clone()[:, l:, :l]

        u = w.clone()[:, :l, :l]
        # compute the correlation sigma_x sigma_x
        c_vv = pt.einsum("anl,aml->anm", v, pt.conj(v))
        c_uu = pt.einsum("anl,aml->anm", u, pt.conj(u))
        c_vu = pt.einsum("anl,aml->anm", v, pt.conj(u))
        c_uv = pt.einsum("anl,aml->anm", u, pt.conj(v))
        c = c_vv + c_vu - c_uu - c_uv

        s_z = 1 - 2 * pt.einsum("aik,aik->ai", v, pt.conj(v))
        s_z_different = pt.einsum("aik,aik->ai", u, pt.conj(u)) - pt.einsum(
            "aik,aik->ai", v, pt.conj(v)
        )

        density_f = c[:, np.arange(l), (np.arange(l) + 1) % l]
        density_f[:, -1] = -1 * density_f[:, -1]

        e_0 = pt.sum(e[:, 0:l], dim=-1) / l
        f = e_0 - pt.mean(h * s_z, dim=-1)

        if i == 0:
            magn_z = s_z
            magn_z_diff = s_z_different
            e_tot = e_0
            f_tot = f
            tot_density_f = density_f
        else:
            magn_z = np.append(magn_z, s_z, axis=0)
            magn_z_diff = np.append(magn_z_diff, s_z_different, axis=0)
            e_tot = np.append(e_tot, e_0)
            f_tot = np.append(f_tot, f)
            tot_density_f = np.append(tot_density_f, density_f, axis=0)

    return hs, magn_z, magn_z_diff, f_tot, tot_density_f, e_tot


def nuv_representability_check(
    model: nn.Module, z: np.ndarray, v: np.ndarray, plot: bool, gs_z: np.ndarray
):

    l = z.shape[-1]
    x = pt.tensor(z, dtype=pt.double)
    x.requires_grad_(True)
    f = pt.mean(model(x), dim=-1)  # the total f value
    f.backward(pt.ones_like(f))
    with pt.no_grad():
        grad = x.grad
        grad = -l * grad.detach().numpy()
        pseudo_pot = grad
    # print(grad.shape)
    g_acc = np.average(np.abs(v - pseudo_pot), axis=-1) / np.average(np.abs(v), axis=-1)

    if plot:
        for i in range(0, 10):
            plt.plot(pseudo_pot[i], label="ml grad")
            plt.plot(v[i], label="pot")
            plt.legend()
            plt.show()

    nbatch = 10
    j_coupling = 1
    _, m, _, f, fm, e = parallel_nambu_diagonalization_ising_model(
        nbatch=nbatch,
        l=l,
        j_coupling=j_coupling,
        hs=pseudo_pot,
        device="cpu",
        pbc=True,
    )
    z_acc = np.average(np.abs(m - gs_z), axis=-1) / np.average(np.abs(z), axis=-1)

    if plot:
        for i in range(0, 10):
            plt.plot(m[i], label="pseudo pot --> z")
            plt.plot(gs_z[i], label="pot --> z")
            plt.legend()
            plt.show()

    return g_acc, z_acc
