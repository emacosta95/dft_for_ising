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
from src.training.models_unet import Energy_unet
from src.training.utils import count_parameters
from src.training.models import Energy



def dataloader(
    type: str,
    model_name: str,
    cut: int,
    n_instances: int,
    lr: int,
    diff_soglia: int,
    n_ensambles: int,
    epochs: int,
    early_stopping: bool,
    variable_lr: bool,
):

    session_name = model_name

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
            "gradient_descent_ensamble_numpy/min_density_" + session_name + ".npz",
            allow_pickle=True,
        )

        min_n = data["min_density"]
        gs_n = data["gs_density"]
        return min_n, gs_n

    elif type == "energy":

        data = np.load(
            "gradient_descent_ensamble_numpy/min_vs_gs_gradient_descent_"
            + session_name
            + ".npz",
            allow_pickle=True,
        )

        min_eng = data["min_energy"]
        gs_eng = data["gs_energy"]
        return min_eng, gs_eng

    elif type == "history":

        data = np.load(
            "gradient_descent_ensamble_numpy/history_" + session_name + ".npz",
            allow_pickle=True,
        )

        history = data["history"]

        history_n = data["history_n"]

        return history, history_n



def trapez(f: pt.tensor, dx: float):

    f_roll = pt.roll(f, shifts=1, dims=-1)
    return pt.sum(f_roll * f, dim=-1) * dx




class ResultsAnalysis:
    def __init__(
        self,
        n_sample: List,
        n_instances: List,
        n_ensambles: List,
        epochs: List,
        diff_soglia: List,
        models_name: List,
        text: List,
        variable_lr: List,
        early_stopping: List,
        lr:int,
        dx: float,
    ):
        self.models_name = models_name
        self.text = text
        self.dx = dx

        self.r_square_list = None
        

        self.min_eng = []
        self.gs_eng = []
        self.min_n = []
        self.gs_n = []

        self.accuracy_prediction_energy_average=None
        self.accuracy_prediction_energy_std=None

        for i in range(len(n_sample)):

            x_min = []
            x_gs = []
            y_min = []
            y_gs = []

            for j in range(len(epochs[i])):

                min_eng, gs_eng = dataloader(
                    "energy",
                    model_name=models_name[i][j],
                    cut=128,
                    n_instances=n_instances[i][j],
                    lr=lr,
                    diff_soglia=diff_soglia[i][j],
                    n_ensambles=n_ensambles[i][j],
                    epochs=epochs[i][j],
                    early_stopping=early_stopping[i][j],
                    variable_lr=variable_lr[i][j],
                )

                min_n, gs_n = dataloader(
                    "density",
                    model_name=models_name[i][j],
                    cut=128,
                    n_instances=n_instances[i][j],
                    lr=lr,
                    diff_soglia=diff_soglia[i][j],
                    n_ensambles=n_ensambles[i][j],
                    epochs=epochs[i][j],
                    early_stopping=early_stopping[i][j],
                    variable_lr=variable_lr[i][j],
                )

                x_min.append(min_eng)
                x_gs.append(gs_eng)
                y_min.append(min_n)
                y_gs.append(gs_n)

            self.min_eng.append(x_min)
            self.gs_eng.append(x_gs)
            self.min_n.append(y_min)
            self.gs_n.append(y_gs)

    def _comparison(self):

        self.list_de = []
        self.list_devde = []
        self.list_dn = []
        self.list_devdn = []
        self.list_delta_e = []
        self.list_delta_devde = []
        self.list_abs_err_n = []
        self.list_R_square = []
        self.list_R_square_energy = []

        for i in range(len(self.min_eng)):

            av_eng_values = []
            std_eng_values = []
            av_dn_values = []
            std_dn_values = []
            av_eng_valore = []
            std_eng_valore = []
            gradient_min_ns = []
            gradient_gs_ns = []
            delta_gradient_ns = []
            av_delta_gradient_ns = []
            dev_delta_gradient_ns = []
            r_square = []
            r_square_energy = []
            abs_err_n = []
            dn_abs_error = []
            min_engs = []
            gs_engs = []
            min_ns = []
            gs_ns = []
            dns = []
            des = []
            dx = self.dx


            for j in range(len(self.min_eng[i])):
                dns.append(
                    np.sqrt(
                        np.sum(
                            (self.min_n[i][j] - self.gs_n[i][j]) ** 2, axis=1
                        )
                    )
                    / np.sqrt(np.sum(self.gs_n[i][j] ** 2, axis=1))
                )
                dn_abs_error.append(
                    np.sum(np.abs(self.min_n[i][j] - self.gs_n[i][j]), axis=1) 
                )
                
                
                av_eng_values.append(
                    np.average(
                        np.abs(
                            (self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j]
                        )
                    )
                )
                r_square_energy.append(
                    1
                    - np.sum((self.gs_eng[i][j] - self.min_eng[i][j]) ** 2)
                    / (self.gs_eng[i][j].shape[0] * np.std(self.gs_eng[i][j]) ** 2)
                )
                av_eng_valore.append(
                    np.average(
                        (self.min_eng[i][j] - self.gs_eng[i][j]) / self.min_eng[i][j]
                    )
                )
                std_eng_valore.append(
                    np.std(
                        ((self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j])
                        #    / np.sqrt(self.min_eng[i][j].shape[0] - 1)
                    )
                )
                std_eng_values.append(
                    np.std(
                        np.abs(
                            (self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j]
                        )
                        #    / np.sqrt(self.min_eng[i][j].shape[0] - 1)
                    )
                )

                av_dn_values.append(np.average(dns[j]))
                std_dn_values.append(
                    np.std(dns[j])
                    # / np.sqrt(dns[j].shape[0] - 1)
                )
                
                abs_err_n.append(np.average(dn_abs_error[j]))

            self.list_de.append(av_eng_values)
            self.list_R_square.append(r_square)
            self.list_R_square_energy.append(r_square_energy)
            self.list_devde.append(std_eng_values)
            self.list_dn.append(av_dn_values)
            self.list_devdn.append(std_dn_values)
            self.list_delta_e.append(av_eng_valore)
            self.list_delta_devde.append(std_eng_valore)
            self.list_abs_err_n.append(abs_err_n)

    def plot_results(
        self,
        xticks: List,
        xposition: List,
        yticks: Dict,
        position: List,
        labels: list,
        xlabel: str,
        title: str,
        loglog: bool,
        marker: list,
        linestyle: list,
        color: list,
        symbol:list,
        symbolx:list,
        symboly:list
    ):

        self._comparison()

        fig = plt.figure(figsize=(20, 20))
        for i, des in enumerate(self.list_de):
            plt.errorbar(
                x=position[i],
                y=des,
                yerr=self.list_devde[i],
                label=labels[i],
                marker=marker[i],
                linestyle=linestyle[i],
                linewidth=3,
                color=color[i],
                markersize=10,
            )
        plt.ylabel(r"$\langle \left|\Delta e \right|/e \rangle$", fontsize=50)
        plt.xlabel(xlabel, fontsize=50)
        plt.xticks(labels=xticks, ticks=xposition)
        if yticks["de"] != None:
            plt.yticks(yticks["de"])
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
        #plt.text(symbolx[0],symboly[0], symbol[0], weight='bold',fontsize=30)
        plt.legend(fontsize=30)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()
        
        fig = plt.figure(figsize=(20, 20))
        for i, devde in enumerate(self.list_devde):
            plt.plot(
                position[i],
                devde,
                label=labels[i],
                linewidth=3,
                linestyle=linestyle[i],
                marker=marker[i],
                markersize=10,
                color=color[i],
            )
        plt.ylabel(r"$\sigma(\left|\Delta e \right|/ \left|e \right|)$", fontsize=50)
        plt.xlabel(xlabel, fontsize=50)
        plt.xticks(labels=xticks, ticks=xposition)
        if yticks["devde"] != None:
            plt.yticks(yticks["devde"])
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=50,
            width=5,
        )
        plt.legend(fontsize=20)
        plt.title(title)

        if loglog:
            plt.loglog()
        plt.show()
        

        fig = plt.figure(figsize=(20, 20))
        for i, dn in enumerate(self.list_dn):
            plt.errorbar(
                x=position[i],
                y=dn,
                yerr=self.list_devdn[i] ,
                label=labels[i],
                linewidth=3,
                linestyle=linestyle[i],
                marker=marker[i],
                markersize=10,
                color=color[i],
            )
        plt.axhline(y=0,linestyle=':',color='gold',linewidth=4)
        plt.ylabel(r"$\langle \left| \Delta n \right|/ \left|n \right| \rangle$", fontsize=50)
        plt.xlabel(xlabel, fontsize=50)
        plt.xticks(labels=xticks, ticks=xposition)
        ##plt.text(symbolx[1],symboly[1], symbol[1], weight='bold',fontsize=30)
        if yticks["dn"] != None:
            plt.yticks(yticks["dn"])
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=50,
            width=5,
        )
        plt.legend(fontsize=30)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()
        
        fig = plt.figure(figsize=(20, 20))
        for i, dn in enumerate(self.list_abs_err_n):
            plt.errorbar(
                x=position[i],
                y=dn,
                yerr=self.list_devdn[i] ,
                label=labels[i],
                linewidth=3,
                linestyle=linestyle[i],
                marker=marker[i],
                markersize=10,
                color=color[i],
            )
        plt.ylabel(r"$\rangle \left|\Delta n \right|_{l0}/ \left|n \right|_{l0} \langle$", fontsize=30)
        plt.xlabel(xlabel, fontsize=30)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=50,
            width=5,
        )
        plt.xticks(labels=xticks, ticks=xposition)
        plt.legend(fontsize=30)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(20, 20))
        for i, devdn in enumerate(self.list_devdn):
            plt.plot(
                position[i],
                devdn,
                label=labels[i],
                linewidth=3,
                linestyle=linestyle[i],
                marker=marker[i],
                markersize=10,
                color=color[i],
            )
        
        plt.xlabel(xlabel, fontsize=30)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=50,
            width=5,
        )

        plt.xticks(labels=xticks, ticks=xposition)
        if yticks["devdn"] != None:
            plt.yticks(yticks["devdn"])
        plt.ylabel(r"$\sigma(|\Delta n|/|n|)$", fontsize=30)
        plt.legend(fontsize=30)
        plt.title(title)
        if loglog:
            plt.loglog()
        plt.show()

        fig = plt.figure(figsize=(20, 20))
        for i, des in enumerate(self.list_delta_e):
            plt.errorbar(
                x=position[i],
                y=des,
                yerr=self.list_delta_devde[i] ,
                label=labels[i],
                linewidth=3,
                linestyle=linestyle[i],
                marker=marker[i],
                markersize=10,
                color=color[i],
            )
        plt.ylabel(r"$\langle \Delta e/e \rangle$", fontsize=30)
        plt.xlabel(xlabel, fontsize=30)
        plt.xticks(labels=xticks, ticks=xposition)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=50,
            width=5,
        )
        plt.legend(fontsize=30)
        plt.title(title)
        plt.show()
        
        fig = plt.figure(figsize=(20, 20))
        

        fig = plt.figure(figsize=(20, 20))
        for i, das in enumerate(self.list_R_square_energy):
            plt.plot(
                position[i],
                das,
                label=labels[i],
                linewidth=3,
                linestyle=linestyle[i],
                marker=marker[i],
                markersize=10,
                color=color[i],
            )
        plt.ylabel(r"$R^2 energy$", fontsize=30)
        plt.xlabel(xlabel, fontsize=30)
        plt.xticks(ticks=xposition, labels=xticks)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
        plt.legend(fontsize=30)
        plt.title(title)
        plt.show()

    def plot_samples(
        self, idx: List, jdx: List, n_samples: int, title: str, l: float,v:np.array ,letterx:float,lettery:float,letter:str, alpha=float,
    ):
        space = np.linspace(0, l, self.min_n[0][0].shape[1])
        
            
        for k in range(n_samples):
            fig, ax1 = plt.subplots(figsize=(20,20))
            
            for i in idx:
                for j in jdx:
                    plt.plot(
                        space,
                        self.min_n[i][j][k],
                        label= self.text[i][j],
                        linewidth=7,
                        alpha=alpha,
                    )
            plt.plot(
                space,
                self.gs_n[i][j][k],
                linestyle="--",
                alpha=1,
                linewidth=7,
                color='red',
                label="ground state",
            )
            ax1.set_xlabel(r"$x$", fontsize=60)
            ax1.set_ylabel(r"$n(x)$", fontsize=60)
            ax1.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=40,
            width=5,
            )
            ax1.legend(fontsize=30,loc='upper left')
            
            

            # ax_twin = ax1.twinx()
            # ax_twin.plot(
            #     space,
            #     v[k],
            #     linestyle="-.",
            #     alpha=1,
            #     color='black',
            #     linewidth=3,
            #     label="potential",
            # )

            #ax_twin.set_ylabel(r"$V(x)$", fontsize=60)
            # ax_twin.tick_params(
            # top=True,
            # right=True,
            # labeltop=False,
            # labelright=True,
            # direction="in",
            # labelsize=40,
            # width=5,
            # )

            plt.xticks([0,5,10,14])
            #ax_twin.legend(fontsize=30,loc='upper right')
            #plt.text(letterx,lettery, letter, weight='bold',fontsize=30)
        

            plt.show()

    def histogram_plot(
        self,
        idx: List,
        jdx: List,
        title: str,
        bins: int,
        density: bool,
        alpha: List,
        hatch: List,
        color: List,
        fill: List,
        range_eng: Tuple,
        range_n: Tuple,
        range_a:Tuple,
        logx: bool,
        labels:List,
        textx:List,
        texty:List,
        i_pred:int,
        j_pred:int
    ):
        fig = plt.figure(figsize=(10, 10))
        for eni, i in enumerate(idx):
            for enj, j in enumerate(jdx):
                dn = np.sqrt(
                    np.sum((self.min_n[i][j] - self.gs_n[i][j]) ** 2, axis=1)
                ) / np.sqrt(np.sum(self.gs_n[i][j] ** 2, axis=1))
                plt.hist(
                    dn,
                    bins,
                    label=labels[i],
                    range=range_n,
                    alpha=alpha[eni][enj],
                    hatch=hatch[eni][enj],
                    fill=fill[eni][enj],
                    color=color[eni][enj],
                    histtype="step",
                )
                
        plt.xlabel(r"$|\Delta n|/|n|$", fontsize=50)
        plt.legend(fontsize=30, loc="best")
        if logx:
            plt.semilogx()
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
        #plt.text(textx[0],texty[0],'(b)', weight='bold',fontsize=30)
        plt.show()
        
        fig = plt.figure(figsize=(10, 10))
        for eni, i in enumerate(idx):
            for enj, j in enumerate(jdx):
                de = (self.min_eng[i][j] - self.gs_eng[i][j]) / self.gs_eng[i][j]
                plt.hist(
                    de,
                    bins,
                    label=labels[i],
                    density=density,
                    alpha=alpha[eni][enj],
                    range=range_eng,
                    hatch=hatch[eni][enj],
                    fill=fill[eni][enj],
                    color=color[eni][enj],
                    histtype="step",
                )
        if self.accuracy_prediction_energy_average !=None:
            plt.axvspan(-1*self.accuracy_prediction_energy_average[i][j],self.accuracy_prediction_energy_average[i_pred][j_pred], color='green', alpha=0.3,label='prediction range')

        plt.xlabel(r"$\Delta e/e$", fontsize=50)
        plt.legend(fontsize=30, loc="upper left")
        #plt.text(textx[1],texty[1],'(a)', weight='bold',fontsize=30)
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
        plt.show()
        #plt.savefig('images/hist_a.eps', format='eps', dpi=1200)
        fig = plt.figure(figsize=(10, 10))
        for eni, i in enumerate(idx):
            for enj, j in enumerate(jdx):
                    da = (
                        np.average(
                            np.abs(
                                np.gradient(self.min_n[i][j], self.dx, axis=1)
                                / (np.gradient(self.gs_n[i][j], self.dx, axis=1) + 10 ** -30)
                            )-1,
                            axis=1,
                     ))

                    plt.hist(
                        da,
                        bins,
                        label=labels[i],
                        range=range_a,
                        alpha=alpha[eni][enj],
                        hatch=hatch[eni][enj],
                        fill=fill[eni][enj],
                        color=color[eni][enj],
                        histtype="step",
                    )
        plt.xlabel(r"$\Delta A[n_{gs},n_{min}]$", fontsize=50)
        plt.legend(fontsize=30, loc="best")
        #plt.text(textx[2],texty[2],'(c)', weight='bold',fontsize=30)
        if logx:
            plt.semilogx()
        plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
        plt.show()
    
    def test_models(self, idx: List, jdx: List, data_path: str):
        self.r_square_list = []
        self.accuracy_prediction_energy_average=[]
        self.accuracy_prediction_energy_std=[]
        r2 = R2Score()

        data=np.load(data_path)
        n_std = data["density"]
        F_std = data["F"]
        e_std=data['energy']
        v_std=data['potential']
        ds = TensorDataset(
            pt.tensor(n_std).view(-1, 1, n_std.shape[-1]), pt.tensor(F_std),pt.tensor(v_std),pt.tensor(e_std)
        )
        dl = DataLoader(ds, batch_size=100)
        for i in idx:
            dde=[]
            std_dde=[]
            for j in jdx:
                model = pt.load(
                    "model_dft_pytorch/" + self.models_name[i][j], map_location="cpu"
                )
                model.eval()
                model = model.to(dtype=pt.double)
                
                des=[]
                std_des=[]
                for n, f,v,e_std in dl:
                    n = n.to(dtype=pt.double)
                    f = f.to(dtype=pt.double)
                    v=v.to(dtype=pt.double)
                    e_std=e_std.numpy()
                    model.eval()
                    energy=Energy(model,pt.tensor(v,dtype=pt.double),self.dx)

                    output = model(n)
                    output = output.view(output.shape[0])
                    r2.update(output, f)
                    eng=energy.batch_calculation(n.squeeze())
                    de=np.average(np.abs(eng.detach().numpy()-e_std)/e_std)
                    des.append(de)
                    devde=np.std(np.abs(eng.detach().numpy()-e_std)/e_std)
                    std_des.append(devde)


                
                delta_e=np.average(des)
                std=np.std(std_des)
                print(model)
                print(f"# parameters={count_parameters(model)}")
                print(f"R_square_test={r2.compute()} for {self.text[i][j]} \n")
                print(f'de={delta_e} std(de)={std}')
                dde.append(delta_e)
                self.r_square_list.append(r2.compute().detach().numpy())
                r2.reset()
            self.accuracy_prediction_energy_average.append(dde)
            self.accuracy_prediction_energy_std.append(std_dde)


def test_models( models_name: List, data_path: List):
        r_square_list = []
        accuracy_prediction_energy_average=[]
        accuracy_prediction_energy_std=[]
        r2 = R2Score()

        for i,model_name in enumerate(models_name):
        
            data=np.load(data_path[i])
            n_std = data["density"]
            F_std = data["F"]
            e_std=data['energy']
            v_std=data['potential']
            ds = TensorDataset(
                pt.tensor(n_std).view(-1, 1, n_std.shape[-1]), pt.tensor(F_std),pt.tensor(v_std),pt.tensor(e_std)
            )
            dl = DataLoader(ds, batch_size=100)
            
        
            model = pt.load(
                "model_rep/" + model_name, map_location="cpu"
            )
            model.eval()
            model = model.to(dtype=pt.double)
            
            dde=[]
            ddevde=[]
            for n, f,v,e_std in dl:
                n = n.to(dtype=pt.double)
                f = f.to(dtype=pt.double)
                v=v.to(dtype=pt.double)
                e_std=e_std.numpy()
                model.eval()
                energy=Energy(model,pt.tensor(v,dtype=pt.double))

                output = model(n)
                output = output.view(output.shape[0])
                r2.update(output, f)
                eng=energy.batch_calculation(n.squeeze())
                de=np.average(np.abs(eng.detach().numpy()-e_std)/np.abs(e_std))
                devde=np.std(np.abs(eng.detach().numpy()-e_std)/np.abs(e_std))
                dde.append(de)
                ddevde.append(devde)
            print(model)
            print(f"# parameters={count_parameters(model)}")
            print(f"R_square_test={r2.compute()} for {model_name} \n")
            print(f'de={de} std(de)={devde}')
            r_square_list.append(r2.compute().detach().numpy())
            r2.reset()
            accuracy_prediction_energy_average.append(np.average(dde))
            accuracy_prediction_energy_std.append(np.average(ddevde))
        return np.asarray(r_square_list),np.asarray(accuracy_prediction_energy_average),np.asarray(accuracy_prediction_energy_std)



def test_models_unet( models_name: List, data_path: List):
        r_square_list = []
        accuracy_prediction_energy_average=[]
        accuracy_prediction_energy_std=[]
        r2 = R2Score()

        for i,model_name in enumerate(models_name):
        
            data=np.load(data_path[i])
            n_std = data["density"]
            F_std = data["F"]
            e_std=data['energy']
            v_std=data['potential']
            ds = TensorDataset(
                pt.tensor(n_std), pt.tensor(F_std),pt.tensor(v_std),pt.tensor(e_std)
            )
            dl = DataLoader(ds, batch_size=100)
            
        
            model = pt.load(
                "model_rep/" + model_name, map_location="cpu"
            )
            model.eval()
            model = model.to(dtype=pt.double)
            
            dde=[]
            ddevde=[]
            for n, f,v,e_std in dl:
                n = n.to(dtype=pt.double)
                f = f.to(dtype=pt.double)
                v=v.to(dtype=pt.double)
                e_std=e_std.numpy()
                model.eval()
                energy=Energy_unet(model,pt.tensor(v,dtype=pt.double))

                output = model(n)
                #print(f.shape)
                r2.update(output.mean(dim=-1), f)
                #print(f[0],pt.mean(output,dim=-1)[0])
                eng=energy.batch_calculation(n.squeeze())
                de=np.average(np.abs(eng.detach().numpy()-e_std)/np.abs(e_std))
                devde=np.std(np.abs(eng.detach().numpy()-e_std)/np.abs(e_std))
                dde.append(de)
                ddevde.append(devde)
            print(model)
            print(f"# parameters={count_parameters(model)}")
            print(f"R_square_test={r2.compute()} for {model_name} \n")
            print(f'de={de} std(de)={devde}')
            r_square_list.append(r2.compute().detach().numpy())
            r2.reset()
            accuracy_prediction_energy_average.append(np.average(dde))
            accuracy_prediction_energy_std.append(np.average(ddevde))
        return np.asarray(r_square_list),np.asarray(accuracy_prediction_energy_average),np.asarray(accuracy_prediction_energy_std)