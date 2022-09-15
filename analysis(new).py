#%% imports
import time
from ast import increment_lineno
from turtle import position
from numpy.lib.mixins import _inplace_binary_method
from pyrsistent import l
import torch as pt
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from scipy import fft, ifft
import os
from src.training.utils_analysis import dataloader, ResultsAnalysis


#%% first analysis --> SCALABILITY AT THE CRITICAL VALUE
n_sample=7
t_range=[49,49,49,99,99,149,199]
h_max=[2.7]*n_sample
ls=[16,32,128,256,512,1024,2048]
n_instances=[100]*n_sample
epochs = [[i * 100 for i in range(t_range[j])] for j in range(n_sample)] 
# model settings
name_session=[f'h_{h_max[i]}_150k_augmentation_1nn_model_unet_{ls[i]}_size_2_layers_20_hc_5_ks_2_ps' for i in range(n_sample)]
early_stopping=False
variational_lr=False
loglr=1

min_density={}
gs_density={}
min_energy={}
gs_energy={}

for i in range(n_sample):
    min_eng=[]
    gs_eng=[]
    min_n=[]
    gs_n=[]
    for j in range(t_range[i]):    
        min_eng_t,gs_eng_t=dataloader('energy',session_name=name_session[i],n_instances=n_instances[i],lr=loglr,diff_soglia=1,epochs=epochs[i][j],early_stopping=False,variable_lr=False,n_ensambles=1)
        min_n_t,gs_n_t=dataloader('density',session_name=name_session[i],n_instances=n_instances[i],lr=loglr,diff_soglia=1,epochs=epochs[i][j],early_stopping=False,variable_lr=False,n_ensambles=1)
        min_eng.append(min_eng_t)
        gs_eng.append(gs_eng_t)
        min_n.append(min_n_t)
        gs_n.append(gs_n_t)
    min_eng=np.asarray(min_eng)
    gs_eng=np.asarray(gs_eng)
    min_n=np.asarray(min_n)
    gs_n=np.asarray(gs_n)

    min_energy[ls[i]]=min_eng
    gs_energy[ls[i]]=gs_eng
    min_density[ls[i]]=min_n
    gs_density[ls[i]]=gs_n        



# %%
print(min_energy[16].shape)
# %%
e_av=np.average(np.abs(min_energy[16]-gs_energy[16])/np.abs(gs_energy[16]),axis=-1)
plt.plot(e_av)
plt.axhline(0,color='red')
plt.show()
# %%
