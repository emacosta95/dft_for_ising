#%% imports
import time
from ast import increment_lineno
from turtle import position
from numpy.lib.mixins import _inplace_binary_method
from pyrsistent import l
import torch
import torch.nn as nn
import numpy as np
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from scipy import fft, ifft
import os
from src.training.utils_analysis import dataloader, ResultsAnalysis


#%% first analysis --> SCALABILITY AT THE CRITICAL VALUE for the 1nn
n_sample=7
t_range=[49,49,49,49,99,199,299]
h_max=[2.7]*n_sample
ls=[16,32,64,128,256,512,1024]
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
        
        min_eng_t=np.asarray(min_eng_t)
        gs_eng_t=np.asarray(gs_eng_t)
        min_n_t=np.asarray(min_n_t)
        gs_n_t=np.asarray(gs_n_t)
        
        if j==0:
            min_eng=min_eng_t.reshape(1,-1)
            gs_eng=gs_eng_t.reshape(1,-1)
            min_n=min_n_t.reshape(1,-1,ls[i])
            gs_n=gs_n_t.reshape(1,-1,ls[i])
        else:
            min_eng=np.append(min_eng,min_eng_t.reshape(1,-1),axis=0)
            gs_eng=np.append(gs_eng,gs_eng_t.reshape(1,-1),axis=0)
            min_n=np.append(min_n,min_n_t.reshape(1,-1,ls[i]),axis=0)
            gs_n=np.append(gs_n,gs_n_t.reshape(1,-1,ls[i]),axis=0)
            
        
    # min_eng=np.asarray(min_eng,dtype=object)
    # gs_eng=np.asarray(gs_eng,dtype=object)
    # min_n=np.asarray(min_n,dtype=object)
    # gs_n=np.asarray(gs_n,dtype=object)

    min_energy[ls[i]]=min_eng
    gs_energy[ls[i]]=gs_eng
    min_density[ls[i]]=min_n
    gs_density[ls[i]]=gs_n        


# %% ANALYSIS OF THE CONVERGENCE AT DIFFERENT SIZES for the absolute value
fig=plt.figure(figsize=(10,10))
for l in ls:
    print(l)
    print(min_energy[l].shape)
    print(gs_energy[l].shape)
    e_av=np.average(np.abs(min_energy[l]-gs_energy[l])/np.abs(gs_energy[l]),axis=-1)
    plt.plot(e_av,label=f'l={l}',linewidth=3)
plt.legend(fontsize=20)
plt.xlabel(r'$t$',fontsize=40)
plt.ylabel(r'$\langle\frac{\left| \Delta e \right|}{\left| e \right|}\rangle$',fontsize=40)


plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=20,
            width=5,
        )
plt.xticks([1000,5000,10000,15000,20000,25000,30000],[r'$10^3$',r'$5 \cdot 10^3$',r'$10^4$',r'$1.5 \cdot 10^4$',r'$2 \cdot 10^4$',r'$2.5 \cdot 10^4$',r'$3 \cdot 10^4$'])
plt.yticks([0.05,0.01,0.005,0.001],[r'$5 \cdot 10^{-2}$',r'$10^{-2}$',r'$5 \cdot 10^{-3}$',r'$ 10^{-3}$'])
plt.axhline(0.0007,color='red',linestyle='--',linewidth=2)
plt.loglog()
plt.show()

# %% ANALYSIS OF THE CONVERGENCE AT DIFFERENT SIZES for the deviation from the gs energy
fig=plt.figure(figsize=(10,10))
for i,l in enumerate(ls):
    e_av=np.average((min_energy[l]-gs_energy[l])/np.abs(gs_energy[l]),axis=1)
    plt.plot(epochs[i],e_av,label=f'l={l}',linewidth=3)
plt.legend(fontsize=20)
plt.xlabel(r'$t$',fontsize=40)
plt.ylabel(r'$\langle\frac{\Delta e}{\left| e \right|}\rangle$',fontsize=40)

plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=20,
            width=5,
        )
plt.xticks([1000,5000,10000,15000,20000,25000,30000],[r'$10^3$',r'$5 \cdot 10^3$',r'$10^4$',r'$1.5 \cdot 10^4$',r'$2 \cdot 10^4$',r'$2.5 \cdot 10^4$',r'$3 \cdot 10^4$'])
plt.yticks([0.05,0.01,0.005,0.001],[r'$5 \cdot 10^{-2}$',r'$10^{-2}$',r'$5 \cdot 10^{-3}$',r'$ 10^{-3}$'])
plt.axhline(0,color='red',label='zero point')
#plt.semilogy()
plt.show()



































# CORRELATION MAP from density to correlation (TEST)

#%% PART I: loading the data
batch=1000
l=64
h_max=2.7
data=np.load(f'data/correlation_1nn/test_1nn_correlation_map_h_{h_max}_n_1000_l_{l}_pbc_j_1.0.npz')
z=data['density'][:batch]
xx=data['correlation'][:batch]
z_torch=torch.tensor(z,dtype=torch.double)
print(z.shape)

model=torch.load(f'model_rep/1nn_den2cor/h_{h_max}_150k_augmentation_unet_periodic_den2cor_[40]_hc_5_ks_2_ps_1_nconv_0_nblock',map_location='cpu')
model.eval()

xx_ml=model(z_torch).detach().numpy()

dxx=np.sqrt(np.average((xx-xx_ml)**2)/np.average((xx)**2))
    
print(dxx)
#%% PART II(a): accuracy analysis
for i in range(1):
    plt.title('comparison correlation')
    plt.plot(xx[i,10,:])
    plt.plot(xx_ml[i,10,:])
    plt.show()
    plt.title('magn z')
    plt.plot(z[i,:])
    plt.show()

    #accuracy measure L2
    plt.imshow(xx[i])
    plt.colorbar()
    plt.show()
    plt.imshow(xx_ml[i])
    plt.colorbar()
    plt.show()
dxx=np.sqrt(np.average((xx-xx_ml)**2)/np.average((xx)**2))
print(dxx)



#%% PART II(b): accuracy analysis of the average value
xx_ml=np.average(xx_ml,axis=0)
xx=np.average(xx,axis=0)


plt.plot(xx[10,:])
plt.plot(xx_ml[10,:])
plt.show()
plt.plot(z[0,:])
plt.show()

#accuracy measure L2
plt.imshow(xx)
plt.colorbar()
plt.show()
plt.imshow(xx_ml)
plt.colorbar()
plt.show()
dxx=np.sqrt(np.average((xx-xx_ml)**2)/np.average((xx)**2))
print(dxx)
    
    






# %%
