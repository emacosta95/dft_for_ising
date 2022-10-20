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

#%% first analysis --> SCALABILITY AT THE CRITICAL VALUE for the 1nn
n_sample=7
t_range=[49,49,49,199,199,799,799]
h_max=[4.5]*n_sample
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
            #if min_eng_t.shape[0]==min_eng.shape[-1]:
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
    print(gs_n.shape)        


# %% ANALYSIS OF THE CONVERGENCE AT DIFFERENT SIZES for the absolute value
fig=plt.figure(figsize=(10,10))
errors_e=[]
sigma_errors_e=[]
for l in ls:
    print(l)
    print(min_energy[l].shape)
    print(gs_energy[l].shape)
    e_av=np.average(np.abs(min_energy[l]-gs_energy[l])/np.abs(gs_energy[l]),axis=-1)
    plt.plot(e_av,label=f'l={l}',linewidth=3)
    errors_e.append(e_av[-1])
    sigma_errors_e.append(np.std(np.abs(min_energy[l]-gs_energy[l])/np.abs(gs_energy[l]),axis=-1)[-1])
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
plt.xticks([1000,5000,10000,15000,20000,25000,30000,60000,80000],[r'$10^3$',r'$5 \cdot 10^3$',r'$10^4$',r'$1.5 \cdot 10^4$',r'$2 \cdot 10^4$',r'$2.5 \cdot 10^4$',r'$3 \cdot 10^4$',r'$6 \cdot 10^4 $',r"$8 \cdot 10^4$"])
plt.yticks([0.05,0.01,0.005,0.001],[r'$5 \cdot 10^{-2}$',r'$10^{-2}$',r'$5 \cdot 10^{-3}$',r'$ 10^{-3}$'])
#plt.axhline(10**-5,color='red',linestyle='--',linewidth=2,label=r'$0.001 \%$ error')
plt.loglog()
plt.legend(fontsize=20)
plt.show()

# %% ANALYSIS OF THE CONVERGENCE AT DIFFERENT SIZES for the deviation from the gs energy
fig=plt.figure(figsize=(10,10))
for i,l in enumerate(ls):
    e_av=np.average((min_energy[l]-gs_energy[l])/np.abs(gs_energy[l]),axis=1)
    plt.plot(epochs[i],e_av,label=f'l={l}',linewidth=3)
plt.legend(fontsize=20)
plt.xlabel(r'$t$',fontsize=40)
plt.ylabel(r'$\langle\frac{\Delta e}{ e }\rangle$',fontsize=40)

plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=20,
            width=5,
        )
plt.xticks([0,10000,20000,30000,40000,50000,60000,80000],[r'$0$',r'$1 \cdot 10^4$',r'$2 \cdot 10^4$',r'$3 \cdot 10^4$',r'$4 \cdot 10^4$',r'$5 \cdot 10^4$',r'$6 \cdot 10^4$',r'$8 \cdot 10^4$'])
plt.yticks([0.01,0.005,0.001],[r'$10^{-2}$',r'$5 \cdot 10^{-3}$',r'$ 10^{-3}$'])
plt.axhline(-0.00,color='red',label='zero')
plt.axhline(-0.001,color='red',linestyle='--',label=r'$0.1 \%$ error')
#plt.semilogy()
plt.legend(fontsize=20)
plt.show()

# %% ANALYSIS OF THE CONVERGENCE AT DIFFERENT SIZES for the deviation from the gs transverse magnetization
fig=plt.figure(figsize=(10,10))
errors_n=[]
sigma_errors_n=[]
for i,l in enumerate(ls):
    dn_av=np.average((np.average(np.abs(min_density[l]-gs_density[l]),axis=-1)/np.average(np.abs(gs_density[l]),axis=-1)),axis=1)
    plt.plot(epochs[i],dn_av,label=f'l={l}',linewidth=3)
    errors_n.append(dn_av[-1])
    sigma_errors_n.append(np.std((np.average(np.abs(min_density[l]-gs_density[l]),axis=-1)/np.average(np.abs(gs_density[l]),axis=-1)),axis=1)[-1])
plt.legend(fontsize=20)
plt.xlabel(r'$t$',fontsize=40)
plt.ylabel(r'$\langle\frac{\left|\Delta z \right|}{\left| z \right|}\rangle$',fontsize=40)

plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=20,
            width=5,
        )
plt.xticks([0,10000,20000,30000,40000,50000,60000,80000],[r'$0$',r'$1 \cdot 10^4$',r'$2 \cdot 10^4$',r'$3 \cdot 10^4$',r'$4 \cdot 10^4$',r'$5 \cdot 10^4$',r'$6 \cdot 10^4$',r"$8 \cdot 10^4$"])
plt.yticks([0.1,0.05,0.01,0.001],[r"$10^{-1}$",r'$5 \cdot 10^{-2}$',r'$10^{-2}$',r'$ 10^{-3}$'])
plt.axhline(0.001,color='red',label=r'$0.1 \%$ error')
#plt.semilogy()
plt.legend(fontsize=20)
plt.show()



# ANALYSIS OF THE FINAL VALUE OF THE GD AND COMPARISON WITH THE SCALE AND THE NUVR ANALYSIS

#%% Histogram plot of the final energy values
t=-1
hist_l=[32,64,512]
range=(-0.004,0.004)
fill=['//','.','-']

plt.figure(figsize=(10,10))
for i,l in enumerate(hist_l):

    plt.hist((min_energy[l][t]-gs_energy[l][t])/gs_energy[l][t],label=f'l={l}',bins=40,range=range,alpha=0.5,fill=fill[i])
plt.legend(fontsize=20)
plt.xlabel(r'$\Delta e / e $',fontsize=30)
plt.ylabel('Counts',fontsize=30)
plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
#plt.xticks([0,-1*10**-3,-5*10**-3,10**-3,5*10**-3],[r"$0$",r"$ 10^{-3}$",r"$-5 \cdot 10^{-3}$",r"$ 10^{-3}$",r"$5 \cdot 10^{-3}$"])


plt.show()


#%% Histogram plot of the final magnetization values
t=-1
hist_l=[32,64,512]
range=(0,0.05)

plt.figure(figsize=(10,10))
for i,l in enumerate(hist_l):

    plt.hist(np.average(np.abs(min_density[l][t]-gs_density[l][t]),axis=-1)/np.average(np.abs(gs_density[l][t]),axis=-1),label=f'l={l}',bins=20,range=range)
plt.legend(fontsize=20)
plt.xlabel(r'$|\Delta z |/ |z| $',fontsize=30)
plt.ylabel('Counts',fontsize=30)
plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
#plt.xticks([0,-1*10**-4,-5*10**-5,10**-4,5*10**-5],[r"$0$",r"$ 10^{-4}$",r"$-5 \cdot 10^{-5}$",r"$ 10^{-4}$",r"$5 \cdot 10^{-5}$"])


plt.show()


#%% errors vs l
t=-1
plt.figure(figsize=(10,10))
plt.errorbar(ls,errors_e,yerr=sigma_errors_e,color='black',marker='o',linewidth=2)
plt.legend(fontsize=20)
plt.xlabel(r'$ l $',fontsize=60)
plt.ylabel(r'$ \langle \frac{\left| \Delta e \right|}{\left|e \right|} \rangle $',fontsize=60)
plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
plt.xticks([16,256,512,1024],[16,256,512,1024])
plt.show()


#%%
t=-1
plt.figure(figsize=(10,10))
plt.errorbar(ls,errors_n,yerr=sigma_errors_n,color='black',marker='o',linewidth=2)
plt.legend(fontsize=20)
plt.xlabel(r'$ l $',fontsize=60)
plt.ylabel(r'$ \langle \frac{\left| \Delta z \right|}{\left|z \right|} \rangle $',fontsize=60)
plt.tick_params(
            top=True,
            right=True,
            labeltop=False,
            labelright=False,
            direction="in",
            labelsize=30,
            width=5,
        )
plt.xticks([16,256,512,1024],[16,256,512,1024])
plt.show()


#%% Analysis h vs gradient accuracy
n_sample=3
h_max=[1.0,2.7,4.5]
ls=[32]*n_sample
n_instances=[100]*n_sample
epochs = [4900]*n_sample 
# model settings
name_session=[f'h_{h_max[i]}_150k_augmentation_1nn_model_unet_{ls[i]}_size_2_layers_20_hc_5_ks_2_ps' for i in range(n_sample)]
early_stopping=False
variational_lr=False
loglr=1

min_density={}
gs_density={}
min_energy={}
gs_energy={}
g_acc={}
z_acc={}

for r in range(n_sample):

    min_eng,gs_eng=dataloader('energy',session_name=name_session[r],n_instances=n_instances[r],lr=loglr,diff_soglia=1,epochs=epochs[r],early_stopping=False,variable_lr=False,n_ensambles=1)
    min_n,gs_n=dataloader('density',session_name=name_session[r],n_instances=n_instances[r],lr=loglr,diff_soglia=1,epochs=epochs[r],early_stopping=False,variable_lr=False,n_ensambles=1)
    
    min_eng=np.asarray(min_eng)
    gs_eng=np.asarray(gs_eng)
    min_n=np.asarray(min_n)
    gs_n=np.asarray(gs_n)
    print(gs_n.shape)
        
    model_name=f'1nn_ising/h_{h_max[r]}_150k_augmentation_unet_periodic_1nn_model_cnn_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock'
    model=torch.load('model_rep/'+model_name,map_location='cpu')
    model.eval()
    v=np.load(f'data/dataset_1nn/valid_unet_periodic_{ls[r]}_l_{h_max[r]}_h_15000_n.npz')['potential']
    print('v_shape=',v.shape)
    dg,dz=nuv_representability_check(model,z=gs_n,v=v[:gs_n.shape[0]],plot=True,gs_z=gs_n)
    
    
    min_energy[h_max[r]]=min_eng
    gs_energy[h_max[r]]=gs_eng
    min_density[h_max[r]]=min_n
    gs_density[h_max[r]]=gs_n
    g_acc[h_max[r]]=dg
    z_acc[h_max[r]]=dz
    print(gs_n.shape)

    

#%% Measure of the nuv 

dds=[]
for h in h_max:
    print(z_acc[h].shape)
    dd=np.average(np.abs(g_acc[h]/z_acc[h]))
    dds.append(dd)
plt.plot(h_max,dds)
plt.xlabel(r'$h_{max}$',fontsize=20)
plt.ylabel(r'$\frac{1}{N}\sum_j \frac{\sum_i \left| h^{(j)}_i - h^{(j)}_{pseudo,i}\right|}{\sum_i \left| z^{(j)}_i - z^{(j)}_{1,i}\right|}$',fontsize=20)








# CORRELATION MAP from density to correlation (TEST)
#%%
logc_l=[]
logc_ml_l=[]
#%% PART I: loading the data
batch=1000
nbatch=10
minibatch=int(batch/nbatch)
l=64
h_max=3.6
device='cpu'
torch.set_num_threads(3)
data=np.load(f'data/correlation_1nn_rebuilt/test_1nn_correlation_map_h_{h_max}_1000_l_{l}_pbc_j_1.0.npz')
z=data['density'][:batch]
xx=data['correlation'][:batch]
z_torch=torch.tensor(z,dtype=torch.double,device=device)
print(z.shape)
#model=torch.load(f'model_rep/1nn_den2cor/h_{h_max}_150k_unet_periodic_den2corRESNET_[40, 40, 40, 40]_hc_5_ks_1_ps_4_nconv_0_nblock',map_location='cpu')
model=torch.load(f'model_rep/1nn_den2cor/h_{h_max}_150k_unet_periodic_den2corLSTM_scalable_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock',map_location=device)

model.eval()


for i in trange(nbatch):
    print(i)
    x=model(z_torch[minibatch*i:minibatch*(i+1)]).cpu().detach().numpy()
    if i==0:
        xx_ml=x
    else:
        xx_ml=np.append(xx_ml,x,axis=0)    



dxx=np.sqrt(np.average((xx-xx_ml)**2)/np.average((xx)**2))
    
print(dxx)
#%% PART II(a): Distribution of ln(C)
print(xx.shape)
c=np.average(xx,axis=-1)
c_ml=np.average(xx_ml,axis=-1)
lnc=np.log(np.abs(c))
lnc_ml=np.log(np.abs(c_ml))
print(lnc.shape)
rs=[6,12,24]

lnc_av=np.average(lnc,axis=0)
lnc_ml=np.average(lnc_ml,axis=0)

plt.plot(np.sqrt(np.arange(lnc_av.shape[0])),lnc_av,label='exact')
plt.plot(np.sqrt(np.arange(lnc_av.shape[0])),lnc_ml,label='ml')
plt.legend()
plt.show()

logc_l.append(lnc_av)
logc_ml_l.append(lnc_ml)
# #comparison for each r
# for r in rs:
#     plt.hist(np.abs(lnc[:,r])/np.sqrt(r),bins=40,label='ln(C)',density=True,alpha=0.5)
#     plt.hist(np.abs(lnc_ml[:,r])/np.sqrt(r),bins=40,label='ln(C)_ml',density=True,alpha=0.5)
#     plt.legend()
#     plt.loglog()
#     plt.title(f'r={r}')
#     plt.show()

# # r behaviour of the exact and reconstructed distribution
# for r in rs:
#     plt.hist(np.abs(lnc[:,r])/np.sqrt(r),bins=40,label=f'r={r}',density=True,alpha=0.5)
#     # plt.hist(np.abs(lnc_ml[:,r])/np.sqrt(r),bins=40,label='ln(C)_ml',density=True,alpha=0.5)
#     plt.legend()
#     plt.loglog()
# plt.show()

# # r behaviour of the exact and reconstructed distribution
# for r in rs:
#     plt.hist(np.abs(lnc_ml[:,r])/np.sqrt(r),bins=40,label=f'r={r}',density=True,alpha=0.5)
#     # plt.hist(np.abs(lnc_ml[:,r])/np.sqrt(r),bins=40,label='ln(C)_ml',density=True,alpha=0.5)
#     plt.legend()
#     plt.loglog()
# plt.show()
#%%
from scipy import stats

ls=[7,15,31,63]
for i in range(4):
    plt.plot(np.sqrt(np.arange(ls[i])),logc_l[i],label=f'{ls[i]}')
plt.legend()
plt.show()

for i in range(4):
    plt.plot(np.sqrt(np.arange(ls[i])),logc_ml_l[i],label=f'{ls[i]}')
plt.legend()
plt.show()


slope_0, intercept, r_value, p_value, std_err = stats.linregress(np.sqrt(np.arange(ls[-1]))[7:35], logc_ml_l[-1][7:35])
print(slope_0,intercept,r_value)
slope_0, intercept, r_value, p_value, std_err = stats.linregress(np.sqrt(np.arange(ls[-1]))[7:35], logc_l[-1][7:35])
print(slope_0,intercept,r_value)


#%% PART II(b): accuracy analysis of the average value
xx_ml=np.average(xx_ml,axis=0)
xx=np.average(xx,axis=0)


plt.plot(xx[1,:])
plt.plot(xx_ml[1,:])
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

# for i in range(xx.shape[-1]):
#     plt.plot(xx[:,i],label='exact')
#     plt.plot(xx_ml[:,i],label='ml')
#     plt.legend()
#     plt.show()
 


g=np.average(xx,axis=-1)
g_ml=np.average(xx_ml,axis=-1)

#%%
x=np.arange(int(l/2)-1)

plt.plot(x,(g),label=f'g')
plt.plot((x),(g_ml),label=f'g_ml')
#plt.axvline(x=2)
#plt.axvline(x=45)
plt.legend()
plt.loglog()
plt.show()

nu=np.average((np.log(g[7:35]))/np.log(x[7:35]))
print(nu)

nu_ml=np.average((np.log(g_ml[7:35]))/np.log(x[7:35]))
print(nu_ml)






#%% compute the slope of the curve
from scipy import stats

slope_0, intercept, r_value, p_value, std_err = stats.linregress(np.log(x[2:45]), np.log(g)[2:45])

print(slope_0,intercept,r_value)

slope_1, intercept_1, r_value_1, p_value_1, std_err_1 = stats.linregress(np.log(x[2:45]), np.log(g_ml)[2:45])

print(slope_1,intercept,r_value)

#%%



#%%
for i in range(xx.shape[-1]):
    plt.plot(z[0])
    plt.plot(xx[0,i])
    plt.plot(xx_ml[0,i])
    plt.title(f'{i}-th component')
    plt.show()








# %% Den2Magn Analysis

#%% Testing the neural network by using DMRG dataset
h_max=1.8
ls=[16,32,64]
ns=[100,100,100]


xs_ml={}
xs={}
zs={}
for i in range(len(ls)):

    data=np.load(f'data/den2magn_dataset_1nn/test_unet_periodic_1nn_l_{ls[i]}_h_{h_max}_ndata_{ns[i]}.npz')
    # data=np.load('data/den2magn_dataset_1nn/train_unet_periodic_1nn_8_l_4.50_h_150000_n.npz')
    x=torch.tensor(data['magnetization_x'],dtype=torch.double)
    z=torch.tensor(data['density'],dtype=torch.double)
    model=torch.load(f'model_rep/1nn_den2magn/h_{h_max:.1f}_15k_unet_periodic_den2magn_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock',map_location='cpu')
    
    x_ml=model(z).detach().numpy()
    xs_ml[ls[i]]=np.abs(x_ml)
    xs[ls[i]]=np.abs(x.detach().numpy()) #we analyze 
    #xs_ml[ls[i]]=(x_ml)
    #xs[ls[i]]=(x.detach().numpy()) #we analyze 
    #x**2 instead of x
    zs[ls[i]]=z.detach().numpy()



# %% I part: have a look to the profiles
l=64

for i in range(10):
    plt.plot(xs_ml[l][i],label='ml')
    plt.plot(xs[l][i],label='exact')
    plt.legend()
    plt.xlabel('l',fontsize=20)
    plt.ylabel('x',fontsize=20)
    plt.show()
    

#metrics absolute relative error
dx= np.average( np.sqrt( np.average((xs[l]-xs_ml[l])**2,axis=-1))/np.sqrt(np.average((xs[l])**2,axis=-1))  )

print(dx)

#%% binder cumulator
u_ml=0.5*(3-np.average(xs_ml[l]**4,axis=-1)/np.average(xs_ml[l]**2,axis=-1)**2 )
u_ml_av=np.average(u_ml)
print(u_ml_av)

u=0.5*(3-np.average(xs[l]**4,axis=-1)/np.average(xs[l]**2,axis=-1)**2)
u_av=np.average(u)
print(u_av)

# %% average over the disorder

x_ml_ave=np.average(xs_ml[l],axis=0)
x_ave=np.average(xs[l],axis=0)

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
# %%
