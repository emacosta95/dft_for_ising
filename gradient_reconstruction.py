#%%
import time
from ast import increment_lineno
from numpy.lib.mixins import _inplace_binary_method
import torch
import torch.nn as nn
import numpy as np
from src.training.models import Energy_unet
from src.training.utils import initial_ensamble_random
from tqdm.notebook import tqdm, trange
import matplotlib.pyplot as plt
from scipy import fft, ifft
import random
from src.training.utils_analysis import test_models, test_models_unet
from src.training.utils_analysis import dataloader, ResultsAnalysis

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")


#%%
    
def parallel_nambu_diagonalization_ising_model(nbatch,l:int,j_coupling:float,hs:np.array,device:str,pbc:bool):
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

    n_dataset=hs.shape[0]

    batch=int(n_dataset/nbatch)
    #uniform means h_ave=0
    hs=torch.tensor(hs,dtype=torch.double,device=device)

    #obc
    j_vec=j_coupling*torch.ones(l,device=device)
    # the 0-th component is null in OBC
    j_vec_l=j_vec.clone()
    if not(pbc):
        j_vec_l[0]=0
    if pbc:
        j_vec_l[0]=-1*j_vec_l[0]
    

    # the l-th component is null in OBC
    j_vec_r=j_vec.clone()
    if not(pbc):
        j_vec_r[-1]=0
    if pbc:
        j_vec_r[-1]=-1*j_vec_r[-1]

    # create the nambu matrix

    # create the j matrix in the nearest neighbourhood case
    j_l=torch.einsum('ij,j->ij',torch.eye(l,device=device),j_vec_l)
    j_l=torch.roll(j_l,shifts=-1,dims=1)
    j_r=torch.einsum('ij,j->ij',torch.eye(l,device=device),j_vec_r)
    j_r=torch.roll(j_r,shifts=1,dims=1)
    # the coupling part for a
    j=-0.5*(j_r+j_l)
    # the coupling part for b
    j_b=-0.5*(j_r-j_l)
    # the b matrix of the nambu matrix
    b=j_b

    for i in trange(nbatch):
        # the external field
        h=hs[i*batch:(i+1)*batch]
        h_matrix=torch.einsum('ij,aj->aij',torch.eye(l,device=device),h)
        # the a matrix of the nambu matrix
        a=j+h_matrix

        # create the nambu matrix
        h_nambu=torch.zeros((batch,2*l,2*l),device=device)
        h_nambu[:,:l,:l]=a
        h_nambu[:,:l,l:]=b
        h_nambu[:,l:,:l]= -1*torch.conj(b)
        h_nambu[:,l:,l:]= -1*torch.conj(a)

        e,w=torch.linalg.eigh(h_nambu)

        # the v coefficients
        v=w.clone()[:,l:,:l]

        u=w.clone()[:,:l,:l]
        #compute the correlation sigma_x sigma_x
        c_vv=torch.einsum('anl,aml->anm',v,torch.conj(v))
        c_uu=torch.einsum('anl,aml->anm',u,torch.conj(u))        
        c_vu=torch.einsum('anl,aml->anm',v,torch.conj(u))
        c_uv=torch.einsum('anl,aml->anm',u,torch.conj(v))
        c=c_vv+c_vu-c_uu-c_uv
        
        s_z=1-2*torch.einsum('aik,aik->ai',v,torch.conj(v))
        s_z_different=torch.einsum('aik,aik->ai',u,torch.conj(u))-torch.einsum('aik,aik->ai',v,torch.conj(v))

        density_f=c[:,np.arange(l),(np.arange(l)+1)% l]
        density_f[:,-1]=-1*density_f[:,-1]

        e_0=torch.sum(e[:,0:l],dim=-1)/l
        f=e_0-torch.mean(h*s_z,dim=-1)
        
        if i==0:
            magn_z=s_z
            magn_z_diff=s_z_different
            e_tot=e_0
            f_tot=f
            tot_density_f=density_f
        else:
            magn_z=np.append(magn_z,s_z,axis=0)
            magn_z_diff=np.append(magn_z_diff,s_z_different,axis=0)
            e_tot=np.append(e_tot,e_0)
            f_tot=np.append(f_tot,f)                   
            tot_density_f=np.append(tot_density_f,density_f,axis=0)                    

    return hs,magn_z,magn_z_diff,f_tot,tot_density_f,e_tot




#%% #ANALYSIS IN THE STANDARD FORM 
ndata=100
h_max=[2.7]*8
n_dataset=[15000]*len(h_max)
ls=[16,32,64,128,256,512,1024,2048]

activation='_gelu'
models_name=[]
data_path=[]
ms={}
vs={}
models={}
for i in range(len(h_max)):
    path_name=f'data/dataset_1nn/valid_unet_periodic_{ls[i]}_l_{h_max[i]}_h_{n_dataset[i]}_n.npz'
    ms[ls[i]]=np.load(path_name)['density']
    vs[ls[i]]=np.load(path_name)['potential']
    models[ls[i]]=(torch.load(f'model_rep/1nn_ising/h_2.7_150k_augmentation_unet_periodic_1nn_model_cnn_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock',map_location='cpu'))


# %% THe CASE WITH MIXED DATASET

h_max=[1.0,2.7,3.0,4.5,]
n_dataset=[15000]*len(h_max)
l=128
ls=[l]*len(h_max)

activation='_gelu'
models_name=[]
data_path=[]
ms={}
vs={}
models={}
for i in range(len(h_max)):
    path_name=f'data/unet_dataset/valid_unet_periodic_{ls[i]}_l_{h_max[i]}_h_{n_dataset[i]}_n.npz'
    ms[h_max[i]]=np.load(path_name)['density']
    vs[h_max[i]]=np.load(path_name)['potential']
    models[h_max[i]]=(torch.load(f'model_rep/unet/h_interval_1-8_augmentation_unet_periodic_ising_cnn_gelu_[30, 60, 120]_hc_9_ks_2_ps_3_nconv_0_nblock',map_location='cpu'))



#%% #ANALYSIS OF THE NONGRIFFITH PHASE WITH MIXED DATASET
l=64
ndata=1000
h_max=[4.5,]
n_dataset=[15000]*len(h_max)
ls=[l]*len(h_max)

activation='_gelu'
models_name=[]
data_path=[]
ms={}
vs={}
models={}
for i in range(len(h_max)):
    path_name=f'data/unet_dataset/valid_periodic_no_griffith_phase_{ls[i]}_l_{h_max[i]}_h_15000_n.npz'
    ms[h_max[i]]=np.load(path_name)['density']
    vs[h_max[i]]=np.load(path_name)['potential']
    models[h_max[i]]=(torch.load(f'model_rep/unet/h_interval_1-8_augmentation_unet_periodic_ising_cnn_gelu_[30, 60, 120]_hc_9_ks_2_ps_3_nconv_0_nblock',map_location='cpu'))

#%% #ANALYSIS OF THE NONGRIFFITH PHASE WITH MIXED DATASET MODEL
l=64
ndata=1000
h_max=[4.5,]
n_dataset=[15000]*len(h_max)
ls=[l]*len(h_max)

activation='_gelu'
models_name=[]
data_path=[]
ms={}
vs={}
models={}
for i in range(len(h_max)):
    path_name=f'data/unet_dataset/valid_periodic_no_griffith_phase_{ls[i]}_l_{h_max[i]}_h_15000_n.npz'
    ms[h_max[i]]=np.load(path_name)['density']
    vs[h_max[i]]=np.load(path_name)['potential']
    models[h_max[i]]=(torch.load(f'model_rep/unet/h_interval_1-8_augmentation_unet_periodic_ising_cnn_gelu_[30, 60, 120]_hc_9_ks_2_ps_3_nconv_0_nblock',map_location='cpu'))


#%% #THE 2NN case
l=16
ndata=100
h_max=[2.71]
n_dataset=[ndata]*len(h_max)
ls=[l]*len(h_max)

activation='_gelu'
models_name=[]
data_path=[]
ms={}
vs={}
models={}
for i in range(len(h_max)):
    path_name=f'data/2nn_dataset/test_unet_periodic_{ls[i]}_l_{h_max[i]}_h_{n_dataset[i]}_n.npz'
    ms[h_max[i]]=np.load(path_name)['density']
    vs[h_max[i]]=np.load(path_name)['potential']
    models[h_max[i]]=(torch.load(f'model_rep/2nn_unet/h_{h_max[i]}_120k_augmentation_unet_periodic_2nn_model_cnn_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock',map_location='cpu'))


#%% #THE DMRG CASE
l=16
ndata=100
h_max=[2.71]
n_dataset=[ndata]*len(h_max)
ls=[l]*len(h_max)

activation='_gelu'
models_name=[]
data_path=[]
ms={}
vs={}
models={}
for i in range(len(h_max)):
    path_name=f'data/2nn_dataset/l_{l}_h_2.71_ndata_10.npz'
    ms[h_max[i]]=np.load(path_name)['density']
    vs[h_max[i]]=np.load(path_name)['potential']
    models[h_max[i]]=(torch.load(f'model_rep/2nn_unet/h_{h_max[i]}_120k_augmentation_unet_periodic_2nn_model_cnn_[20, 40]_hc_5_ks_2_ps_2_nconv_0_nblock',map_location='cpu'))



#%% Gradient accuracy
g_acc=[]
pseudo_pot={}
v_acc={}
for i,h in enumerate(h_max):
    x=ms[ls[i]][:ndata]    
    x=torch.tensor(x,dtype=torch.double)
    x.requires_grad_(True)
    f=torch.mean(models[ls[i]](x),dim=-1)
    #print(f.shape)
    f.backward(torch.ones_like(f))
    with torch.no_grad():
        grad = x.grad
        grad=-ls[i]*grad.detach().numpy()
        pseudo_pot[ls[i]]=grad
    #print(grad.shape)
    v_acc[ls[i]]=np.sqrt(np.average((grad-vs[ls[i]][:ndata])**2,axis=-1))/np.sqrt(np.average((vs[ls[i]][:ndata])**2,axis=-1))
    g_acc.append(np.sqrt(np.average((grad-vs[ls[i]][:ndata])**2,axis=-1))/np.sqrt(np.average((vs[ls[i]][:ndata])**2,axis=-1)))
#%%
g_acc=[np.average(g) for g in g_acc]
plt.plot(ls,g_acc)
plt.show()
#%%
h=64
x=ms[h][0:1]

for i in range(0,10):
    plt.plot(pseudo_pot[h][i],label='ml grad')
    plt.plot(vs[h][i],label='pot')
    plt.show()



# %% DIAGONALIZATION WITH PSEUDO POTENTIALS in Ising 1nn
nbatch=10
j_coupling=-1

pseudo_m={}
m_acc={}
for l in ls:
    _,m,_,f,fm,e=parallel_nambu_diagonalization_ising_model(nbatch=nbatch,l=l,j_coupling=j_coupling,hs=pseudo_pot[l],device='cpu',pbc=True)
    pseudo_m[l]=m
    m_acc[l]=np.sqrt(np.average((m-ms[l][:ndata])**2,axis=-1))/np.sqrt(np.average((ms[l][:ndata])**2,axis=-1))
    
#%% DIAGONALIZATION WITH PSEUDO POTENTIALS in Ising 2nn
from test_quspin_2nn import transverse_magnetization

nbatch=1
j_coupling=-1
m_acc={}
pseudo_m={}
#for h in h_max:
#CONTINUA DA QUA
   
#%% Measure of NUV-representability

measure={}
for h in ls:
    measure[h]=(1-(m_acc[h]/v_acc[h])**2)
    plt.hist(measure[h],bins=100,label=h,density=True,)
plt.xlabel(r'$\mu$',fontsize=20)
plt.legend(fontsize=20)
plt.show()

for h in ls:
    for i in range(10):
        if measure[h][i]<0:
            print(measure[h][i])
            plt.plot(ms[h][i],label='m')
            plt.plot(pseudo_m[h][i],label='pseudo m')
            plt.legend()
            plt.show()

            plt.plot(vs[h][i],label='v')
            plt.plot(pseudo_pot[h][i],label='pseudo v')
            plt.legend()
            plt.show()




# %%

#%%
h=4.5
for i in range(10):
    plt.plot(eigs[h][i,:])
    plt.axhline(y=0,color='red')
    plt.show()

# %% MIN ENG AND COMPARISON WITH V AND M
h=4.5
x=ms[h]
for i in range(10):
    min_eigs=np.min(eigs[h],axis=-1)    
    plt.plot(vs[h][i],label='v')
    plt.plot(pseudo_pot[h][i],label=f'eff v eig={min_eigs[i]}')
    plt.legend()
    plt.show()

    plt.plot(x[i],label='m')
    plt.plot(m[i],label=f'eff m eig={min_eigs[i]}')
    plt.legend()
    plt.show()

# %% SCATTER PLOT V-PSEUDOV VS eig
t=np.linspace(0,0.8,100)
for h in h_max:
    min_eigs=np.min(eigs[h],axis=-1)
    min_eig=min_eigs[min_eigs>=0]
    v_shows=v_acc[h][min_eigs>=0]
    #plt.plot(t,0.8*np.exp(-180*t))
    plt.scatter(min_eig[:],v_shows[:],label=f'h={h}')
#plt.semilogx()
plt.legend()
plt.show()

# %% SCATTER PLOT V-PSEUDOV VS M-PSEUDOM
for h in h_max:
    plt.scatter(v_acc[h],m_acc[h],label=f'h={h}')
    plt.legend()
    plt.show()
# %% TRY TO UNDERSTAND THE INSTABILITIES IN A GD CASE
labels = ["unet 256",'unet 128','unet 64']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_4.5_h_15000_n.npz')['potential'],
#np.load('data/unet_dataset/valid_unet_periodic_512_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_256_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_128_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_4.5_h_15000_n.npz')['potential'],
#np.load('data/unet_dataset/valid_unet_periodic_32_l_4.5_h_15000_n.npz')['potential']
]
n_sample = [29,19,9]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    #["h_4.5_augmentation_ising_model_unet_gelu_1024_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
#    ["h_mixed_augmentation_ising_model_unet_gelu_512_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_4.5_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_4.5_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps"] * n_sample[1],
    ["h_4.5_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[2],
#    ["h_mixed_augmentation_ising_model_unet_gelu_32_size_3_layers_30_hc_ks_2_ps"] * n_sample[4],
    ]
text = [
    #[f"1024" for epoch in epochs[0]],
#    [f"512" for epoch in epochs[0]],
    [f"256" for epoch in epochs[0]],
    [f"128" for epoch in epochs[0]],
    [f"64" for epoch in epochs[0]],
#    [f"32" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [0,1,2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.01, 0.02)
range_n = (0, 0.5)
range_a=(-10,50)
bins = 50

result = ResultsAnalysis(
    n_sample=n_sample,
    n_instances=n_instances,
    n_ensambles=n_ensambles,
    epochs=epochs,
    diff_soglia=diff_soglia,
    models_name=models_name,
    text=text,
    variable_lr=variable_lr,
    early_stopping=early_stopping,
    lr=1,
    dx=14 / 256,
)


# Plot all the main results
result.plot_results(
    xticks=[r'$0$',r'$2.5k$',r'$5.0k$',r'$17.5k$',r'$300k$'],
    xposition=[0,2500,5000,7500,10000],
    yticks=yticks,
    position=epochs,
    xlabel="steps",
    labels=labels,
    title=None,
    loglog=False,
    linestyle=["--",':','-.','-','-','--'],
    marker=['o','*','1','2','2','o'],
    color= ['black','red','green','violet','black','grey'],
    symbol=['(a)','(b)'],
    symbolx=[8000,8000],
    symboly=[0.2,0.2],
)


# %%
idx=-1
jdx=-1
h=4.5

for i in range(100):
    if np.abs(result.min_eng[idx][jdx][i]-result.gs_eng[idx][jdx][i])>0.01:
        print(i)
        plt.plot(result.min_n[idx][jdx][i],label=f'eng={result.min_eng[idx][jdx][i]}')
        plt.plot(result.gs_n[idx][jdx][i])
        plt.legend()
        plt.show()
        plt.plot(vs[h][i],label='pot')
        plt.plot(pseudo_pot[h][i],label='pseudo pot')
        plt.legend()
        plt.show()
        plt.plot(ms[h][i],label='m')
        plt.plot(pseudo_m[h][i],label='pseudo m')
        plt.legend()
        plt.show()
        
# %%
