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
from src.training.models import Energy




#%% Data periodic CNN l=64
labels = ['h=1','h=1.9','h=2.4','h=2.7','h=3.0','h=3.5']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.4_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.9_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.4_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.7_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.5_h_15000_n.npz')['potential']
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
activation='_gelu'
models_name = [
    ["h_1.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.4_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.9_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.4_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_3.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_3.5_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
]
text = [
    [f"h=1.0" for epoch in epochs[0]],
    [f"h=1.4" for epoch in epochs[0]],
    [f"h=1.9" for epoch in epochs[0]],
    [f"h=2.4" for epoch in epochs[0]],
    [f"h=2.7" for epoch in epochs[0]],
    [f"h=3.0" for epoch in epochs[0]],
#    [f"h=3.5" for epoch in epochs[0]],
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,-1]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.02)
range_n = (0, 0.1)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% Data periodic CNN gelu with Z^2 symmetry l=64
labels = ["h=2.7 z2 tr",'h=2.7','h=2.7 z2']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_2.7_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.7_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
activation=''
models_name = [
    ["h_2.7_augmentation_z2_traslation_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_300k_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_augmentation_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
]
text = [
    [f"h=2.7 z2" for epoch in epochs[0]],
    [f"h=2.7" for epoch in epochs[0]],
    [f"h=2.7 z2 trasl" for epoch in epochs[0]],
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% Data periodic CNN relu single dataset vs different dataset symmetry l=64
labels = ["mixed",'single','mixed 450k']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
models_name = [
    ["h_1.0_2.7_3.5_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_2.7_3.5_450k_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    
    ]
text = [
    [f"h=mixed" for epoch in epochs[0]],
    [f"h=single" for epoch in epochs[0]],
    [f"h=mixed 450k" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% Data periodic CNN relu single dataset vs different dataset symmetry l=64
labels = ["mixed h=1",'mixed Z2 h=1','mixed 450k h=1','h=1']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
models_name = [
    ["h_1.0_2.7_3.5_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_2.7_3.5_augmentation_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_2.7_3.5_450k_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    
    
    ]
text = [
    [f"h=mixed" for epoch in epochs[0]],
    [f"h=mixed augmentation" for epoch in epochs[0]],
    [f"h=mixed 450k" for epoch in epochs[0]],
    [f"h=1.0 " for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% Data periodic CNN relu mixed h with and without Z2 l=64
labels = ["mixed h=1",'mixed Z2 h=1','mixed 450k h=1','h=1']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
models_name = [
    ["h_1.0_2.7_3.5_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_2.7_3.5_augmentation_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_2.7_3.5_450k_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.0_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    
    
    ]
text = [
    [f"h=mixed" for epoch in epochs[0]],
    [f"h=mixed augmentation" for epoch in epochs[0]],
    [f"h=mixed 450k" for epoch in epochs[0]],
    [f"h=1.0 " for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]



#%% Data periodic CNN relu single dataset vs different dataset symmetry l=64
labels = ["gelu Z2+trasl 2.7",'gelu 2.7 500k','gelu Z2 2.7','h infinity augmentation']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
models_name = [
    ["h_2.7_augmentation_z2_traslation_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    #["h_2.7_500k_periodic_ising_model_cnn_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_500k_periodic_ising_model_cnn_gelu_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_augmentation_periodic_ising_model_cnn_gelu_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ['h_2.7_augmentation_z2_h_infinity_periodic_ising_model_cnn_gelu_64_size_30_hc_3_ks_2_ps']*n_sample
    ]
text = [
    [f"z2+trasl gelu" for epoch in epochs[0]],
    #[f"relu" for epoch in epochs[0]],
    [f"gelu" for epoch in epochs[0]],
    [f"z2 gelu " for epoch in epochs[0]],
    [f"h infinity " for epoch in epochs[0]], 
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% Data periodic CNN gelu different samples l=64
labels = ["2.7",' 2.7 300k',' 2.7 500k']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
activation='_gelu'
models_name = [
    ["h_2.7_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_300k_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_500k_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ]
text = [
    [f"150k" for epoch in epochs[0]],
    [f"300k" for epoch in epochs[0]],
    [f"500k" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [-1,0,1]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%%
labels = ["3.0",'3.0 z2 h infinity','mixed h=3']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_3.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.0_h_15000_n.npz')['potential'],
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
activation='_gelu'
models_name = [
    ["h_3.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_3.0_augmentation_z2_h_infinity_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample, 
    ["h_1.0_2.7_3.5_augmentation_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,  
    ]
text = [
    [f"h=3" for epoch in epochs[0]],
    [f"h=3 augmentation z2" for epoch in epochs[0]],
    [f"h=3 mixed" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [-1,0,1]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.03)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% UNET ANALYSIS h=3.0
labels = ['unet 1024','unet 512',"unet 256",'unet 128','unet 64','unet 32']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/unet_dataset/valid_unet_periodic_1024_l_3.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_512_l_3.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_256_l_3.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_128_l_3.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_3.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_32_l_3.0_h_15000_n.npz')['potential']
]
n_sample = [69,49,49,19,11,11]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    ["h_3.0_augmentation_ising_model_unet_gelu_1024_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_3.0_augmentation_ising_model_unet_gelu_512_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_3.0_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps"] * (n_sample[1]),
    ["h_3.0_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps"] * n_sample[2],
    ["h_3.0_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[3],
    ["h_3.0_augmentation_ising_model_unet_gelu_32_size_3_layers_30_hc_ks_2_ps"] * n_sample[4],
    ]
text = [
    [f"1024" for epoch in epochs[0]],
    [f"512" for epoch in epochs[0]],
    [f"256" for epoch in epochs[0]],
    [f"128" for epoch in epochs[0]],
    [f"64" for epoch in epochs[0]],
    [f"32" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [3,4,5]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.006, 0.006)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]


#%% UNET ANALYSIS h=1.0
labels = ['unet 1024','unet 512',"unet 256",'unet 128','unet 64','unet 32']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_1.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_512_l_1.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_256_l_1.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_128_l_1.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_32_l_1.0_h_15000_n.npz')['potential']
]
n_sample = [69,49,39,19,10,10]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    ["h_1.0_augmentation_ising_model_unet_gelu_1024_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_1.0_augmentation_ising_model_unet_gelu_512_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_1.0_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps"] * (n_sample[1]),
    ["h_1.0_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps"] * n_sample[2],
    ["h_1.0_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[3],
    ["h_1.0_augmentation_ising_model_unet_gelu_32_size_3_layers_30_hc_ks_2_ps"] * n_sample[4],
    ]
text = [
    [f"1024" for epoch in epochs[0]],
    [f"512" for epoch in epochs[0]],
    [f"256" for epoch in epochs[0]],
    [f"128" for epoch in epochs[0]],
    [f"64" for epoch in epochs[0]],
    [f"32" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [-1,-2]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.0001, 0.0001)
range_n = (0, 0.01)
range_a=(-10,50)
bins = 50
#labels=['l=64',]


#%% UNET ANALYSIS h=e
labels = ['unet 1024','unet 512',"unet 256",'unet 128','unet 64','unet 32']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/unet_dataset/valid_unet_periodic_1024_l_2.7_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_512_l_2.7_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_256_l_2.7_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_128_l_2.7_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_2.7_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_32_l_2.7_h_15000_n.npz')['potential']
]
n_sample = [39,29,19,10,10]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    ["h_2.7_augmentation_ising_model_unet_gelu_1024_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_2.7_augmentation_ising_model_unet_gelu_512_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_2.7_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps"] * (n_sample[1]),
    ["h_2.7_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps"] * n_sample[2],
    ["h_2.7_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[3],
    ["h_2.7_augmentation_ising_model_unet_gelu_32_size_3_layers_30_hc_ks_2_ps"] * n_sample[4],
    ]
text = [
    [f"1024" for epoch in epochs[0]],
    [f"512" for epoch in epochs[0]],
    [f"256" for epoch in epochs[0]],
    [f"128" for epoch in epochs[0]],
    [f"64" for epoch in epochs[0]],
    [f"32" for epoch in epochs[0]],
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

range_eng = (-0.006, 0.006)
range_n = (0, 0.3)
range_a=(-10,50)
bins = 50
#labels=['l=64',]


#%% UNET ANALYSIS h=4.5
labels = ['unet 512',"unet 256",'unet 128','unet 64','unet 32']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 20000 for i in range(149)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_512_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_256_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_128_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_32_l_4.5_h_15000_n.npz')['potential']
]
n_sample = [299,149,99,49,10]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    #["h_4.5_augmentation_ising_model_unet_gelu_1024_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_4.5_augmentation_ising_model_unet_gelu_512_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_4.5_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps"] * (n_sample[1]),
    ["h_4.5_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps"] * n_sample[2],
    ["h_4.5_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[3],
    ["h_4.5_augmentation_ising_model_unet_gelu_32_size_3_layers_30_hc_ks_2_ps"] * n_sample[4],
    ]
text = [
    #[f"1024" for epoch in epochs[0]],
    [f"512" for epoch in epochs[0]],
    [f"256" for epoch in epochs[0]],
    [f"128" for epoch in epochs[0]],
    [f"64" for epoch in epochs[0]],
    [f"32" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [2,3,4]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.5, 0.5)
range_n = (0, 0.5)
range_a=(-10,50)
bins = 50
#labels=['l=64',]


#%% UNET ANALYSIS test
labels = ['mixed']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 20000 for i in range(4)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_4.5_h_15000_n.npz')['potential'],
]
n_sample = [9]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    ["h_interval_1-8_less_450k_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[0],
    ]
text = [
    #[f"1024" for epoch in epochs[0]],
    [f"no G points" for epoch in epochs[0]],

    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [0]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.5, 0.5)
range_n = (0, 0.5)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% UNET ANALYSIS h=4.5
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
    ["h_1.0_2.7_4.5_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps"] * (n_sample[0]),
    ["h_1.0_2.7_4.5_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps"] * n_sample[1],
    ["h_1.0_2.7_4.5_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"] * n_sample[2],
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
#labels=['l=64',]

#%% Unet analysis no griffith paramagnetic phase
labels = ['no griffith same dataset','different dataset']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 20000 for i in range(4)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_4.5_h_15000_n.npz')['potential'],
np.load('data/unet_dataset/valid_unet_periodic_64_l_4.5_h_15000_n.npz')['potential'],
]
n_sample = [9,9]
n_hc = len(labels)
n_instances = [[100] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    ["h_4.5_no_griffith_phase_same_dataset_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"]*n_sample[0],
    ["h_4.5_no_griffith_phase_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps"]*n_sample[1],
]
text = [
    #[f"1024" for epoch in epochs[0]],
    [f"no G points" for epoch in epochs[0]],
    [f"no G points with griffith dataset" for epoch in epochs[0]],

    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [0]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.5, 0.5)
range_n = (0, 0.5)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% DMRG TEST IN THE 2NN CASE
labels = ['L=16','L=64','L=128','L=256','L=512']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 1000 for i in range(5)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_4.5_h_15000_n.npz')['potential'],
np.load('data/2nn_dataset/l_16_h_2.71_ndata_10.npz')['potential'],
np.load('data/2nn_dataset/l_64_h_2.71_ndata_10.npz')['potential'],
np.load('data/2nn_dataset/l_128_h_2.71_ndata_10.npz')['potential'],
np.load('data/2nn_dataset/l_256_h_2.71_ndata_10.npz')['potential'],
np.load('data/2nn_dataset/l_512_h_2.71_ndata_10.npz')['potential'],
]
n_sample = [5 for i in range(len(labels))]
n_hc = len(labels)
n_instances = [[10] * n for n in n_sample] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10] * n for n in n_sample] 
models_name = [
    ["h_2.71_150k_augmentation_2nn_model_unet_dmrg_16_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
    ["h_2.71_150k_augmentation_2nn_model_unet_64_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
    ["h_2.71_150k_augmentation_2nn_model_unet_128_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
    ["h_2.71_150k_augmentation_2nn_model_unet_256_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
    ["h_2.71_150k_augmentation_2nn_model_unet_512_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
]
text = [
    #[f"1024" for epoch in epochs[0]],
    [f"L=16" for epoch in epochs[0]],
    [f"L=64" for epoch in epochs[0]],
    [f"L=128" for epoch in epochs[0]],
    [f"L=256" for epoch in epochs[0]],
    [f"L=512" for epoch in epochs[0]],
    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [0]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.5, 0.5)
range_n = (0, 0.5)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% DMRG TEST and COMPARISON
labels = ['dmrg dataset L=16','exact diagonalization dataset L=16']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 1000 for i in range(5)]
vs=[  #np.load('data/unet_dataset/valid_unet_periodic_1024_l_4.5_h_15000_n.npz')['potential'],
np.load('data/2nn_dataset/l_16_h_2.71_ndata_10.npz')['potential'],
np.load('data/2nn_dataset/test_unet_periodic_16_l_2.71_h_100_n.npz')['potential'],
]
n_sample = [5,5]
n_hc = len(labels)
n_instances = [[10]*n_sample[0],[10]*n_sample[1]] 
n_ensambles = [[1] * n for n in n_sample]
epochs = [[i * 1000 for i in range(n_sample[j])] for j in range(len(n_sample))] 
diff_soglia = [[10]*n_sample[0],[10]*n_sample[1]] 
models_name = [
    ["h_2.71_150k_augmentation_2nn_model_unet_dmrg_16_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
    ["h_2.71_150k_augmentation_2nn_model_unet_16_size_2_layers_20_hc_5_ks_2_ps"]*n_sample[0],
]
text = [
    #[f"1024" for epoch in epochs[0]],
    [f"dmrg dataset L=16" for epoch in epochs[0]],
    [f"exact diagonalization dataset L=16" for epoch in epochs[0]],

    ]
title = f"Gradient descent evolution"
variable_lr = [[False] *n for n in n_sample] 
early_stopping = [[False] * n for n in n_sample] 
# Histogram settings
idx = [0]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.5, 0.5)
range_n = (0, 0.5)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

#%% Initialize the analysis object
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


#%% Plot all the main results
result.plot_results(
    xticks=[r'$0$',r'$500$',r'$2.5k$',r'$4k$'],
    xposition=[0,500,2500,4000],
    yticks=yticks,
    position=epochs,
    xlabel="steps",
    labels=labels,
    title=None,
    loglog=True,
    linestyle=["--",':','-.','-','-','--'],
    marker=['o','*','1','2','2','o'],
    color= ['black','red','green','violet','black','grey'],
    symbol=['(a)','(b)'],
    symbolx=[8000,8000],
    symboly=[0.2,0.2],
)
# %% Plot single samples
idx = [0]
jdx = [-1]
result.plot_samples(
    idx=idx, jdx=jdx, n_samples=10,title=None, l=128, alpha=0.7,v=vs[idx[0]],letterx=12.5,lettery=1.2,letter='(b)'
)

# %% Histogram plots

result.histogram_plot(
    idx=idx,
    jdx=jdx,
    bins=bins,
    title=title,
    density=density,
    range_eng=range_eng,
    range_n=range_n,
    range_a=range_a,
    alpha=alpha,
    hatch=hatch,
    color=color,
    fill=fill,
    logx=False,
    labels=labels,
    textx=[0.19,0.012,47],
    texty=[80,120,170],
    i_pred=0,
    j_pred=0
)

# %% Analysis of F for the m_min and m_gs
idx=0
jdx=-1

#fix the magnetization for the study
gs_n=pt.tensor(result.gs_n[idx][jdx],dtype=pt.double)
min_n=pt.tensor(result.min_n[idx][jdx],dtype=pt.double)
#model=pt.load('model_rep/'+models_name[idx][jdx],map_location='cpu')
model=pt.load('model_rep/unet/h_interval_1-8_augmentation_unet_periodic_ising_cnn_gelu_[30, 60, 120]_hc_9_ks_2_ps_3_nconv_0_nblock',map_location='cpu')
#model_2=pt.load('model_rep/'+models_name[-1][jdx],map_location='cpu')
model.eval()
#model_2.eval()
print(gs_n.shape,min_n.shape)
f_gs=model(gs_n).detach().numpy()
f_min=model(min_n).detach().numpy()

f_gs=np.average(f_gs,axis=-1)
f_min=np.average(f_min,axis=-1)

gs_ext=np.average(result.gs_n[idx][jdx]*vs[idx][0:result.gs_n[idx][jdx].shape[0]],axis=1)
min_ext=np.average(result.min_n[idx][jdx]*vs[idx][0:result.gs_n[idx][jdx].shape[0]],axis=1)

plt.hist((gs_ext-min_ext)/gs_ext,bins=20,label='Dext')
plt.hist((f_gs-f_min)/f_gs,bins=20,label='Df')
plt.legend()
plt.show()


#%%
for i in range(100):
    fig=plt.figure(figsize=(20,20))
    plt.plot(result.gs_n[idx][jdx][i],label=f'f_gs-f_min={f_gs[i]-f_min[i]:.3f}',color='black')
    plt.plot(result.min_n[idx][jdx][i],label=f'min ext_gs-ext_min={gs_ext[i]-min_ext[i]:.3f}',linestyle=':',alpha=1,color='red')
    plt.plot(vs[idx][i])
    plt.axhline(y=0,linestyle='--',color='red')
    plt.axhline(y=-1,linestyle='--',color='red')  
    plt.legend()
    plt.show()

#%% external potential
gs_ext=np.average(result.gs_n[idx][jdx]*vs[idx][0:result.gs_n[idx][jdx].shape[0]],axis=1)
min_ext=np.average(result.min_n[idx][jdx]*vs[idx][0:result.gs_n[idx][jdx].shape[0]],axis=1)

plt.hist(gs_ext-min_ext,bins=20)
plt.xlabel('gs_ext-min_ext')
plt.show()

for i in range(100):
    if gs_ext[i]+f_gs[i]-f_min[i]-min_ext[i]< -1000:
        print(i)
        count=np.sum(vs[idx][i,vs[idx][i]<=1])
        print(count/vs[idx].shape[-1])
        fig=plt.figure(figsize=(20,20))
        plt.plot(result.gs_n[idx][jdx][i],label=f'ext_gs-ext_min={gs_ext[i]+f_gs[i]-f_min[i]-min_ext[i]:.3f}')
        plt.plot(result.min_n[idx][jdx][i],label='min')
        plt.plot(vs[idx][i])
        plt.axhline(y=-1,linestyle='--',color='red')
        plt.axhline(y=0,linestyle='--',color='red')
        plt.legend()
        plt.show()


#%% Analysis of the potential

de=np.abs(result.gs_eng[0][-1]-result.min_eng[0][-1])
v=vs[0][0:50]
v_pathological=v[de>0.01]
print(v_pathological.shape)

for i in range(v_pathological.shape[0]):
    plt.plot(v_pathological[i],label='pathological')
    plt.plot(v[i],label='pot')
    plt.legend()
    plt.show()

plt.plot()
plt.plot()

#%%
n_gs=pt.tensor(result.gs_n[0][-1],dtype=pt.double)
n_min=pt.tensor(result.min_n[0][-1],dtype=pt.double)

model=pt.load('model_rep/unet/h_interval_1-8_augmentation_unet_periodic_ising_cnn_gelu_[30, 60, 120]_hc_9_ks_2_ps_3_nconv_0_nblock',map_location='cpu')
model.eval()


f_gs=model(n_gs).detach().numpy()
f_min=model(n_min).detach().numpy()

for i in range(100):
    plt.plot(f_gs[i],label=f'gs {result.gs_eng[0][-1][i]:.2f}')
    plt.plot(f_min[i],label=f'min {result.min_eng[0][-1][i]:.2f}')
    plt.legend()
    plt.show()












#%%
from scipy.fft import fft, ifft
dv_pathological=np.gradient(v_pathological,axis=-1)
dv=np.gradient(v,axis=-1)

fft_v=fft(v)
fft_v_pathological=fft(v_pathological)

for i in range(8):
    plt.plot(fft_v[i],label='ok')
    plt.plot(fft_v_pathological[i],label='not ok')
    plt.legend()
    plt.show()
    plt.plot(np.imag(fft_v[i]),label='ok')
    plt.plot(np.imag(fft_v_pathological[i]),label='not ok')
    plt.legend()
    plt.show()














#%% Counting

counts=[]
for i in range(100):
        count=np.sum(vs[idx][i,vs[idx][i]<=0.01])
        counts.append(count/vs[idx].shape[-1])

plt.scatter(counts,result.gs_eng[idx][jdx]-result.min_eng[idx][jdx])
plt.show()


#%% Z2 symmetry

f_gs=model(gs_n).detach().numpy().reshape(-1)
f_gs_z2=model(-1*gs_n).detach().numpy().reshape(-1)

plt.hist(f_gs-f_gs_z2,bins=50)
plt.show()

#%% Traslation symmetry
shift=1
f_gs=model(gs_n).detach().numpy().reshape(-1)
f_gs_z2=model(pt.roll(gs_n,shifts=shift,dims=-1)).detach().numpy().reshape(-1)

plt.hist(f_gs-f_gs_z2,bins=50)
plt.show()


# %% Test the functional value with +- 1 sigma_z

f_1=model(pt.ones((1,1,64),dtype=pt.double)).detach().numpy().reshape(-1)
print(f_1.item())
           
f_1=model(-1*pt.ones((1,1,64),dtype=pt.double)).detach().numpy().reshape(-1)
print(f_1.item())


# %% We should check the correlation between the -1 values and the errors
idxs=[]
not_idxs=[]
for i in range(result.gs_n[idx][jdx].shape[0]):
    m=result.min_n[idx][jdx][i][result.min_n[idx][jdx][i]<-1+10**-3]
    if m.shape[0]!=0:
        idxs.append(i)
    else:
        not_idxs.append(i)
idxs=np.asarray(idxs)
not_idx=np.asarray(not_idxs)
jdxs=[]
for i in range(result.gs_n[idx][jdx].shape[0]):
    r=result.gs_n[idx][jdx][i][result.gs_n[idx][jdx][i]<-1+10**-3]
    if r.shape[0]!=0:
        print(result.gs_n[idx][jdx][i])
        jdxs.append(i)
jdxs=np.asarray(jdxs)
#%%
print(idxs.shape)
print(jdxs.shape)

for i in idxs:
    plt.plot(result.gs_n[idx][jdx][i],label='gs')
    plt.plot(result.min_n[idx][jdx][i],label='min')
    plt.plot(vs[idx][i])
    plt.axhline(y=-1)
    plt.axhline(y=0)
    plt.title('pathological',fontsize=20)
    plt.show()

for i in not_idxs:
    plt.plot(result.gs_n[idx][jdx][i],label='gs')
    plt.plot(result.min_n[idx][jdx][i],label='min')
    plt.plot(vs[idx][i])
    plt.axhline(y=-1)
    plt.axhline(y=0)
    plt.title('not pathological',fontsize=20)
    plt.show()
# %%
plt.hist(result.gs_eng[idx][jdx][idxs]-result.min_eng[idx][jdx][idxs],bins=20,density=True)
plt.hist(result.gs_eng[idx][jdx]-result.min_eng[idx][jdx],bins=20,density=True)
plt.show()



#%% Entanglement entropy for a single particle
n_min={}
n_gs={}
for i in range(len(result.min_n)):
    n_min[result.min_n[i][-1].shape[-1]]=result.min_n[i][-1]
    n_gs[result.gs_n[i][-1].shape[-1]]=result.gs_n[i][-1]
#%%
print(n_min[32].shape)

ls=[1024,512,256,128,64,32]

ent_min=[]
ent_gs=[]
for l in ls:
    nmin=(1-n_min[l])/2
    ent_min.append(np.average(nmin*np.log(nmin+10**-10)))

    ngs=(1-n_gs[l])/2
    ent_gs.append(np.average(ngs*np.log(ngs+10**-10)))

plt.plot(ls,ent_min)
plt.plot(ls,ent_gs)
plt.show()

#%% Errors vs scalability
ls=[1024,512,256,128,64,32]
print(len(result.list_de))
print(len(result.list_de[0]))
for dn in result.list_dn:
    plt.plot(dn)
plt.show()

des_h_e=[result.list_de[i][-1] for i in range(len(ls))]
dns_h_e=[result.list_dn[i][-1] for i in range(len(ls))]

plt.plot(ls,des_h_e)
plt.show()

plt.plot(ls,dns_h_e)
plt.show()


#%%

plt.plot(ls,dns_h_3,label='h=3',linestyle='--',marker='o',color='blue',linewidth=3)
plt.plot(ls,dns_h_e,label='h=e',linestyle=':',marker='*',color='red',linewidth=3)
plt.plot(ls,dns_h_1,label='h=1',marker='1',color='black',linewidth=3)
plt.legend(fontsize=10)
plt.xlabel(r'$l$',fontsize=30)
plt.ylabel(r'$| \Delta m | / |m |$',fontsize=30)
plt.show()

plt.plot(ls,des_h_3,label='h=3',linestyle='--',marker='o',color='blue',linewidth=3)
plt.plot(ls,des_h_e,label='h=e',linestyle='--',marker='*',color='red',linewidth=3)
plt.plot(ls,des_h_1,label='h=1',linestyle='--',marker='1',color='black',linewidth=3)
plt.legend(fontsize=10)
plt.xlabel(r'$l$',fontsize=30)
plt.ylabel(r'$| \Delta e | / |e |$',fontsize=30)
plt.show()







# %% Analysis of the gd errors for Gelu and ReLU

# Data periodic CNN l=64
labels = ['h=1',"h=1.4",'h=1.9','h=2.4','h=2.7','h=3.0']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.4_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.9_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.4_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.7_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.5_h_15000_n.npz')['potential']
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
activation=''
models_name = [
    ["h_1.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.4_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.9_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.4_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_3.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
#    ["h_3.5_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
]
text = [
    [f"h=1.0" for epoch in epochs[0]],
    [f"h=1.4" for epoch in epochs[0]],
    [f"h=1.9" for epoch in epochs[0]],
    [f"h=2.4" for epoch in epochs[0]],
    [f"h=2.7" for epoch in epochs[0]],
    [f"h=3.0" for epoch in epochs[0]],
#    [f"h=3.5" for epoch in epochs[0]],
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,-1]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.02)
range_n = (0, 0.1)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

re = ResultsAnalysis(
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

re.plot_results(
    xticks=[r'$0$',r'$2.5k$',r'$5k$',r'$7.5k$',r'$10k$'],
    xposition=[0,2500,5000,7500,10000],
    yticks=yticks,
    position=epochs[0],
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



# Data periodic CNN l=64
labels = ['h=1',"h=1.4",'h=1.9','h=2.4','h=2.7','h=3.0']
yticks = {
    "de": None,
    "devde": None,
    "dn": None,
    "devdn": None,
}
xticks = [i * 2000 for i in range(6)]
vs=[ np.load('data/dataset/valid_sequential_periodic_64_l_1.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.4_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_1.9_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.4_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_2.7_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.0_h_15000_n.npz')['potential'],
np.load('data/dataset/valid_sequential_periodic_64_l_3.5_h_15000_n.npz')['potential']
]
n_sample = 11
n_hc = len(labels)
n_instances = [[100] * n_sample] * n_hc
n_ensambles = [[1] * n_sample] * n_hc
epochs = [[i * 1000 for i in range(n_sample)]] * n_hc
diff_soglia = [[10] * n_sample] * n_hc
activation='_gelu'
models_name = [
    ["h_1.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.4_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_1.9_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.4_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_2.7_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
    ["h_3.0_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
 #   ["h_3.5_periodic_ising_model_cnn"+activation+"_64_size_30_hc_3_ks_2_ps"] * n_sample,
]
text = [
    [f"h=1.0" for epoch in epochs[0]],
    [f"h=1.4" for epoch in epochs[0]],
    [f"h=1.9" for epoch in epochs[0]],
    [f"h=2.4" for epoch in epochs[0]],
    [f"h=2.7" for epoch in epochs[0]],
    [f"h=3.0" for epoch in epochs[0]],
#    [f"h=3.5" for epoch in epochs[0]],
]
title = f"Gradient descent evolution"
variable_lr = [[False] * n_sample] * n_hc
early_stopping = [[False] * n_sample] * n_hc
n_sample = [n_sample] * n_hc

# Histogram settings
idx = [0,1,-1]
jdx = [-1]
hatch = [ ["."], [None],[ "//"] ]
color = [ ["black"], ["red"], ["blue"] ]
fill = [[False], [True],[False]]
alpha=[[1],[0.5],[1]]
density = False

range_eng = (-0.02, 0.02)
range_n = (0, 0.1)
range_a=(-10,50)
bins = 50
#labels=['l=64',]

resultGelu = ResultsAnalysis(
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

resultGelu.plot_results(
    xticks=[r'$0$',r'$2.5k$',r'$5k$',r'$7.5k$',r'$10k$'],
    xposition=[0,2500,5000,7500,10000],
    yticks=yticks,
    position=epochs[0],
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
# %% Plot the comparison
#hs=[1.,1.4,1.9,2.4,2.7,3.0]

dn_relu=np.asarray(re.list_dn)
dn_gelu=np.asarray(resultGelu.list_dn)
dn_test=np.asarray(result.list_dn)
devdn_test=np.asarray(result.list_devdn)
plt.errorbar(x=np.arange(len(dn_test)),y=dn_test[:,-1],yerr=devdn_test[:,-1],label='augmentation gelu critical',linestyle='--',marker='o',linewidth=3,color='red')
plt.axhline(dn_gelu[0,-1],label='gelu',marker='*',linewidth=3)
plt.xlabel(r'$diff mode$',fontsize=20)
plt.ylabel(r'$|\Delta m|$',fontsize=20)
plt.legend(fontsize=20)
plt.show()

# %% Evolution behaviour

data_unet_128=np.load('gradient_descent_ensamble_numpy/history_h_4.5_augmentation_ising_model_unet_gelu_128_size_3_layers_30_hc_ks_2_ps_number_istances_100_n_ensamble_1_different_initial_epochs_19000_lr_1.npz')
data_unet_256=np.load('gradient_descent_ensamble_numpy/history_h_4.5_augmentation_ising_model_unet_gelu_256_size_3_layers_30_hc_ks_2_ps_number_istances_100_n_ensamble_1_different_initial_epochs_29000_lr_1.npz')
data_unet_64=np.load('gradient_descent_ensamble_numpy/history_h_4.5_augmentation_ising_model_unet_gelu_64_size_3_layers_30_hc_ks_2_ps_number_istances_100_n_ensamble_1_different_initial_epochs_9000_lr_1.npz')

print(data_unet_128['history'].shape)
hist_256=data_unet_256['history']
hist_64=data_unet_64['history']
hist_128=data_unet_128['history']
#%%

plt.plot(hist_256[:4000],label='256')
plt.plot(hist_64[:4000],label='64')
plt.plot(hist_128[:4000],label='128')
plt.legend()
plt.show()
plt.plot(vs[2][i])
plt.axhline(y=0)
#plt.plot(result.min_n[1][-1][i],label='450k')
plt.plot(result.min_n[2][-1][i],label='128')
plt.plot(result.gs_n[2][-1][i],label='gs')
plt.legend()
plt.show()

# %%
idx=-2
jdx=-1
for i in range(100):
    if result.min_eng[idx][jdx][i]>0:
        print(f'gs_eng={result.gs_eng[idx][jdx][i]} \n')
        print(f'idx={i},min_eng={result.min_eng[idx][jdx][i]} \n')
# %%
