import os
import numpy as np


hc = ["150k"]
hs = [2.7]
epochs = [5000]
ls = list(range(33, 64))
for j, k in enumerate(ls):
    for i, l in enumerate(hc):
        os.system(
            f"nohup python run.py --model_name='1nn_ising/h_{hs[0]}_150k_unet_no_aug_[40, 40, 40, 40, 40, 40]_hc_{5}_ks_1_ps_6_nconv_0_nblock' --target_path=data/dataset_1nn/240123/unet_periodic_{k}_l_{hs[0]}_h_6000_n.npz --run_name=h_{hs[0]}_{l}_1nn_model_unet_16_l_train_{k}_size_6_layers_40_hc_{5}_ks_1_ps --init_path=data/dataset_1nn/240123/unet_periodic_{k}_l_{hs[0]}_h_6000_n.npz --epochs={epochs[0]} --n_instances=50 --device=cpu --num_threads=1 > output/run_h_cnn.txt &"
        )
        # os.system(
        #     f"nohup python run.py --model_name='1nn_ising/h_{hs[0]}_150k_unet"
        #     + l
        #     + f"hc_{5}_ks_1_ps_{hc_n[i]}_nconv_0_nblock' --target_path=data/dataset_1nn/240123/unet_periodic_{k}_l_{hs[0]}_h_6000_n.npz --run_name=h_{hs[0]}_150k_augmentation_1nn_model_unet_{k}_size_{hc_n[i]}_layers_40_hc_{5}_ks_1_ps --init_path=data/dataset_1nn/240123/unet_periodic_{k}_l_{hs[0]}_h_6000_n.npz --epochs=6000 --n_instances=300 --device=cpu --num_threads=1 > output/run_h_cnn.txt &"
        # )
