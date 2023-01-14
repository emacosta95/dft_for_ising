import os
import numpy as np

ls_train = [8, 12, 18, 24]
hs = [2.7]
# ks = [1, 3, 5, 7, 9, 11]
ls = list(range(33, 64))
for k in ls:
    for l in ls_train:
        os.system(
            f"nohup python run.py --model_name='1nn_ising/h_{hs[0]}_150k_l_{l}_cnn_[40, 40, 40, 40]_hc_{5}_ks_1_ps_4_nconv_0_nblock' --target_path=data/dataset_1nn/field2density_221122/valid_unet_periodic_{k}_l_{hs[0]}_h_200_n.npz --run_name=h_{hs[0]}_l_train_{l}_150k_augmentation_1nn_model_unet_{k}_size_4_layers_40_hc_{5}_ks_1_ps --init_path=data/dataset_1nn/field2density_221122/valid_unet_periodic_{k}_l_{hs[0]}_h_200_n.npz --epochs=5000 --device=cpu --num_threads=1 > output/run_l_{l}_cnn_150k.txt &"
        )
