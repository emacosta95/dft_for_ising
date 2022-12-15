import os
import numpy as np

ls = np.arange(33,65)
hs = [5.44]
# ks = [1, 3, 5, 7, 9, 11]
ls_train = [8, 12, 18, 20]
for k in ls_train:
    for l in ls:
        os.system(
            f"nohup python run.py --model_name='2nn_ising/h_{hs[0]}_15k_l_{k}_cnn_[40, 40, 40, 40]_hc_{5}_ks_1_ps_4_nconv_0_nblock' --target_path=data/dataset_2nn/test_dataset_021222/test_periodic_{l}_l_{hs[0]}_h_100_n.npz --run_name=h_{hs[0]}_l_train_{k}_15k_augmentation_2nn_model_unet_{l}_size_4_layers_40_hc_{5}_ks_1_ps --init_path=data/dataset_2nn/test_dataset_021222/test_periodic_{l}_l_{hs[0]}_h_100_n.npz --epochs=2000 --device=cpu --num_threads=1 > output/run_l_{l}_cnn_15k.txt &"
        )
