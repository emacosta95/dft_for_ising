import os

l = [24]
# data = [300000] * len(l)
hs = [2.40, 3.60, 7.20, 8.40]
# data_str = ["150k"] * len(l)
nblocks = None
hidden_channel = " 40 40 40 40  "
ks = [5] * len(hs)
padding = [2] * len(hs)
model_type = "Den2Func"
epochs = [3000] * len(hs)
for i in range(len(hs)):
    print(
        f"nohup python train.py   --hidden_channel"
        + hidden_channel
        + f"  --kernel_size={ks[i]} --padding={padding[i]}  --model_name='2nn_ising/h_{hs[i]}_cnn' --data_path='data/dataset_2nn/train_unet_periodic_2nn_augmentation_16_l_{hs[i]:.2f}_h_30000_n.npz --model_type={model_type} --pooling_size=1 --epochs={epochs[i]} > output/train_unet_{model_type}.txt &"
    )
    os.system(
        f"nohup python train.py  --hidden_channel"
        + hidden_channel
        + f"  --kernel_size={ks[i]} --padding={padding[i]}  --model_name='2nn_ising/h_{hs[i]}_cnn' --data_path='data/dataset_2nn/train_unet_periodic_2nn_augmentation_16_l_{hs[i]:.2f}_h_30000_n.npz' --model_type={model_type} --pooling_size=1 --epochs={epochs[i]} > output/train_unet_{model_type}.txt &"
    )
    # hidden_channel = hidden_channel + " 40"
