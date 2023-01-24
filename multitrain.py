import os

blocks = 2
data = [300000] * blocks
ks = [1, 3]
hs = [2.7] * len(ks)
data_str = ["150k"] * len(data)
hidden_channel = " 40 40 40 40 40 40 "
padding = [0, 1]
model_type = "REDENTnopooling"
epochs = [7000] * len(data)
for i in range(len(data)):
    print(
        f"nohup python train.py  --hidden_channel"
        + hidden_channel
        + f"  --kernel_size={ks[i]} --padding={padding[i]}  --model_name='1nn_ising/h_{hs[i]}_"
        + data_str[i]
        + f"_unet' --data_path='data/dataset_1nn/train_unet_periodic_augmentation_{16}_l_{hs[i]}_h_{data[i]}_n.npz' --model_type={model_type} --pooling_size=1 --epochs={epochs[i]} > output/train_unet_{model_type}.txt &"
    )
    os.system(
        f"nohup python train.py  --hidden_channel"
        + hidden_channel
        + f"  --kernel_size={ks[i]} --padding={padding[i]}  --model_name='1nn_ising/h_{hs[i]}_"
        + data_str[i]
        + f"_unet' --data_path='data/dataset_1nn/train_unet_periodic_augmentation_{16}_l_{hs[i]}_h_{data[i]}_n.npz' --model_type={model_type} --pooling_size=1 --epochs={epochs[i]} > output/train_unet_{model_type}.txt &"
    )


# os.system(
#     f"nohup python train.py  --hidden_channel 40 40 40 40 40 40   --kernel_size=5 --padding=2  --model_name='1nn_ising/h_2.7_unet' --data_path='data/dataset_1nn/train_unet_periodic_augmentation_16_l_2.7_h_300000_n.npz' --model_type=REDENTnopooling --pooling_size=1 --epochs=7000 > train_unet_170123.txt &"
# )
