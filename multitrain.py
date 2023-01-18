import os

data = [300000]
ks = [5] * len(data)
hs = [2.7] * len(ks)
data_str = ["150k"]
hidden_channel = " 40 40 40 40 40 40"
padding = [2] * len(ks)
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

    # hidden_channel = hidden_channel + " 40"


# os.system(
#     f"nohup python train.py  --hidden_channel 40 40 40 40 40 40 40 40  --kernel_size=5 --padding=2  --model_name='1nn_ising/h_2.7_l_24_cnn' --data_path='data/dataset_1nn/train_unet_periodic_augmentation_16_l_2.7_h_300000_n.npz' --model_type=Den2Func --pooling_size=1 --epochs=7000 > output/train_unet_170123.txt &"
# )
