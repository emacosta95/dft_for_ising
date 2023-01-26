import os


data = [5000, 10000, 15000, 30000, 150000]
ks = [5] * len(data)
hs = [2.7] * len(ks)
data_str = ["5k", "10k", "15k", "30k", "150k"] * len(data)
hidden_channel = " 40 40 40 40 40 40 "
padding = [2] * len(data)
model_type = "REDENTnopooling"
epochs = [3000] * len(data)
for i in range(len(data)):
    file_name = (
        f"nohup python train.py  --hidden_channel"
        + hidden_channel
        + f"  --kernel_size={ks[i]} --padding={padding[i]}  --model_name='1nn_ising/h_{hs[i]}_"
        + data_str[i]
        + f"_unet_no_aug' --data_path='data/dataset_1nn/train_without_augmentation/unet_periodic_{16}_l_{hs[i]}_h_{data[i]}_n.npz' --model_type={model_type} --pooling_size=1 --epochs={epochs[i]} > output/train_unet_{model_type}.txt &"
    )
    print(file_name)
    os.system(file_name)


# os.system(
#     f"nohup python train.py  --hidden_channel 40 40 40 40 40 40   --kernel_size=5 --padding=2  --model_name='1nn_ising/h_2.7_unet' --data_path='data/dataset_1nn/train_unet_periodic_augmentation_16_l_2.7_h_300000_n.npz' --model_type=REDENTnopooling --pooling_size=1 --epochs=7000 > train_unet_170123.txt &"
# )
