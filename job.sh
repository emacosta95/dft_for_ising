#!/usr/bin/env bash

#SBATCH --job-name="train_different_sizes"
#SBATCH --time=24:00:00
#SBATCH --mem=4G
#SBATCH --array=5
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --partition=m100_usr_prod
#SBATCH --output=/m100/home/userexternal/ecosta01/dft_for_ising/output/output_JOB_%j.out
#SBATCH --error=/m100/home/userexternal/ecosta01/dft_for_ising/output/error_JOB_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emanuele.costa@unicam.it
#SBATCH --account=IscrC_SMORAGEN

#=============================
# environment

source activate dft_env

echo "Running on "`hostname`

#=============================
# user definitions



#=============================
# running

srun python train.py  --hidden_channel 40 40 40   --kernel_size=$SLURM_ARRAY_TASK_ID --padding=2  --model_name=1nn_ising/h_2.7_unet_no_aug --data_path=data/1nn_ising/train_without_augmentation/unet_periodic_16_l_2.7_h_150000_n.npz --model_type=REDENTnopooling --pooling_size=1 --epochs=3000 

