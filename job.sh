#!/usr/bin/env bash

#SBATCH --job-name="quench_simulations"
#SBATCH --time=05:00:00
#SBATCH --mem=4G
#SBATCH --array=8,24
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4        # 1 gpus per node out of 4
#SBATCH --partition=m100_usr_prod
#SBATCH --output=/g100/userexternal/ecosta01/output/230123_JOB_%j.out
#SBATCH --error=/g100/userexternal/ecosta01/output/230123_JOB_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL,TIME_LIMIT
#SBATCH --mail-user=emanuele.costa@unicam.it
#SBATCH --account=icei_Pilati

#=============================
# environment

source activate dft_env

echo "Running on "`hostname`

#=============================
# user definitions





#=============================
# running

srun python train.py  --hidden_channel 40 40 40 40 40 40  --kernel_size=5 --padding=2  --model_name=1nn_ising/h_2.7_unet_l_$SLURM_ARRAY_TASK_ID --data_path=data/1nn_ising/train_unet_periodic_augmentation_$l$\_l_2.7_h_300000_n.npz' --model_type=REDENTnopooling --pooling_size=1 --epochs=7000 

