#!/bin/bash
#SBATCH -J mace-LPS-LBO
#SBATCH -p dzagnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
##SBATCH --cpus-per-task=8
####SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

module purge
source ~/miniconda3/bin/activate mace-0.3.13
module load nvidia/nvhpc/20.9
module load nvidia/cuda/11.7
###conda activate torch2.0.1


##ROOT_DIR=/work/home/USER/LiCl-GaF3/remove-errstructure
##SEED=3
##RES_DIR=/work/home/USER/LiCl-GaF3/results

####cd $ROOT_DIR

srun python /work/home/USER/software/mace-0.3.13/mace/mace/cli/run_train.py \
    --name=Li2CO3-LiF \
    --foundation_model="./mace-mpa-0-medium.model"  \
    --model_dir=./results1 \
    --log_dir=./results1 \
    --checkpoints_dir=./results1 \
    --results_dir=./results1 \
    --train_file="./soap_train_1800.xyz" \
    --valid_fraction=0.1 \
    --test_file="./soap_test_200.xyz" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=0.0 \
    --loss='universal' \
    --forces_key=dft_forces \
    --energy_key=dft_energy \
    --lr=0.0005 \
    --scaling="rms_forces_scaling" \
    --batch_size=8 \
    --max_num_epochs=300 \
    --ema \
    --ema_decay=0.99 \
    --weight_decay=1e-6 \
    --amsgrad \
    --default_dtype="float64" \
    --clip_grad=10 \
    --device=cuda \
    --seed=123 \
    --restart \
    --num_samples_pt=500 \
    --swa_energy_weight=100.0 \
    --swa_forces_weight=10.0 \
    --swa_stress_weight=0.0 \
    --swa \
    --swa_lr=5e-4 \
    --E0s='{3:-0.014670, 5:-0.032092 , 15:-0.055928 , 16:-0.054604, 8: -0.051660}' \

