#!/bin/bash
#SBATCH -J v22-SR
#SBATCH -p dzagnormal
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -c 8
#SBATCH -x gpu13
##SBATCH --cpus-per-task=8
####SBATCH --gpus-per-task=1
#SBATCH --gres=gpu:1

module purge
source ~/miniconda3/bin/activate mace_pdf
module load nvidia/nvhpc/20.9
module load nvidia/cuda/11.7
###conda activate torch2.0.1


#ROOT_DIR=/work/home/USER/LiCl-GaF3/remove-errstructure
##SEED=3
#RES_DIR=/work/home/USER/LiCl-GaF3/results

#cd $ROOT_DIR

srun python /work/home/USER/software/mace-0.3.12/mace/cli/run_train.py \
    --name=Na-NSS-v22-SR \
    --foundation_model="mace-mpa-0-medium.model"  \
    --model_dir=./results \
    --log_dir=./results \
    --checkpoints_dir=./results \
    --results_dir=./results \
    --heads="{'default': {'train_file': ['soap_train_900.xyz','soap_train_1800.xyz','repulsion22_train.xyz'], 'valid_file': ['soap_test_100.xyz','soap_test_200.xyz','repulsion22_valid.xyz']}}" \
    --energy_weight=1.0 \
    --forces_weight=10.0 \
    --stress_weight=10.0 \
    --loss='universal' \
    --forces_key=forces \
    --energy_key='free_energy' \
    --stress_key=stress \
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
    --num_samples_pt=1000 \
    --swa_energy_weight=100.0 \
    --swa_forces_weight=10.0 \
    --swa_stress_weight=100.0 \
    --swa \
    --swa_lr=5e-4 \
    --E0s='{11:-0.25828324, 16:-0.88924654, 51:-1.4274714}' \
