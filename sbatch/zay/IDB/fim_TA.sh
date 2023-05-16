#!/bin/bash
#SBATCH --job-name=id_backbone
#SBATCH --output=../../../slurm/IDB/task-%A_%a_no_fn_tuned.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=csb@v100


cd ../../../
dataset_path="${SCRATCH}/"
path_to_subsets="${WORK}/episode_600/binary_agnostic_{}.npy"

module purge
module load pytorch-gpu/py3/1.12.1


set -x
python fim_distTA.py \
    --dataset-path "${dataset_path}" \
    --info ${path_to_subsets}
