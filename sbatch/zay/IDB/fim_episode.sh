#!/bin/bash
#SBATCH --job-name=id_backbone
#SBATCH --output=../../../slurm/IDB/task-%A_%a_no_fn_tuned.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=8:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=csb@v100
#SBATCH --array=0-7
mode=$1
dat_ind=$SLURM_ARRAY_TASK_ID


list_dat=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list_dat[$dat_ind]}

cd ../../../

dataset_path="${SCRATCH}/"
path_to_subsets=""${WORK}/episode_600/binary_agnostic_{}.npy""
task_file="${WORK}/episode_600/magnitude600_${mode}_test_${dat}.pt" 


module purge
module load pytorch-gpu/py3/1.12.1



python fim_dist_episodes.py \
    --dataset-path ${dataset_path} \
    --load-episodes "${WORK}/episode_600/magnitude600_${mode}_test_${dat}.pt" \
    --out-file "fim/result_${mode}_${dat}.pt" \
    --target-dataset metadataset_${dat}
