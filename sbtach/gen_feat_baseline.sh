#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -M volta
#SBATCH -J FS_raph2
#SBATCH -p batch
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 00:01:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-21
#SBATCH --output=../slurm/task-%A_%a.out
set -eux

module load arch/skylake
module load Python/3.8.6
module load CUDA/11.2.0
module load cuDNN/CUDA-11.2

source /hpcfs/users/a1881717/lab/bin/activate

list1=("aircraft" "cub")

# Get the current task ID from the SLURM_ARRAY_TASK_ID environment variable
task_id=$SLURM_ARRAY_TASK_ID
# Get the current string from the list based on the task ID
dat=${list1[$SLURM_ARRAY_TASK_ID]}

python ../main.py --dataset-path /hpcfs/users/a1881717/datasets/ --validation-dataset metadataset_${dat}_validation --test-dataset metadataset_${dat}_test --freeze-backbone  --load-backbone /hpcfs/users/a1881717/backbones/resnet12_metadataset_imagenet_64.pt  --epoch 1 --save-features-prefix /hpcfs/users/a1881717/work_dir/baseline/features/${dat}/feat --backbone resnet12