#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -J gen_feat
#SBATCH -p gpunodes
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm/gen_feat/task-%A_%all_fs.out



dat_ind=${1:-0} ; shift

source /gpfs/users/a1881717/env.sh
set -eux
list=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")

# Get the current task ID from the SLURM_ARRAY_TASK_ID environment variable

# Get the current string from the list based on the task ID
dat=${list[${dat_ind}]}
if [ "$dat" == "traffic_signs" ]; then
    python ../main.py --dataset-path /gpfs/users/a1881717/datasets/  --test-dataset metadataset_${dat}_test --freeze-backbone  --load-backbone /gpfs/users/a1881717/exp_cub_vs_air/backbones/backbones_DINO  --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/DINO/features/feat --backbone resnet12 --few-shot --few-shot-shots 0 --few-shot-runs 10000 --few-shot --few-shot-ways 0
else
    python ../main.py --dataset-path /gpfs/users/a1881717/datasets/ --validation-dataset metadataset_${dat}_validation --test-dataset metadataset_${dat}_test --freeze-backbone  --load-backbone /gpfs/users/a1881717/exp_cub_vs_air/backbones/backbones_DINO  --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/DINO/features/feat --backbone resnet12 --few-shot --few-shot-shots 0 --few-shot-runs 10000 --few-shot --few-shot-ways 0
fi