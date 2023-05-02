#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -J FS_gen
#SBATCH -p v100
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 01:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-7
#SBATCH --output=../slurm/gen_feat/task-%A_%a.out
##SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM

dat_ind=${1:-0} ; shift

source /gpfs/users/a1881717/env.sh

list=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")

# Get the current string from the list based on the task ID
dat=${list[$SLURM_ARRAY_TASK_ID]}

set -eux
if [ "$dat" == "traffic_signs" ]; then
    python ../main.py --dataset-path /gpfs/users/a1881717/datasets/ \
    --test-dataset metadataset_${dat}_test --freeze-backbone \
    --load-backbone /gpfs/users/a1881717/resnet12_metadataset_imagenet_64.pt \
    --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/baseline/features/${dat}/feat --backbone resnet12 \
    --few-shot-shots 5 --few-shot-ways 15 --few-shot-queries 15  \
    --save-test /gpfs/users/a1881717/work_dir/baseline/results_baseline_5s5w.pt 

    $@
else
    python ../main.py --dataset-path /gpfs/users/a1881717/datasets/ \
    --validation-dataset metadataset_${dat}_validation \
    --test-dataset metadataset_${dat}_test --freeze-backbone \
    --load-backbone /gpfs/users/a1881717/resnet12_metadataset_imagenet_64.pt \
    --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/baseline/features/${dat}/feat --backbone resnet12 \
    --few-shot-shots 5 --few-shot-ways 15 --few-shot-queries 15  \
    --save-test /gpfs/users/a1881717/work_dir/baseline/results_baseline_5s5w.pt 
    $@
fi

