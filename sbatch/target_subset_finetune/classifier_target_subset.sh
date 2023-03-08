#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -J FS_cls
#SBATCH -p gpunodes
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-299
#SBATCH --output=../../slurm/classifier/task-%A_%all_fs.out

set -eux

dat_ind=${1:-0} ; shift

source /gpfs/users/a1881717/env.sh

export WANDB_MODE=offline

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")

task_id=$SLURM_ARRAY_TASK_ID
# Get the current string from the list based on the task ID
dat=${list1[$dat_ind]}
index=$task_id

python ../../main.py \
  --dataset-path /gpfs/users/a1881717/datasets/ \
  --load-backbone /gpfs/users/a1881717/resnet12_metadataset_imagenet_64.pt \
  --subset-file /gpfs/users/a1881717/work_dir/pred_perf/target_subsets/binary_50_${dat}.npy \
  --index-subset ${index} \
  --training-dataset metadataset_imagenet_train \
  --epoch 20 --dataset-size 10000 --wd 0.0001 --lr 0.001 \
  --save-classifier /gpfs/users/a1881717/work_dir/pred_perf/target_subsets/classifiers/${dat}/classifier_${index} \
  --backbone resnet12 --batch-size 128 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot --optimizer adam \
  $@

wandb sync
