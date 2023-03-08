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
#SBATCH -p gpunodes
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 2:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --output=../../slurm/gen_feat/task-%A_%all_fs.out

set -eux

dat_ind=${1:-0} ; shift

source /gpfs/users/a1881717/env.sh

export WANDB_MODE=offline

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")

dat=${list1[$dat_ind]}

python ../../main.py --freeze-backbone --freeze-classifier \
    --load-classifier /gpfs/users/a1881717/exp_cub_vs_air/classifiers/classifier_2birds\
    --load-backbone /gpfs/users/a1881717/exp_cub_vs_air/backbones/backbones_2birds \
    --dataset-path /gpfs/users/a1881717/datasets/ --epoch 1 \
    --training-dataset metadataset_imagenet_train --backbone resnet12 \
    --test-dataset metadataset_${dat}_test \
    --save-logits /gpfs/users/a1881717/exp_cub_vs_air/logits/logits_${dat}_test.pt2 \
    --subset-file /gpfs/users/a1881717/exp_cub_vs_air/selection/no_birds_vis_execpt2.npy \
    --index-subset 0 \
    $@

wandb sync