#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -J train
#SBATCH -p gpunodes
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 96:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --output=../slurm/train/task-%A_%a_.out
set -eux

source /gpfs/users/a1881717/env.sh



python ../main.py \
  --dataset-path /gpfs/users/a1881717/datasets/   \
  --training-dataset metadataset_imagenet_train \
  --epoch 300  --wd 0.001 --optimizer adam --lr 0.001  \
  --scheduler cosine --backbone resnet12 --batch-size 64   \
  --save-backbone /gpfs/users/a1881717/exp_cub_vs_air/backbones/backbones_DINO \
  --save-classifier /gpfs/users/a1881717/exp_cub_vs_air/classifiers/classifier_DINO \
  --steps "[['dino']]"

