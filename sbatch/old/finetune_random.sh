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
#SBATCH -t 2:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-10
#SBATCH --output=../slurm/task-%A_%afinetune_random.out

set -eux

module load arch/skylake
module load Python/3.8.6
module load CUDA/11.2.0
module load cuDNN/CUDA-11.2

source /hpcfs/users/a1881717/lab/bin/activate




export WANDB_MODE=offline

python ../main.py --dataset-path /hpcfs/users/a1881717/datasets/   --few-shot --few-shot-shots 0 --few-shot-runs 10000 --few-shot --few-shot-ways 0  --load-backbone /hpcfs/users/a1881717/backbones/resnet12_metadataset_imagenet_64.pt  --subset-file /hpcfs/users/a1881717/work_dir/binary_agnostic_random.npy --index-subset $SLURM_ARRAY_TASK_ID --training-dataset metadataset_imagenet_train --epoch 20 --dataset-size 10000 --wd 0.0001 --lr 0.001  --load-classifier /hpcfs/users/a1881717/work_dir/random/classifiers/classifier_$SLURM_ARRAY_TASK_ID --backbone resnet12 --batch-size 128 --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --few-shot  --save-backbone /hpcfs/users/a1881717/work_dir/random/backbones/backbone_$SLURM_ARRAY_TASK_ID --save-classifier /hpcfs/users/a1881717/work_dir/random/classifiers/classifier_finetune_$SLURM_ARRAY_TASK_ID  

wandb sync
