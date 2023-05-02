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
#SBATCH --array=0-299
#SBATCH --output=../../slurm/gen_logits/task-%A_%a.out




source /gpfs/users/a1881717/env.sh

export WANDB_MODE=offline
set -eux
index=$SLURM_ARRAY_TASK_ID
dat=$1

python ../../main.py --freeze-backbone --freeze-classifier \
    --load-classifier /gpfs/users/a1881717/work_dir/pred_perf/target_subsets/classifiers/${dat}/classifier_finetune_${index} \
    --load-backbone /gpfs/users/a1881717/work_dir/pred_perf/target_subsets/backbones/${dat}/backbone_${index} \
    --dataset-path /gpfs/users/a1881717/datasets/ --epoch 1 \
    --training-dataset metadataset_imagenet_train --backbone resnet12 \
    --test-dataset metadataset_${dat}_test \
    --save-logits /gpfs/users/a1881717/work_dir/pred_perf/target_subsets/logits/${dat}/logits_${dat}_test_${index}.pt \
    --subset-file /gpfs/users/a1881717/work_dir/pred_perf/target_subsets/binary_50_${dat}.npy \
    --index-subset ${index} \

wandb sync