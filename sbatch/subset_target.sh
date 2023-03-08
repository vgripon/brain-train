#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -J id_backbone
#SBATCH -p gpunodes
#SBATCH -N 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=6-6
#SBATCH --output=../slurm/id_backbone_episode/task-%A_%a_id_backbone_episode.out
set -eux


source /gpfs/users/a1881717/env.sh

difficulty=$@


python ../main.py --dataset-path /gpfs/users/a1881717/datasets/   --test-dataset metadataset_mscoco_test \
  --freeze-backbone   --load-backbone /gpfs/users/a1881717/resnet12_metadataset_imagenet_64.pt \
  --epoch 1  --backbone resnet12 --few-shot-shots 0 --few-shot-ways 0 --few-shot-runs 1000 \
  --subset-file /gpfs/users/a1881717/work_dir/pred_perf/${difficulty}_subsets.npy --index-subset $SLURM_ARRAY_TASK_ID --subset-split test \
  --few-shot --save-stats /gpfs/users/a1881717/work_dir/pred_perf/${difficulty}.pt --device cpu