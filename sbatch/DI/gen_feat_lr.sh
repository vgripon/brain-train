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
#SBATCH --array=46-46
#SBATCH --output=../../slurm/gen_feat/task-%A_%all_fs.out

source /gpfs/users/a1881717/env.sh

export WANDB_MODE=offline

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
list2=(0.0 0.00001 0.0001 0.001 0.01 0.1)

# Calculate the indexes for both lists
index1=$(( SLURM_ARRAY_TASK_ID / ${#list2[@]} ))
index2=$(( SLURM_ARRAY_TASK_ID % ${#list2[@]} ))

# Get the values from both lists
dat=${list1[$index1]}
lr=${list2[$index2]}

if [ "$dat" == "traffic_signs" ]; then
python ../../main.py --dataset-path /gpfs/users/a1881717/datasets/ \
  --test-dataset metadataset_${dat}_test --freeze-backbone \
  --load-backbone /gpfs/users/a1881717/work_dir/DI_lr/backbones/${dat}/backbones_${lr} \
  --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/DI_lr/features/${dat}/f_${lr} --backbone resnet12 \
  --save-test /gpfs/users/a1881717/work_dir/DI_lr/full_results/results_${dat}_${lr}.pt 
  $@
else
python ../../main.py --dataset-path /gpfs/users/a1881717/datasets/ \
 --validation-dataset metadataset_${dat}_validation \
  --test-dataset metadataset_${dat}_test --freeze-backbone \
  --load-backbone /gpfs/users/a1881717/work_dir/DI_lr/backbones/${dat}/backbones_${lr} \
  --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/DI_lr/features/${dat}/f_${lr} --backbone resnet12\
  --save-test /gpfs/users/a1881717/work_dir/DI_lr/full_results/results_${dat}_${lr}.pt 
  $@
fi

wandb sync
