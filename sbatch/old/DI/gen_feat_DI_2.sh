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
#SBATCH --array=0-7



dat_ind=${1:-0} ; shift

source /gpfs/users/a1881717/env.sh

export WANDB_MODE=offline

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")

dat=${list1[$dat_ind]}
dat2=${list1[$SLURM_ARRAY_TASK_ID]}
set -eux
if [ "$dat2" == "traffic_signs" ]; then
python ../../main.py --dataset-path /gpfs/users/a1881717/datasets/ \
  --test-dataset metadataset_${dat2}_test --freeze-backbone \
  --load-backbone /gpfs/users/a1881717/work_dir/DI/backbones/${dat}/backbones \
  --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/DI/features/${dat2}/f_${dat} --backbone resnet12 \
  --save-test /gpfs/users/a1881717/work_dir/DI/results_cross_DI.pt
  $@
else
python ../../main.py --dataset-path /gpfs/users/a1881717/datasets/ \
 --validation-dataset metadataset_${dat2}_validation \
  --test-dataset metadataset_${dat2}_test --freeze-backbone \
  --load-backbone /gpfs/users/a1881717/work_dir/DI/backbones/${dat}/backbones \
  --epoch 1 --save-features-prefix /gpfs/users/a1881717/work_dir/DI/features/${dat2}/f_${dat} --backbone resnet12 \
  --save-test /gpfs/users/a1881717/work_dir/DI/results_cross_DI.pt
  $@
fi

wandb sync
