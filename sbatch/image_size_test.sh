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
#SBATCH --output=../slurm/other/task-%A_%a_image_size.out
set -eux


source /gpfs/users/a1881717/env.sh

Xmin=50
Xmax=250
N=10
step=$(echo "($Xmax - $Xmin) / ($N - 1)" | bc)

for ((i=0; i<$N; i++)); do
    value=$(echo "$Xmin + $i * $step" | bc)
    rounded=$(printf "%.0f" "$value")
    python ../main.py --dataset-path /gpfs/users/a1881717/datasets/   --test-dataset metadataset_mscoco_test \
    --freeze-backbone   --load-backbone /gpfs/users/a1881717/resnet12_metadataset_imagenet_64.pt \
    --epoch 1  --backbone resnet12 --few-shot-shots 0 --few-shot-ways 0 --few-shot-runs 1000 \
    --test-image-size $rounded \
    --few-shot --save-stats /gpfs/users/a1881717/work_dir/pred_perf/image_size.pt 
done

