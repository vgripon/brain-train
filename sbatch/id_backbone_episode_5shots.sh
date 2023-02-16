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
#SBATCH -t 03:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-39
#SBATCH --output=../slurm/id_backbone_episode/task-%A_%a_id_backbone_episode.out
set -eux



source /gpfs/users/a1881717/env.sh

export WANDB_MODE=offline



list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
list2=("snr" "loo" "fake_acc" "hard" "soft")
length=${#list2[@]}
valtest="validation"
mag_or_ncm="magnitude"
task_id=$SLURM_ARRAY_TASK_ID
dat=${list1[$((task_id / length))]}
proxy=${list2[$((task_id % length))]}
fsfinetune="/gpfs/users/a1881717/5shots_work_dir/runs_fs/features/${dat}"
dirvis="/gpfs/users/a1881717/work_dir/vis/features/${dat}/"
dirsem="/gpfs/users/a1881717/work_dir/sem/features/${dat}/"
dirrandom="/gpfs/users/a1881717/work_dir/random/features/${dat}/"
loadepisode="/gpfs/users/a1881717/5shots_work_dir/runs_fs/episodes/${mag_or_ncm}_5shots_${dat}.pt"

directories=($dirvis $dirsem $dirrandom)
result="["
count=0
if [ "$dat" == "traffic_signs" ]; then
valtest="test"
else
valtest="validation"
fi

for dir in "${directories[@]}"; do
  echo $dir
  files=$(find "$dir" -type f -name "*$valtest*") 
  for file in $files; do
    result="$result'$file',"
    count=$((count+1))
  done
done


# Remove the trailing comma and add the closing bracket
result="${result%,}]"
result="\"$result\""


echo $result
echo "$dat"
echo "$proxy"



python ../id_backbone.py --out-file /gpfs/users/a1881717/5shots_work_dir/runs_fs/5shots_selection.pt --valtest $valtest --fs-finetune $fsfinetune --load-episode $loadepisode --num-cluster $count --target-dataset $dat --proxy $proxy --competing-features $result --dataset-path /users/local/datasets/  --seed 1 --few-shot-ways 0 --few-shot-shots 0 --few-shot-queries 0  --few-shot-runs 200 --dataset-path /gpfs/users/a1881717/datasets/