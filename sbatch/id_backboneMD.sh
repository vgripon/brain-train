#!/bin/bash

# This script should be launched from the experiment root.
# When EXP_NAME is set, it creates a subdir.
#
# Example usage:
# sbatch .../slurm/run_imagenet.sh --arch=resnet18
# ( EXP_NAME=resnet18 sbatch .../slurm/run_imagenet.sh --arch=resnet18 )
# ( EXP_NAME=resnet50 sbatch .../slurm/run_imagenet.sh --arch=resnet50 )
# ( EXP_NAME=resnet50-b128-lr0.05 sbatch .../slurm/run_imagenet.sh --arch=resnet50 --batch-size=128 --learning-rate=0.05 )

#SBATCH -J FS_raph2
#SBATCH -p gpunodes
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-47
#SBATCH --output=../slurm/id_backboneMD/task-%A_%a_id_backbone_MD.out


source /gpfs/users/a1881717/env.sh


set -eux
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
list2=("hard" "loo" "soft" "fake_acc" "snr" "rankme")
valtest="test"
length2=${#list2[@]}
task_id=$SLURM_ARRAY_TASK_ID
dat=${list1[$((task_id / length2))]}
proxy=${list2[$((task_id % length2))]}
fsfinetune="/home/raphael/Documents/brain-train/working_dirs//work_dir/runs_fs/features/${dat}"
dirvis="/home/raphael/Documents/brain-train/working_dirs//work_dir/vis/features/${dat}/"
dirsem="/home/raphael/Documents/brain-train/working_dirs/work_dir/sem2/features/${dat}/"
dirrandom="/home/raphael/Documents/brain-train/working_dirs//work_dir/random/features/${dat}/"
dirvisem="/home/raphael/Documents/brain-train/working_dirs//work_dir/visem/features/${dat}/"

directories=($dirvis $dirsem $dirrandom $dirvisem)
result="["
count=0
#if [ "$dat" == "traffic_signs" ]; then
valtest="test"
#else
#valtest="validation"
#fi
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


echo "$count"
echo "$dat"
echo "$proxy"

python ../id_backbone.py --valtest $valtest --num-cluster $count --target-dataset $dat \
 --proxy $proxy --competing-features $result --dataset-path /users/local/datasets/ \
 --seed 1 --few-shot-ways 0 --few-shot-shots 0 --few-shot-queries 0  --few-shot-runs 10000 \
 --dataset-path /gpfs/users/a1881717/datasets/ --out-file /gpfs/users/a1881717/work_dir/result_VSR_sem2_visem_MD.pt
