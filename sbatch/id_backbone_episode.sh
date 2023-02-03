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
#SBATCH -t 01:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-0
#SBATCH --output=../slurm/task-%A_%a_id_backbone_episode.out
set -eux

module load arch/skylake
module load Python/3.8.6
module load CUDA/11.2.0
module load cuDNN/CUDA-11.2

source /hpcfs/users/a1881717/lab/bin/activate



list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
list2=("snr")

valtest="validation"
mag_or_ncm="magnitude"
task_id=$SLURM_ARRAY_TASK_ID
dat=${list1[$((task_id / 3))]}
proxy=${list2[$((task_id % 3))]}
fsfinetune="/hpcfs/users/a1881717/work_dir/runs_fs/features/${dat}"
dir="/hpcfs/users/a1881717/work_dir/vis/features/${dat}/"
featureslist=$(ls $dir)
array=($featureslist)
loadepisode="/hpcfs/users/a1881717/work_dir/runs_fs/episodes/${mag_or_ncm}_${dat}.pt"

for item in "${array[@]}"; do
  if [[ $item == *"$valtest"* ]]; then
    filtered_array+=("$item")
  fi
done


length=${#filtered_array[@]}
string='"['
for item in "${array[@]}"; do
  string+="'$dir$item',"
done
string=${string::-1} # remove the last comma
string+=']"'
echo $string


echo "$featureslist"
echo "$length"
echo "$dat"
echo "$proxy"

python ../id_backbone.py --valtest $valtest --fs-finetune $fsfinetune --load-episode $loadepisode --num-cluster $length --target-dataset $dat --proxy $proxy --competing-features $string --dataset-path /users/local/datasets/  --seed 1 --few-shot-ways 0 --few-shot-shots 0 --few-shot-queries 0  --few-shot-runs 100