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
#SBATCH -p a100
#SBATCH -N 1
#SBATCH -N 1
#SBATCH -c 4
#SBATCH -t 24:00:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH --array=0-47
#SBATCH --output=../../slurm/id_backbone_episode/task-%A_%a_id_backbone_episode.out



source /gpfs/users/a1881717/env.sh



list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
list2=("snr" "loo" "fake_acc" "hard" "soft" "hnm")
length=${#list2[@]}
valtest="test"
mag_or_ncm="magnitude"
task_id=$SLURM_ARRAY_TASK_ID
dat=${list1[$((task_id / length))]}
proxy=${list2[$((task_id % length))]}
fsfinetune="/gpfs/users/a1881717/MD_work_dir/support_direct_MD/features/${dat}"
dirvis="/gpfs/users/a1881717/work_dir/vis/features/${dat}/"
dirsem="/gpfs/users/a1881717/work_dir/sem4/features/${dat}/"
dirrandom="/gpfs/users/a1881717/work_dir/random/features/${dat}/"
dirvisem="/gpfs/users/a1881717/work_dir/visem/features/${dat}/"
loadepisode="/gpfs/users/a1881717/work_dir/magnitudes_test/${mag_or_ncm}_MD_test_${dat}.pt"
#outfile="/gpfs/users/a1881717/MD_work_dir/support_MD/result_support_MD.pt"
#cheated="/gpfs/users/a1881717/work_dir/DI/features/${dat}/fmetadataset_${dat}_${valtest}_features.pt"
lr="0.1"
cheated="/gpfs/users/a1881717/work_dir/DI_lr/features/${dat}/f_${lr}metadataset_${dat}_${valtest}_features.pt"
outfile="/gpfs/users/a1881717/work_dir/DI_lr/result_id_backbone_support_MD_${lr}.pt"
directories=($dirvis $dirsem $dirrandom $dirvisem)
result="["
count=0

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
set -eux
#python ../../id_backbone.py --out-file $outfile --cheated $cheated --valtest $valtest  --fs-finetune $fsfinetune \
python ../../id_backbone.py --out-file $outfile --cheated $cheated --valtest $valtest \
--load-episode $loadepisode --num-cluster $count --target-dataset $dat --proxy $proxy \
 --competing-features $result  --seed 1 --few-shot-ways 0 \
 --few-shot-shots 0 --few-shot-queries 0  --few-shot-runs 200 --dataset-path /gpfs/users/a1881717/datasets/