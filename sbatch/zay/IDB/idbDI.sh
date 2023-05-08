#!/bin/bash
#SBATCH --job-name=id_backbone
#SBATCH --output=../../../slurm/IDB/task-%A_%a_no_fn_tuned.out
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --time=1:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH --hint=nomultithread
#SBATCH --account=csb@v100
#SBATCH --array=0-7
mode=$1

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")




few_shot_runs="600"
dataset_path="${SCRATCH}/"
epoch="1"
backbone="resnet12"
batch_size="128"
few_shot="--few-shot"

if [ "$mode" == "MD" ]; then
    few_shot_shots="0"
    few_shot_ways="0"
    few_shot_queries="0"
elif [ "$mode" == "1s5w" ]; then
    few_shot_shots="1"
    few_shot_ways="5"
    few_shot_queries="15"
elif [ "$mode" == "5s5w" ]; then
    few_shot_shots="5"
    few_shot_ways="5"
    few_shot_queries="15"
else
    echo "Invalid mode. Please choose between MD, 1s5w, and 5s5w."
    exit 1
fi

valtest="test"
mag_or_ncm="magnitude"

task_id=$SLURM_ARRAY_TASK_ID
dat=${list1[$dat_ind]}
test_dataset="metadataset_${dat}_test"
count=0
proxy="snr"

cheated="${WORK}/results/DI/features/${dat}/f_20_0.001metadataset_${dat}_test_features.pt"
baseline="${WORK}/results/B/f_baselinemetadataset_${dat}_test_features.pt"
loadepisode="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt"
outfile="${WORK}/results/IDB/idb_DI_${mode}_${dat}.pt"



echo $result
echo "$dat"
echo "$proxy"


module purge
module load pytorch-gpu/py3/1.12.1

python ../../../id_backbone.py \
    --out-file $outfile \
    --valtest $valtest \
    --load-episode $loadepisode \
    --num-cluster $count \
    --target-dataset $dat \
    --cheated ${cheated} \
    --proxy $proxy \
    --seed 1 \
    --few-shot-shots "${few_shot_shots}" \
    --few-shot-ways "${few_shot_ways}" \
    --few-shot-queries "${few_shot_queries}" \
    --few-shot-runs $few_shot_runs \
    ${few_shot} \
    --dataset-path "${dataset_path}" \
    --baseline "${baseline}"