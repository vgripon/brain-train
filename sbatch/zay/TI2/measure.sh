#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=../../../slurm/TI/task%jmeasure.out # fichier de sortie (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=4 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=2 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=01:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=csb@v100 # comptabilite V100
#SBATCH --array=0-600
mode=$1
dat_ind=$2
list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$dat_ind]}

load_backbone_base="${WORK}/results/TI2/backbones/${mode}/${dat}"
load_backbone="${load_backbone_base}/backbones_"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
loadepisode="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt"
indexepisode=$SLURM_ARRAY_TASK_ID
dataset_path="${SCRATCH}/"
test_dataset="metadataset_${dat}_test"
epoch="1"
few_shot_runs="1"
index_subset=$SLURM_ARRAY_TASK_ID
backbone="resnet12"
batch_size="128"
few_shot="--few-shot"
lr=0.001

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
save_stats="${WORK}/results/TI2/measure/${mode}/${dat}/${SLURM_ARRAY_TASK_ID}.json"

module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes

python ../../../main.py \
    --freeze-backbone \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --load-episodes "${loadepisodes}" \
    --test-dataset "${test_dataset}" \
    --index-episode "${indexepisode}" \
    --epoch "${epoch}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --few-shot-shots "${few_shot_shots}" \
    --few-shot-ways "${few_shot_ways}" \
    --few-shot-queries "${few_shot_queries}" \
    --few-shot-runs "${few_shot_runs}" \
    ${few_shot} \
    ${task_queries} \
    --save-stats $save_stats \

