#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=../../../slurm/DI/task%jtask.out # fichier de sortie (%j = job ID)
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=4 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:1 # reserver 4 GPU
#SBATCH --cpus-per-task=2 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=01:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-t3 # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=csb@v100 # comptabilite V100
#SBATCH --array=0-7

list1=("aircraft" "cub" "dtd" "fungi" "omniglot" "mscoco" "traffic_signs" "vgg_flower")
# Get the current string from the list based on the task ID
dat=${list1[$SLURM_ARRAY_TASK_ID]}

dataset_path="${SCRATCH}/"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
subset_file="${WORK}/episode_600/binary_${dat}_50.npy"
index_subset="0"
training_dataset="metadataset_imagenet_train"
epoch="10"
dataset_size="10000"
wd="0.0001"
scheduler="cosine"
backbone="resnet12"
batch_size="128"
few_shot_shots="0"
few_shot_ways="0"
few_shot_queries="0"
few_shot="--few-shot"
lr=0.001
optimizer="adam"
freeze_backbone='--freeze-backbone --force-train'

save_backbone_base="${WORK}/results/DI2/backbones/${dat}/"
save_classifier_base="${WORK}/results/DI2/classifiers/${dat}/"
save_classifier="${save_classifier_base}/classifier"



module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes

python ../../../main.py \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --subset-file "${subset_file}" \
    --index-subset "${index_subset}" \
    --training-dataset "${training_dataset}" \
    --epoch "${epoch}" \
    --dataset-size "${dataset_size}" \
    --wd "${wd}" \
    --lr "${lr}" \
    --scheduler "${scheduler}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --few-shot-shots "${few_shot_shots}" \
    --few-shot-ways "${few_shot_ways}" \
    --few-shot-queries "${few_shot_queries}" \
    ${few_shot} \
    ${freeze_backbone} \
    --save-classifier "${save_classifier}" \
    --optimizer "${optimizer}" 
    
    