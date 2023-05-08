#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=../../../slurm/S/task%jtask.out # fichier de sortie (%j = job ID)
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
mag_or_ncm="magnitude"
dataset_path="${SCRATCH}/"
load_backbone="${WORK}/resnet12_metadataset_imagenet_64.pt"
task_file="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt" \
index_subset=$SLURM_ARRAY_TASK_ID
train_dataset="metadataset_${dat}_test"
epoch="15"
wd="0.0002"
scheduler="cosine"
backbone="resnet12"
batch_size="128"
lr=0.01
optimizer="Adam"
freeze_backbone="--freeze-backbone --force-train"


save_classifier_base="${WORK}/results/S/${mode}/classifiers/${dat}/"
save_classifier="${save_classifier_base}/classifier_finetune_freeze_${SLURM_ARRAY_TASK_ID}"


module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes

python ../../../main.py \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --task-file "${task_file}" \
    --index-subset "${index_subset}" \
    --training-dataset "${train_dataset}" \
    --epoch "${epoch}" \
    --wd "${wd}" \
    --lr "${lr}" \
    --scheduler "${scheduler}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --save-classifier "${save_classifier}" \
    --optimizer ${optimizer} \
    ${freeze_backbone}