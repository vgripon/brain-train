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
task_file="${WORK}/episode_600/${mag_or_ncm}600_${mode}_test_${dat}.pt" \
index_subset=$SLURM_ARRAY_TASK_ID
test_dataset="metadataset_${dat}_test"
epoch="1"
wd="0.0001"
scheduler="cosine"
backbone="resnet12"
batch_size="128"
lr=0.001
optimizer="adam"
freeze_backbone="--freeze-backbone"
task_queries="--task-queries"
mkdir "${WORK}/results/S/measure/${mode}"
mkdir "${WORK}/results/S/measure/${mode}/${dat}"
save_stats="${WORK}/results/S/measure/${mode}/${dat}/${SLURM_ARRAY_TASK_ID}.json"

load_backbone_base="${WORK}/results/S/${mode}/backbones/${dat}/"
load_backbone="${load_backbone_base}/backbones1e4_${SLURM_ARRAY_TASK_ID}"

load_classifier_base="${WORK}/results/S/${mode}/classifiers/${dat}/"
load_classifier="${load_classifier_base}/classifier_finetune1e4_${SLURM_ARRAY_TASK_ID}"


module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes

python ../../../main.py \
    --dataset-path "${dataset_path}" \
    --load-backbone "${load_backbone}" \
    --load-classifier "${load_classifier}" \
    --task-file "${task_file}" \
    --index-subset "${index_subset}" \
    --test-dataset "${test_dataset}"  ${task_queries} \
    --epoch "${epoch}" \
    --wd "${wd}" \
    --lr "${lr}" \
    --backbone "${backbone}" \
    --batch-size "${batch_size}" \
    --save-stats ${save_stats} \
    ${freeze_backbone}