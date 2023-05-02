#!/bin/bash
#SBATCH --job-name=TravailGPU # nom du job
#SBATCH --output=TravailGPU%j.out # fichier de sortie (%j = job ID)
#SBATCH --error=TravailGPU%j.err # fichier d’erreur (%j = job ID)
#SBATCH --constraint=v100-16g # demander des GPU a 16 Go de RAM
#SBATCH --nodes=1 # reserver 1 nœud
#SBATCH --ntasks=4 # reserver 4 taches (ou processus)
#SBATCH --gres=gpu:4 # reserver 4 GPU
#SBATCH --cpus-per-task=10 # reserver 10 CPU par tache (et memoire associee)
#SBATCH --time=01:00:00 # temps maximal d’allocation "(HH:MM:SS)"
#SBATCH --qos=qos_gpu-dev # QoS
#SBATCH --hint=nomultithread # desactiver l’hyperthreading
#SBATCH --account=csb@v100 # comptabilite V100
module purge # nettoyer les modules herites par defaut
#conda deactivate # desactiver les environnements herites par defaut
module load pytorch-gpu/py3/1.12.1 # charger les modules
set -x # activer l’echo des commandes
srun python ../../main.py # executer son script