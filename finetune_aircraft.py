import os
from time import sleep
os.system('ps aux | grep subdomains.py')
N=10   # number of subdomains 
os.system('python create_dataset_files.py --dataset-path /users2/libre/datasets/ --subdomain /users2/libre/raphael/aircraft_finetune.npy --fast-create-dataset')
os.system('python main.py --dataset-path /users2/libre/datasets/ --few-shot-shots 5 --few-shot-ways 5 --few-shot-queries 15 --training-dataset metadataset_imagenet_0 --validation-dataset metadataset_aircraft_validation --test-dataset metadataset_aircraft_test --epochs 50 --lr 0.1 --cosine --few-shot --few-shot-unbalanced-queries --load-backbone /users2/libre/clusters/clusters --save-backbone /users2/libre/raphael/finetune_aircraft.pt --save-features-prefix /users2/libre/raphael/finetuned_aircraft --save-classifier /users2/libre/raphael/finetune_aircraft_classifier.pt  --backbone resnet12')# --wandb raflaf --wandbProjectName brain-train' )