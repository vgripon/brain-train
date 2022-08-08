import os
from time import sleep
os.system('ps aux | grep subdomains.py')
N=50   # number of subdomains 
os.system('python create_dataset_files.py --dataset-path /users2/libre/datasets/ --subdomain /users/local/clusters_imagenet/'+str(N)+'clusters.npy')

for i in range(N):
    os.system('python main.py --dataset-path /users2/libre/datasets/ --few-shot-shots 5 --few-shot-ways 5 --few-shot-queries 15 --training-dataset metadataset_imagenet_cluster_train_'+str(i)+' --validation-dataset metadataset_aircraft_validation --test-dataset metadataset_aircraft_test --epochs 200 --lr 0.1 --cosine --few-shot --few-shot-unbalanced-queries  --save-features-prefix /users/local/r21lafar/features/subdomains/clusters'+str(i)+' --save-backbone /users/local/r21lafar/models/subdomains/clusters'+str(i) )
    sleep(3)  # time to exit process