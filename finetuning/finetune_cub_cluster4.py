import os
from time import sleep
os.system('ps aux | grep subdomains.py')
popos = True
sl12 = False

N=10   # number of subdomains 
suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb --wandbProjectName brain-train --wd 0.0001'


if popos: 
    prefix = 'python main.py --dataset-path /home/datasets/ --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --few-shot-shots 5 --few-shot-ways 5 --few-shot-queries 15 --training-dataset metadataset_imagenet_4 --validation-dataset metadataset_cub_validation --test-dataset metadataset_cub_test --few-shot --few-shot-unbalanced-queries' 
    dir1 = '/home/raphael/Documents/models/'
    key = 'optimal'
    #create dataset_json
    os.system('python create_dataset_files.py --dataset-path /home/datasets/ --subdomain /home/raphael/Documents/models/'+str(N)+'clusters.npy --fast-create-dataset')

elif sl12:
    prefix = 'python main.py --dataset-path /users2/libre/datasets/ --load-backbone /users2/libre/clusters/clusters --few-shot-shots 5 --few-shot-ways 5 --few-shot-queries 15 --training-dataset metadataset_imagenet_4 --validation-dataset metadataset_cub_validation --test-dataset metadataset_cub_test --few-shot --few-shot-unbalanced-queries' 
    dir1 = '/users2/libre/raphael/'
    suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb --wandbProjectName brain-train --wd 0.0001'
    key = ''
    #create dataset_json
    os.system('python create_dataset_files.py --dataset-path /users2/libre/datasets/ --subdomain /users/local/clusters_imagenet/'+str(N)+'clusters.npy --fast-create-dataset')
    

#get_classifier
os.system( prefix + ' --freeze-backbone --force-train --epochs 30 --lr 0.1   --save-classifier {0}finetune_cubc4_classifier{1}.pt  {2} --batch-size 128'.format(dir1, key, suffix))

#finetuning
list_lr = [0.5,0.1,0.05,0.01,0.005]

for lr in list_lr:
    lr_str = str(lr)
    print(type(lr_str))
    command = prefix +' --epochs 20 --lr {0}  --save-backbone {1}finetune_cubc4_{0}{3}.pt --save-features-prefix {1}finetuned_cubc4_{0}{3} --save-classifier {1}classifier_finetuned_cubc4_{0}{3} --load-classifier {1}finetune_cubc4_classifier.pt  {2} --batch-size 128'.format(lr_str, dir1, suffix, key)
    print(command)
    os.system(command)