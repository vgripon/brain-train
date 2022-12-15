import os
from time import sleep
os.system('ps aux | grep subdomains.py')
popos = False
sl12 = True

N=10   # number of subdomains 
suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb --wandbProjectName brain-train'


if popos: 
    prefix = 'python main.py --dataset-path /home/datasets/ --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --few-shot-shots 5 --few-shot-ways 5 --few-shot-queries 15 --training-dataset metadataset_imagenet_4 --validation-dataset metadataset_cub_validation --test-dataset metadataset_cub_test --few-shot --few-shot-unbalanced-queries' 
    dir1 = '/home/raphael/Documents/models/'
    key = 'optimal'
    #create dataset_json
    os.system('python create_dataset_files.py --dataset-path /home/datasets/ --subdomain /home/raphael/Documents/models/'+str(N)+'clusters.npy --fast-create-dataset')

elif sl12:
    prefix = 'python few_shot_evaluation.py --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --test-dataset metadataset_cub_test --few-shot ' 
    dir1 = '/users2/libre/raphael/'
    suffix = ''
    key = ''
    #create dataset_json
    

#get_basline
command = os.system( prefix + ' --test-features {1}baseline{0}metadataset_cub_test_features.pt'.format(dir1, key))

#finetuning
list_lr = [0.5,0.1,0.05,0.01,0.005]

for lr in list_lr:
    lr_str = str(lr)
    print(type(lr_str))
    command = prefix + ' --test-features {1}finetuned_cubc4_{0}metadataset_cub_test_features.pt '.format(dir1, key)
    print(command)
    os.system(command)