import os
from time import sleep
os.system('ps aux | grep subdomains.py')
popos = True
sl12 = False

N=10   # number of subdomains 
prefix = 'python few_shot_evaluation.py --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0  --few-shot-runs 1000 --test-dataset metadataset_cub_test --few-shot ' 


if popos: 
    dir1 = '/home/raphael/Documents/models/'
    key = 'optimal'

elif sl12:
    dir1 = '/users2/libre/raphael/'
    key = ''
    
#get_basline
command = prefix + ' --test-features {0}baseline{1}metadataset_cub_test_features.pt'.format(dir1, key)
os.system(command)

#finetuning
list_lr = [0.5,0.1,0.05,0.01,0.005]

for lr in list_lr:
    lr_str = str(lr)
    command = prefix + ' --test-features {0}finetuned_cubc4_{1}{2}metadataset_cub_test_features.pt '.format(dir1,lr_str, key)
    print(command)
    os.system(command)