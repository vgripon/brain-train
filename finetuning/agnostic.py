import numpy as np
import sys
import os
import torch

popos = False
sys.path.append(os.getcwd())
sys.path.append(os.getcwd()+'/../')
from args import args
import json




suffix = '--backbone resnet12   --wd 0.0001 '
if args.wandb != '':
    suffix+= ' --wandb '+args.wandb
if args.wandb_dir != '':
    ' --wandb-dir '+args.wandb_dir
dir1 = args.work_folder
clusters = np.load(args.subdomain)
if popos: 
    #create dataset_json
    os.system('python create_dataset_files.py --dataset-path '+args.dataset_path+ ' --subdomain {} --fast-create-dataset'.format( args.subdomain))
else:
    os.system('python create_dataset_files.py --dataset-path '+args.dataset_path+ ' --subdomain {} --fast-create-dataset'.format( args.subdomain))

for cluster in range(0, len(np.unique(clusters))):
    key = args.subdomain[-25:-4]+ str(cluster)
    if popos: 
        prefix = 'python main.py --dataset-path '+args.dataset_path+ ' --load-backbone  '+ args.load_backbone  +' --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_{}  --few-shot '.format(str(cluster))
    else:
        prefix = 'python main.py --dataset-path '+args.dataset_path+ ' --load-backbone '+ args.load_backbone  +' --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_{}  --few-shot '.format(str(cluster))
    print(os.path.join(args.dataset_path,'datasets_subdomain.json'))
    with open(os.path.join(args.dataset_path,'datasets_subdomain.json')) as f:
        dic = json.load(f)

    print(dic['metadataset_imagenet_{}'.format(str(cluster))]['name_classes'])    

                    
    os.system( prefix +  ' --freeze-backbone --force-train --epochs 10 --dataset-size 10000 --lr 0.01  --save-classifier {0}/classifiers/{1}.pt   {2} --batch-size 128  --info {1} --wandbProjectName classifier'.format(dir1, key, suffix))
    #finetuning

    for lr in [0.01]:
        lr_str = str(lr)
        print(type(lr_str))
        command = prefix +' --epochs 5 --dataset-size 1000 --lr {3}  --save-backbone {0}/backbones/{1}{3}.pt --save-features-prefix {0}/features/finetuned_{1}{3} --save-classifier {0}/classifiers/classifier_finetuned_{1}{3}.pt --load-classifier {0}/classifiers/{1}.pt  {2} --batch-size 128 --scheduler linear --wandbProjectName finetuning --info {1}'.format(dir1, key, suffix, lr_str)
        print(command)
        os.system(command)