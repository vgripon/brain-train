import numpy as np
import sys
import os
import torch
popos = True
if popos:
    sys.path.append('/home/raphael/Documents/brain-train')
else:
    sys.path.append('/homes/r21lafar/Documents/brain-train')
from args import args
import json


#translate file


def make_npy0(file_descending_ordered_index, topk=20):
    ind = torch.load(os.path.join(args.work_folder,file_descending_ordered_index))
    x = np.ones(ind.shape[0]).astype(int)
    selected = ind[:topk].cpu().detach().numpy()
    x[selected] = 0
    np.save(os.path.join(args.work_folder,'processed.npy'), x)



validtest = '--validation-dataset metadataset_aircraft_validation --test-dataset metadataset_aircraft_test'
suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb --wandbProjectName brain-train --wd 0.0001'
dir1 = args.work_folder
for k in [2]:
    #create dataset_json
    make_npy0(args.selection_file, topk = k)
    if popos: 
        os.system('python create_dataset_files.py --dataset-path /home/datasets/ --subdomain {}/processed.npy --fast-create-dataset'.format(args.work_folder))
        prefix = 'python main.py --dataset-path /home/datasets/ --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_0  --few-shot ' 
        key = 'optimal'+str(k)

    else:
        os.system('python create_dataset_files.py --dataset-path /users2/libre/datasets/ --subdomain {}/processed.npy --fast-create-dataset'.format(args.work_folder))
        prefix = 'python main.py --dataset-path /users2/libre/datasets/ --load-backbone /users2/libre/raphael/resnet12_metadataset_imagenet_64.pt --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_0  --few-shot ' 
        key = ''+str(k)
        
    print(os.path.join(args.dataset_path,'datasets_subdomain.json'))
    with open(os.path.join(args.dataset_path,'datasets_subdomain.json')) as f:
        dic = json.load(f)
    print('\n\n k = {} \n\n'.format(str(k)))
    print(dic['metadataset_imagenet_0']['name_classes'])    

    #get_classifier
    #os.system( prefix + validtest +' --freeze-backbone --freeze-classifier --epochs 1  {2} --batch-size 128 --save-features-prefix {0}baseline{1}'.format(dir1, key, suffix))
    #os.system( prefix +  ' --freeze-backbone --force-train --epochs 5 --lr 0.1  --save-classifier {0}finetune_aircraft_classifier{1}.pt   {2} --batch-size 128 --save-features-prefix {0}baseline{1} '.format(dir1, key, suffix))
    #finetuning
    list_lr = [ 0.00005, 0.00003,0.00001,0.000005 ,0.0 ]

    for lr in list_lr:
        lr_str = str(lr)
        print(type(lr_str))
        command = prefix + validtest +' --epochs 5 --lr {0}  --save-backbone {1}finetune_aircraft_{0}{3}.pt --save-features-prefix {1}finetuned_aircraft_{0}{3} --save-classifier {1}classifier_finetuned_aircraft_{0}{3} --load-classifier {1}finetune_aircraft_classifier{3}.pt  {2} --batch-size 128'.format(lr_str, dir1, suffix, key)
        print(command)
        os.system(command)