import numpy as np
import sys
import os
import torch

popos = False
if popos:
    sys.path.append('/home/raphael/Documents/brain-train')
else:
    sys.path.append('/homes/r21lafar/Documents/brain-train')  #put the path of brain-train
from args import args
import json
n_runs = 5

#translate file


def make_npy0( topk, choice ):
    x = np.ones(712).astype(int)   #selection
    selected = torch.randperm(x.shape[0])[:topk]
    torch.save(selected, 'finetuning/selections/'+choice+'.pt')
    x[selected] = 0
    np.save(os.path.join(args.work_folder, 'processed.npy'), x)




suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb  --wd 0.0001 '
dir1 = args.work_folder
for choice in ['random']:                                      #THIS IS the most important line
    for iteration, selection in enumerate(['magnitude']*n_runs):                     #THIS could be anything I just want to open the file
    
        for k in [50]:
            make_npy0(topk = k, choice = choice+str(iteration))
            key = str(k)+choice+selection
            if popos: 
                #create dataset_json
                os.system('python create_dataset_files.py --dataset-path /home/datasets/ --subdomain {}/processed.npy --fast-create-dataset'.format(args.work_folder))
                prefix = 'python main.py --dataset-path /home/datasets/ --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_0  --few-shot ' 
            else:
                os.system('python create_dataset_files.py --dataset-path '+ args.dataset_path+' --subdomain {}/processed.npy --fast-create-dataset'.format(args.work_folder))
                prefix = 'python main.py --dataset-path '+ args.dataset_path+' --load-backbone '+ args.load_backbone+' --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_0  --few-shot ' 

            print(os.path.join(args.dataset_path,'datasets_subdomain.json'))
            with open(os.path.join(args.dataset_path,'datasets_subdomain.json')) as f:
                dic = json.load(f)


            print(dic['metadataset_imagenet_0']['name_classes'])    
            os.system( prefix  +' --freeze-backbone --freeze-classifier --epochs 1  {2} --batch-size 128 --save-features-prefix {0}/features/baseline --wandbProjectName finetuning --info {1} '.format(dir1, key, suffix))

                
            os.system( prefix +  ' --freeze-backbone --force-train --epochs 10 --dataset-size 10000 --lr 0.01  --save-classifier {0}/classifiers/{3}{1}.pt   {2} --batch-size 128  --info {1} --wandbProjectName classifier'.format(dir1, key, suffix,  dataset))
            #finetuning

            for lr in [0.01]:
                lr_str = str(lr)
                print(type(lr_str))
                command = prefix  +' --epochs 1 --dataset-size 1000 --lr {4}  --save-backbone {0}/backbones/{3}{1}{4}.pt --save-features-prefix {0}/features/finetuned_{1}{4} --save-classifier {0}/classifiers/{3}classifier_finetuned_{1}{4}.pt --load-classifier {0}/classifiers/{3}{1}.pt  {2} --batch-size 128 --scheduler linear --wandbProjectName finetuning --info {1}'.format(dir1, key, suffix,  dataset, lr_str)
                print(command)
                os.system(command)