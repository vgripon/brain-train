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


def make_npy0(file_descending_ordered_index, topk=20, choice = 'best'):
    ind = torch.load(os.path.join('finetuning/selections',file_descending_ordered_index))
    x = np.ones(ind.shape[0]).astype(int)
    if choice == 'best':
        selected = ind[:topk].cpu().detach().numpy()
    elif choice == 'worst':
        selected = ind[-topk:].cpu().detach().numpy()
    elif choice == 'random':
        selection = torch.randint(ind.shape[0], (topk,))
        selected = ind[selection].cpu().detach().numpy()
    x[selected] = 0
    np.save(os.path.join(args.work_folder, 'processed.npy'), x)




suffix = '--backbone resnet12  --wandb raflaf --wandb-dir wandb  --wd 0.0001 '
dir1 = args.work_folder
for choice in ['random']:                                      #THIS IS the most important line
    for selection in ['magnitude']*n_runs:                     #THIS could be anything I just want to open the file
        for dataset in  ['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'traffic_signs', 'vgg_flower']:
            if dataset != 'traffic_signs':
                validtest = '--validation-dataset metadataset_{0}_validation --test-dataset metadataset_{0}_test'.format(dataset)
            else:
                validtest = '--test-dataset metadataset_{0}_test'.format(dataset)
            for k in [50]:
                make_npy0(selection+'_selected_{}.pt'.format(dataset), topk = k, choice = choice)
                key = str(k)+choice+selection

                if popos: 
                    #create dataset_json
                    os.system('python create_dataset_files.py --dataset-path /home/datasets/ --subdomain {}/processed.npy '.format(args.work_folder))
                    prefix = 'python main.py --dataset-path /home/datasets/ --load-backbone /home/raphael/Documents/models/resnet12_metadataset_imagenet_64.pt --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_0  --few-shot ' 

                else:
                    os.system('python create_dataset_files.py --dataset-path '+ args.dataset_path+' --subdomain {}/processed.npy '.format(args.work_folder))
                    prefix = 'python main.py --dataset-path '+ args.dataset_path+' --load-backbone '+ args.load_backbone+' --few-shot-shots 0 --few-shot-ways 0 --few-shot-queries 0 --training-dataset metadataset_imagenet_0  --few-shot ' 

                print(os.path.join(args.dataset_path,'datasets_subdomain.json'))
                with open(os.path.join(args.dataset_path,'datasets_subdomain.json')) as f:
                    dic = json.load(f)

                print('\n\n k = {} , dataset = {} , choice = {} \n\n'.format(str(k), dataset, choice))

                print(dic['metadataset_imagenet_0']['name_classes'])    
                os.system( prefix + validtest +' --freeze-backbone --freeze-classifier --epochs 1  {2} --batch-size 128 --save-features-prefix {0}/features/baseline --wandbProjectName finetuning --info {1} '.format(dir1, key, suffix))

                
                os.system( prefix +  ' --freeze-backbone --force-train --epochs 10 --lr 0.01  --save-classifier {0}/classifiers/{3}{4}{1}.pt   {2} --batch-size 128  --info {1} --wandbProjectName classifier'.format(dir1, key, suffix, str(i), dataset))
                #finetuning

                for lr in [0.01]:
                    lr_str = str(lr)
                    print(type(lr_str))
                    command = prefix + validtest +' --epochs 1 --lr {5}  --save-backbone {0}/backbones/{3}{4}{1}{5}.pt --save-features-prefix {0}/features/{3}finetuned_{1}{5} --save-classifier {0}/classifiers/{3}{4}classifier_finetuned_{1}{5}.pt --load-classifier {0}/classifiers/{3}{4}{1}.pt  {2} --batch-size 128 --scheduler linear --wandbProjectName finetuning --info {1}'.format(dir1, key, suffix, str(i), dataset, lr_str)
                    print(command)
                    os.system(command)