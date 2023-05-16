import sys
sys.path.insert(0, "../aws-cv-task2vec")
import task2vec
from models import get_model
import task_similarity
from args import args
from dataloaders import prepareDataLoader
import backbones
import torch
import numpy as np

max_sample=200

for VSRX in ['V', 'S', 'R', 'X']:
    all_embeddings={'competing_agnostics_with_baseline' : [], 'tasks': []}
    for i in range(11):
        cfg_subset = {'phase': 'train',
                    'name' : 'metadataset_imagenet',
                    'subset_file': args.info.format(VSRX),
                    'task_file' : '',
                    'index_subset':i,
                    'force-test-transforms':True}


        subset = prepareDataLoader("custom_metadataset", cfg=cfg_subset)[0]
        probe_network = get_model('resnet18', pretrained=True, num_classes=subset['num_classes'])
        embedding_subset =  task2vec.Task2Vec(probe_network,max_samples=max_sample, skip_layers=6).embed(subset['dataloader'].dataset)
        all_embeddings['competing_agnostics_with_baseline'].append(embedding_subset)
        print('cluster number {}'.format(i))
    torch.save(all_embeddings['competing_agnostics_with_baseline'],'fim/embeddings_{}.pt'.format(VSRX))
    print(VSRX, 'done')

cfg_baseline={'phase': 'train',
            'name' : 'metadataset_imagenet',
            'subset_file':'',
            'task_file' : '',
            'force-test-transforms':True}

baseline = prepareDataLoader("custom_metadataset", cfg=cfg_baseline)[0]
probe_network = get_model('resnet18', pretrained=True, num_classes=baseline['num_classes'])
embedding_baseline =  task2vec.Task2Vec(probe_network,max_samples=max_sample, skip_layers=6).embed(baseline['dataloader'].dataset)
torch.save([embedding_baseline], 'fim/embeddings_baseline.pt')