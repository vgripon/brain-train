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

all_embeddings={'competing_agnostics_with_baseline' : [], 'tasks': []}
for VSRX in ['V' ,'S','R','X', 'baseline']:
    all_embeddings['competing_agnostics_with_baseline']+= torch.load('fim/embeddings_{}.pt'.format(VSRX))

for i in range(600):
    cfg_task={'phase': 'train',
                'name' : args.target_dataset,
                'subset_file':'',
                'task_file' : args.load_episodes,
                'index_episode' : i,
                'phase' : 'test',
                'force-test-transforms':True}

    task = prepareDataLoader("custom_metadataset", is_train=False,cfg=cfg_task)[0]
    probe_network = get_model('resnet18', pretrained=True, num_classes=task['num_classes'])
    embedding_task =  task2vec.Task2Vec(probe_network,max_samples=max_sample, skip_layers=6).embed(task['dataloader'].dataset)
    all_embeddings['tasks'].append(embedding_task)
    print('episode : {}'.format(i))


torch.save(all_embeddings['tasks'],'{}_embbedings'.format(args.out_file))

ALL_embeddings = all_embeddings['tasks']+all_embeddings['competing_agnostics_with_baseline']

D = task_similarity.pdist(ALL_embeddings,distance='cosine')
print(D.shape)
np.save(args.out_file, D)