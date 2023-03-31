import torch
import numpy as np
from joblib import load
from sklearn.metrics.cluster import homogeneity_score , completeness_score , v_measure_score
import networkx as nx
import json
import pandas as pd
from args import args
from few_shot_evaluation import EpisodicGenerator, ImageNetGenerator, OmniglotGenerator
import os


if 'omniglot' in args.target_dataset:
    data_info_omniglot_file = os.path.join(args.dataset_path, 'omniglot.json')
    with open(data_info_omniglot_file) as f:
        data_info_omniglot = json.load(f)
    data_info_omniglot=data_info_omniglot['metadataset_omniglot_{}'.format(args.valtest)]

vis,sem,random ,visem= np.load('/gpfs/users/a1881717/work_dir/binary_agnostic_vis.npy'),np.load('/gpfs/users/a1881717/work_dir/binary_agnostic_sem4.npy'),np.load('/gpfs/users/a1881717/work_dir/binary_agnostic_random.npy'),np.load('/gpfs/users/a1881717/work_dir/binary_agnostic_visem.npy')
selection = np.concatenate((vis,sem,random,visem), axis= 0) 
real_selection = (selection==0)*1
real_selection = torch.tensor(real_selection)
datasets=['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'vgg_flower', 'traffic_signs']
model = load('/gpfs/users/a1881717/brain-train/finetuning/adjusted.joblib')
valtest='val'



def get_subsets_logits(ten):
    M, D = real_selection.shape

    subtensors = []
    for i in range(M):
        indices = torch.nonzero(real_selection[i]).squeeze()  # get the indices of the non-zero elements in the ith row of b
        subtensor = ten[:, indices]  # use advanced indexing to extract the relevant columns of a for the ith row of b
        subtensors.append(subtensor)

    return subtensors



def homogeneity(ten,labels_true):
    pred = torch.argmax(ten,dim=1)
    h=homogeneity_score(labels_true,pred.cpu())
    return h

def magnitude(ten):
    n=torch.norm(ten, dim=(0))
    sn=n.sort(descending=True)[0]
    return sn[:50].sum().cpu().item()/ten.shape[0]




def yield_proxy(episodes, datasets):
    from id_backbone import init_seed
    hnm=[]
    valtest=args.valtest
    if datasets=='traffic_signs':
        valtest='test'
    log = torch.load('/gpfs/users/a1881717/work_dir/logits/logits_{}_{}.pt'.format(datasets,valtest))
    for x in log:
        x['features'] = x.pop('logits')
    if datasets=='omniglot':
        Generator = OmniglotGenerator
        generator = Generator(datasetName='omniglot', num_elements_per_class= [len(feat['features']) for feat in log], dataset_path=args.dataset_path)
        generator.dataset = data_info_omniglot
    else:
        args.dataset_path = None
        Generator = EpisodicGenerator
        generator = Generator(datasetName=None, num_elements_per_class= [len(feat['features']) for feat in log], dataset_path=args.dataset_path)
    episodes_used=[]
    if episodes!=None:
        for run in range(args.few_shot_runs):
            episodes_used.append({'shots_idx' : episodes['shots_idx'][run], 'queries_idx' : episodes['queries_idx'][run], 'choice_classes' : episodes['choice_classes'][run]})
    for run in range(args.few_shot_runs):
        if episodes==None:
            init_seed(args.seed+run)
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries, max_queries = args.max_queries)
        else:
            episode = episodes_used[run]
        shots, queries = generator.get_features_from_indices(log, episode)

        ten = torch.softmax(torch.cat(shots),dim=1)
        labels_true = torch.cat([torch.tensor([i]).repeat(x.shape[0]) for i,x in enumerate(shots)])
        subsets = get_subsets_logits(ten)
        L_y=[]
        for x in subsets:
            h=homogeneity(x,labels_true)
            m=magnitude(x)
            y = model.predict(np.array([[h,m]]))
            L_y.append(y)
        hnm.append(L_y)
    return np.array(hnm).T


