import torch
import numpy as np
from joblib import load
from sklearn.metrics.cluster import homogeneity_score , completeness_score , v_measure_score
import networkx as nx
import json
import pandas as pd
from args import args

vis,sem,random = np.load('working_dirs/binary_agnostic_vis.npy'),np.load('working_dirs/binary_agnostic_sem.npy'),np.load('working_dirs/binary_agnostic_random.npy')
selection = np.concatenate((vis,sem,random), axis= 0) 
real_selection = (selection==0)*1
real_selection = torch.tensor(real_selection)
datasets=['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'vgg_flower', 'traffic_signs']
model = load('finetuning/adjusted.joblib')
valtest='val'


def get_subsets_logits(ten):
    M, D = real_selection.shape

    subtensors = []
    for i in range(M):
        indices = torch.nonzero(real_selection[i]).squeeze()  # get the indices of the non-zero elements in the ith row of b
        #print(row_indices)
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




def choose_backbone(episodes, datasets, generator):
    hnm=[]
    valtest='val'
    if datasets=='traffic_signs':
        valtest='test'
    log = torch.load('/home/raphael/Documents/models/old_logits/logits_{}_{}.pt'.format(datasets,valtest))
    for x in log:
        x['features'] = x.pop('logits')
    if episodes==None:
        episodes_used=[]
        for i in range(args.few_shot_runs):
            episodes_used.append(generator.sample_episode(ways=args.few_shot_ways, n_shots=args.n_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries, max_queries = args.max_queries))
    else:
        for run in (args.few_shot_runs):
            episodes_used = {'shots_idx' : episodes['shots_idx'][run], 'queries_idx' : episodes['queries_idx'][run], 'choice_classes' : episodes['choice_classes'][run]}
    for epi in episodes:
        shots, queries = generator.get_features_from_indices(log, epi)

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
    return hnm


