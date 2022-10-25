from args import args
import torch
import dataloaders
from dataloaders import trainSet
import os

# precompute all distances ??? 


def get_features_metadata(features):
    num_elements_per_class = []
    vec = features[0]['features']
    mean = [0,features[0]['features'].shape[0]]
    for x in features[1:]:
        class_feat = x['features']
        mean.append(x['features'].mean(0))
        vec = torch.cat((vec,class_feat), dim = 0) 
        num_elements_per_class.append(num_elements_per_class[-1]+ class_feat.shape[0])
    return {'num_elements_per_class' :num_elements_per_class , 'vec' : vec, 'centroids' : torch.stack(mean)}



def select_closest(support, query, base_features, preselect_class = False, transductive=False, K_sample = 10 , K_class = 5, run_number = change_me):
    if transductive:
        #to do join the support and queries
        run = support + query
    else:
        run = support
    if preselect_class:  #we preselect using the class centroids
        D_class = torch.cdist(run , base_features['centroids'])
        _, cl = torch.topk(-D_class , K_class, dim = 1)   #We have too many classes 
        counts = {item:cl.ravel().count(item) for item in cl.ravel()}
        selected_class = list(counts.keys())
        #l_keys , l_values = counts.keys(), [keyval[1] for keyval in counts.items() ] 
        #selected_class = [l_keys[i] for i in (-l_values).argsort()[:K_class]]
        index = torch.zeros(base_features['vec'].shape[0], dtype=torch.bool)
        for j in selected_class:
            index[base_features['num_elements_per_class'][selected_class[j]] : base_features['num_elements_per_class'][selected_class[j]+1]]  = True
        
        D_sample = torch.cdist(run, base_features['vec'][index])
        _, samples = torch.topk(-D_sample , K_class, dim = 1)
    else:
        D_sample = torch.cdist(run, base_features['vec'])
        _, samples = torch.topk(-D_sample , K_class, dim = 1)

    
    save2file(samples, run_number)
    return samples

    
def save2file(index, run_number):
    data = trainSet[0]['dataloader'].dataset.data
    targets = trainSet[0]['dataloader'].dataset.targets
    select_data = [data[i] for i in index]
    selected_targets = [targets[i] for i in index]
    if os.isfile(args.seed+'.json'):
        # todo :) 
        #1) make it save 

