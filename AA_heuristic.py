import torch
import numpy as np
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

work_dir =  args.subset_file

vis,sem,random ,visem= np.load(work_dir+'binary_agnostic_V.npy'),np.load(work_dir+'binary_agnostic_S.npy'),np.load(work_dir+'binary_agnostic_R.npy'),np.load(work_dir+'binary_agnostic_X.npy')
selection = np.concatenate((vis,sem,random,visem), axis= 0) 
real_selection = (selection==0)*1
real_selection = torch.tensor(real_selection)
datasets=['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'vgg_flower', 'traffic_signs']
valtest='test'



def get_subsets_aa(ten):
    M, D = real_selection.shape

    AA = []
    for i in range(M):
        indices = torch.nonzero(real_selection[i]).squeeze()  # get the indices of the non-zero elements in the ith row of real_selection
        aa = ten[:, indices].sum()  # use advanced indexing to extract the relevant columns of a for the ith row of real_selection
        AA.append(aa)

    return AA




def yield_proxy(episodes, datasets):
    from id_backbone import init_seed
    valtest=args.valtest
    log = torch.load(args.load_logits.format(datasets,valtest))
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
    fullAA=[]
    if episodes!=None:
        for run in range(args.few_shot_runs):
            episodes_used.append({'shots_idx' : episodes['shots_idx'][run], 'queries_idx' : episodes['queries_idx'][run], 'choice_classes' : episodes['choice_classes'][run]})
    for run in range(args.few_shot_runs):
        if episodes==None:
            init_seed(args.seed+run)
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries, max_queries = args.max_queries)
        else:
            episode = episodes_used[run]
            print('used episode',run)
        shots, queries = generator.get_features_from_indices(log, episode)

        ten = torch.softmax(torch.cat(shots),dim=1)
        fullAA.append(get_subsets_aa(ten))
        
    return torch.tensor(fullAA)


if __name__ == "__main__":
    episodes = torch.load(args.load_episodes)['episodes']
    out = yield_proxy(episodes=episodes, datasets=args.target_dataset).detach().numpy()
    np.save(args.out_file,out.T) #must be npy

