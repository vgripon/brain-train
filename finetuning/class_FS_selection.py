import sys
import torch
import torch.nn as nn
popos = True
if popos:
    sys.path.append('.')
else:
    sys.path.append('/homes/r21lafar/Documents/brain-train')
print(sys.path)
from few_shot_evaluation import EpisodicGenerator, ImageNetGenerator, OmniglotGenerator
from args import args
import os
import classifiers
import sys
import random 
from tqdm import tqdm
import numpy as np
import json 
k=10

with open(os.path.join(args.dataset_path,'datasets_subdomain.json')) as f:
    dataset_json = json.load(f)


def print_classes(ordered_acc, std, accuracy, nb_sample):
    conf_inter = 1.96* std / np.sqrt(nb_sample)
    print('\n Best classes are \n' )
    for x in ordered_acc[:10]:
        print("\t{:.3f} ±{:.3f}".format(accuracy[x].item(), conf_inter[x].item()  ), dataset_json['metadataset_imagenet_train']['name_classes'][x])
    #print('\n Worst classes are \n' )
    #for x in ordered_acc[-10:]:
    #    print("\t{:.3f} ±{:.3f}".format(accuracy[x].item(), conf_inter[x].item()  ), dataset_json['metadataset_imagenet_train']['name_classes'][x])


def measure_acc_by_dim(logits): # not used here yet
    means = torch.mean(logits, dim = 1)
    print(f"{logits.shape=},{means.shape=}")
    print(f"{logits.unsqueeze(1).shape=},{means.unsqueeze(1).unsqueeze(0).shape=}")
    perfect_samples = means.unsqueeze(1).repeat(1,100,1) 
    D = abs(logits.unsqueeze(0) - means.unsqueeze(1).unsqueeze(1))
    Dp = abs(perfect_samples.unsqueeze(0) - means.unsqueeze(1).unsqueeze(1))
    pred = D.argmin(dim = 1)
    pred_p = Dp.argmin(dim = 1)
    mismatch = pred==pred_p
    accuracy = mismatch.view(-1, D.shape[-1]).float().mean(dim = 0)
    std = mismatch.view(-1, D.shape[-1]).float().std(dim = 0)
    ordered_acc = accuracy.argsort(descending=True)
    return ordered_acc, std, accuracy



def GenFewShot(features, datasets = None, write_file=False):
    list_shots, list_queries = [],[]
    list_episodes = {'choice_classes': [], 'shots_idx': [], 'queries_idx': []}
    if datasets=='omniglot':
            Generator = OmniglotGenerator
            generator = Generator(datasetName='omniglot', num_elements_per_class= [feat['features'].shape[0] for feat in features], dataset_path=args.dataset_path)
            generator.dataset = dataset_json['metadataset_omniglot_test']
    else:
        args.dataset_path = None
        Generator = EpisodicGenerator
        generator = Generator(datasetName=None, num_elements_per_class= [feat['features'].shape[0] for feat in features], dataset_path=args.dataset_path)
    for run in range(args.few_shot_runs):
        shots = []
        queries = []
        episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
        shots, queries = generator.get_features_from_indices(features, episode)
        for k,v in episode.items():
            list_episodes[k].append(v)
        list_shots.append(shots)
        list_queries.append(queries)
    return list_episodes, list_shots, list_queries



def ncm_dim(shots,queries): 
    centroids = torch.stack([shotClass.mean(dim = 0) for shotClass in shots])
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        if queriesClass == []:
            pass
        else:
            distances = abs(queriesClass.unsqueeze(1) - centroids.unsqueeze(0))
            score += (distances.argmin(dim = 1) - i == 0).float().sum(0).cpu()
            total += queriesClass.shape[0]
    return score / total



print('hello')
if __name__=='__main__':
    for dataset in ['cub', 'aircraft', 'dtd',  'fungi', 'omniglot', 'traffic_signs', 'vgg_flower', 'mscoco']:
        print('\n\n {} \n\n'.format(dataset))
        if popos:
            logits_from_file = torch.load('/home/raphael/Documents/models/old_logits/logits_{}_test.pt'.format(dataset))
        else:
            logits_from_file = torch.load('/users2/libre/raphael/old_logits/logits_{}_test.pt'.format(dataset))

        softmax = nn.Softmax(dim = 1)
        #feats = torch.stack([x['logits'] for x in logits_from_file])
        #logits = softmax(feats)
        #logits_list = [{'features' : logits[i]  } for i in range(logits.shape[0])]
        logits_dic = [{'features' : softmax(x['logits'])} for x in logits_from_file]
        logits_list = [softmax(x['logits']) for x in logits_from_file]

        
        
        list_episodes, list_shots, list_queries = GenFewShot(logits_dic , datasets = dataset, write_file=False)
        magnitudes = {'mag': [],'episodes':list_episodes, 'ord': []}
        for i in range(args.few_shot_runs):
            magnitude = torch.cat(list_shots[i], dim  = 0)
            nb_sample = magnitude.shape[0]
            mean_mag= torch.mean(magnitude, dim =  0)
            magnitudes['mag'].append(mean_mag)
            std_mag = torch.std(magnitude, dim =  0)
            ordered_mag = torch.argsort(mean_mag, descending=True) 
            magnitudes['ord'].append(ordered_mag)
            print('\n\n {} \n\n'.format(dataset))
            print('choice_classes = ',list_episodes['choice_classes'][i].tolist())
            print('number of shots per class', [len(x) for x in list_episodes['shots_idx'][i]])
            print_classes(ordered_mag, std_mag, mean_mag, nb_sample)
            #torch.save(ordered_mag,'finetuning/selections/runs/magnitude_selected_{}{}.pt'.format(dataset,i))
        magnitudes['mag'] = torch.stack(magnitudes['mag'])
        magnitudes['ord'] = torch.stack(magnitudes['ord'])
        torch.save(magnitudes,os.getcwd()+'/finetuning/selections/runs/magnitudes_test/magnitude_MD_test_{}.pt'.format(dataset))
        bin = np.ones(magnitudes['ord'].shape,dtype=np.bool_)
        for i in range(magnitudes['ord'].shape[0]):
            bin[i,magnitudes['ord'][i,:k].cpu().numpy()]=0
        #print(b[i])

        np.save(os.getcwd()+'/finetuning/selections/runs/magnitudes_test/binaryFS_test_MD_{}_{}.npy'.format(k,dataset), bin)


