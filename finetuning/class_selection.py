import sys
import torch
import torch.nn as nn
popos = True
if popos:
    sys.path.append('/home/raphael/Documents/brain-train')
else:
    sys.path.append('/homes/r21lafar/Documents/brain-train')

from few_shot_evaluation import EpisodicGenerator, ImageNetGenerator, OmniglotGenerator
from args import args
import os
import classifiers
import sys
import random 
from tqdm import tqdm
import numpy as np

import json 
with open(os.path.join(args.dataset_path,'datasets_subdomain.json')) as f:
    dataset_json = json.load(f)
print(dataset_json['metadataset_omniglot_test'].keys())
def print_classes(ordered_acc, std, accuracy, nb_sample):
    conf_inter = 1.96* std / np.sqrt(nb_sample)
    print('\n Best classes are \n' )
    for x in ordered_acc[:10]:
        print("\t{:.3f} ±{:.3f}".format(accuracy[x].item(), conf_inter[x].item()  ), dataset_json['metadataset_imagenet_train']['name_classes'][x])
    print('\n Worst classes are \n' )
    for x in ordered_acc[-10:]:
        print("\t{:.3f} ±{:.3f}".format(accuracy[x].item(), conf_inter[x].item()  ), dataset_json['metadataset_imagenet_train']['name_classes'][x])


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



def testFewShot(features, datasets = None, write_file=False):
    results = torch.zeros(len(features), 2)
    accs = []
    if datasets=='omniglot':
            Generator = OmniglotGenerator
            generator = Generator(datasetName=None, num_elements_per_class= [feat['features'].shape[0] for feat in features], dataset_path=args.dataset_path)
            generator.dataset = dataset_json['metadataset_omniglot_validation']
            sys.exit(0)
    else:
        args.dataset_path = None
        Generator = EpisodicGenerator
        generator = Generator(datasetName=None, num_elements_per_class= [feat['features'].shape[0] for feat in features], dataset_path=args.dataset_path)
    for run in range(args.few_shot_runs):
        shots = []
        queries = []
        episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
        shots, queries = generator.get_features_from_indices(features, episode)
        accs.append(ncm_dim(shots,queries))
    accs = 100 * torch.stack(accs)
    return accs



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




if __name__=='__main__':
    for dataset in ['cub', 'aircraft', 'dtd', 'mscoco', 'fungi', 'omniglot', 'traffic_signs', 'vgg_flower']:
        print('\n\n {} \n\n'.format(dataset))
        if popos:
            logits_from_file = torch.load('/home/raphael/Documents/models/logits_{}_val.pt'.format(dataset))
        else:
            logits_from_file = torch.load('/users2/libre/raphael/logits_{}_val.pt'.format(dataset))

        softmax = nn.Softmax(dim = 1)
        #feats = torch.stack([x['logits'] for x in logits_from_file])
        #logits = softmax(feats)
        #logits_list = [{'features' : logits[i]  } for i in range(logits.shape[0])]
        logits_dic = [{'features' : softmax(x['logits'])} for x in logits_from_file]
        logits_list = [softmax(x['logits']) for x in logits_from_file]

        # MAGNITUDE SELECTION (ACCURACY BY DIM WITH METADATASET SAMPLING)
        magnitude = torch.cat(logits_list)
        nb_sample = magnitude.shape[0]
        mean_mag= torch.mean(magnitude, dim =  0)
        std_mag = torch.std(magnitude, dim =  0)
        ordered_mag = torch.argsort(mean_mag, descending=True) 
        print_classes(ordered_mag, std_mag, mean_mag, nb_sample)
        torch.save(ordered_mag,'finetuning/selections/magnitude_selected_{}.pt'.format(dataset))

        # HARD SELECTION (ACCURACY BY DIM WITH ALL SAMPLES IN SUPPORT SET)
        #ordered_acc, std, accuracy = measure_acc_by_dim(logits)
        #torch.save(ordered_mag,'finetuning/selections/hard_selected_{}.pt'.format(dataset))
        #print_classes(ordered_acc, std, accuracy)

        # NCM SELECTION (ACCURACY BY DIM WITH METADATASET SAMPLING)
        results = testFewShot(logits_dic , datasets = dataset, write_file=False)
        std = results.std(0)
        mean = results.mean(0)
        ordered_acc =mean.argsort(descending=True)
        torch.save(ordered_mag,'finetuning/selections/NCM_selected_{}.pt'.format(dataset))
        print_classes(ordered_acc, std, mean, nb_sample = args.few_shot_runs)



