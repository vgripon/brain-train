from args import args
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import classifiers
import sys
from scipy.stats import kde
import random 
from few_shot_evaluation import EpisodicGenerator, ImageNetGenerator, OmniglotGenerator
from tqdm import tqdm
import random
from utils import *
from time import time
import json
from collections import defaultdict
import hashlib
import torch.nn as nn
#import hm_selection
import filelock
import itertools
from heuristics import *


if 'omniglot' in args.target_dataset:
    data_info_omniglot_file = os.path.join(args.dataset_path, 'omniglot.json')
    with open(data_info_omniglot_file) as f:
        data_info_omniglot = json.load(f)
    data_info_omniglot=data_info_omniglot['metadataset_omniglot_{}'.format(args.valtest)]


def testFewShot_proxy(filename, datasets = None, n_shots = 0, proxy = [], tqdm_verbose = False, QR = args.QR, use_classifier=False,episodes=None ):
    if episodes!=None:
        nb_episodes = len(episodes['shots_idx'])
    else:
        nb_episodes=args.few_shot_runs
        episodes = {'shots_idx' : [], 'queries_idx' : [], 'choice_classes' : []}
    if not os.path.isdir(filename):
        features = [torch.load(filename, map_location=args.device)]
        allow_classifier=False
    else:
        features = [torch.load(os.path.join(filename,'0metadataset_{0}_{1}_features.pt'.format(args.target_dataset, args.valtest)), map_location=args.device)]
        #first run just to get the genrator righ
        allow_classifier=True #we allow the use of support set tuned classifiers for each run. Such classifier do not exist if the file is unique for many runs (as in the if condition)
    for i in range(len(features)):
        accs = []
        fake_acc = []
        snr = []
        feature = features[i]
        chance = []
        loo = []
        soft,hard = [],[] 
        rankme, rankme_t = [],[]
        hnm=[]
        
        if datasets=='omniglot':
            Generator = OmniglotGenerator
            generator = Generator(datasetName='omniglot', num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
            generator.dataset = data_info_omniglot
        else:
            args.dataset_path = None
            Generator = EpisodicGenerator
            generator = Generator(datasetName=None, num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        for run in tqdm(range(nb_episodes)) if tqdm_verbose else range(nb_episodes):
            if episodes!=None:
                if os.path.isdir(filename):  #task informed setting
                    feature = torch.load(os.path.join(filename,str(run)+'metadataset_{0}_{1}_features.pt'.format(args.target_dataset, args.valtest)), map_location=args.device)
                episode = {'shots_idx' : episodes['shots_idx'][run], 'queries_idx' : episodes['queries_idx'][run], 'choice_classes' : episodes['choice_classes'][run]}

            else:
                shots = []
                queries = []
                init_seed(args.seed+run)
                episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=n_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries, max_queries = args.max_queries)
            
            
            shots, queries = generator.get_features_from_indices(feature, episode)
            chance.append(1/len(shots)) # = 1/n_ways
            if use_classifier and allow_classifier:
                file_classifier = os.path.join(filename, '../..', 'classifiers',args.target_dataset, 'classifier_finetune_'+str(run) )
                classifier = classifiers.simple_LR(shots[0][0].shape[0], len(shots)).to(args.device)
                classifier.load_state_dict(torch.load(file_classifier))
                perf = logit(shots, queries, classifier,episode)
            else:
                perf = classifiers.evalFewShotRun(shots, queries)

            accs.append(perf)
            if 'snr' in proxy:
                snr.append(SNR(shots)[0])
            if 'fake_acc' in proxy:
                #init_seed(args.seed)
                if QR:
                    #print('before QR', [x.std(0).mean() for x in queries])
                    fake_acc.append(QRsamplingtest(shots, queries, perf))
                else:
                    fake_data = fake_samples2(shots)
                    fake_acc.append(classifiers.evalFewShotRun(shots, fake_data))
            if 'loo' in proxy:
                loo.append(loo_shuffle(shots))
            if 'hard' in proxy:
                hard.append(classifiers.evalFewShotRun(shots, shots))
            if 'soft' in proxy:
                soft.append(confidence(shots))
            if 'rankme' in proxy:
                rankme.append(Rankme(shots))
            if 'rankme_t' in proxy:
                rankme_t.append(Rankme(shots, queries))

        accs = 100 * torch.tensor(accs)
        fake_acc = 100 * torch.tensor(fake_acc)
        chance = 100 * torch.tensor(chance)
        snr = torch.tensor(snr)
        loo = 100*torch.tensor(loo)
        hard = 100 * torch.tensor(hard)
        soft = 100 * torch.tensor(soft)
        rankme , rankme_t = torch.tensor(rankme), torch.tensor(rankme_t)
        return {'acc' : accs,'snr': snr, 'fake_acc'+QR*'QR'+args.isotropic*'isotropic'  : fake_acc, 'chance' : chance, 'loo' : loo, 'soft' : soft, 'hard': hard, 'rankme' : rankme, 'rankme_t' : rankme_t}




def compare(dataset, seed = args.seed, n_shots = args.few_shot_shots, proxy = '', save = False):
    if args.load_episodes!='':
        episodes = torch.load(args.load_episodes)['episodes']
        number_of_episode = len(episodes['shots_idx'])
    else:
        number_of_episode=args.few_shot_runs
    N = args.num_clusters
    out={}
    filename_baseline = args.baseline
    res_baseline = testFewShot_proxy(filename_baseline, datasets = dataset,n_shots = n_shots, proxy=proxy, tqdm_verbose = True,episodes=episodes)
    baseline=torch.zeros((2,number_of_episode))
    baseline[0]=res_baseline['acc']
    baseline[1]=res_baseline[proxy+args.QR*'QR'+args.isotropic*'isotropic']
    out['baseline']=baseline
    TA_scores = torch.zeros((N ,2,number_of_episode))
    for i in tqdm(range(N)):
        filename = eval(eval(args.competing_features))[i]
        res = testFewShot_proxy(filename, datasets = dataset, n_shots = n_shots, proxy = [proxy],episodes=episodes)
        TA_scores[i,0] = res['acc']
        TA_scores[i,1] = res[proxy+args.QR*'QR'+args.isotropic*'isotropic']
    if N>0:
        out['TA'] = TA_scores
    if args.fs_finetune!='':
        res_fn = testFewShot_proxy(args.fs_finetune, datasets = dataset, n_shots = n_shots, proxy = [proxy],use_classifier=args.use_classifier,episodes=episodes)
        fs_tuned=torch.zeros((2,number_of_episode))
        fs_tuned[0]=res_fn['acc']  #custom finetune is before-before last
        fs_tuned[1] = res_fn[proxy+args.QR*'QR'+args.isotropic*'isotropic']
        out['fs_tuned']=fs_tuned
    if args.cheated!='':
        res_cheated = testFewShot_proxy(args.cheated, datasets = dataset, n_shots = n_shots, proxy = [proxy],episodes=episodes)
        cheated=torch.zeros((2,number_of_episode))
        cheated[0] = res_cheated['acc'] #custom finetune is before-before last
        cheated[1] = res_cheated[proxy+args.QR*'QR'+args.isotropic*'isotropic']
        out['cheated']=cheated
    
    
    print(dataset, n_shots, 'n_shots', 'proxy', proxy)
    if N>0:
        selection_pool = torch.cat((TA_scores,baseline.unsqueeze(0)),dim=0)
        nb_backbones=selection_pool.shape[0]
        random_indices = torch.randint(0,nb_backbones,(number_of_episode,))
        random_backbone = torch.diagonal(selection_pool[random_indices, 0])
        print_metric(random_backbone.ravel(), 'random_backbone')
        opti_indices = torch.argmax(selection_pool[:, 1], dim=0)         # Take values along the specified axis (axis 0)
        opti = torch.diagonal(selection_pool[opti_indices, 0])
        print_metric(opti.ravel(), 'opti: ')
        max_indices = torch.argmax(selection_pool[:, 0], dim=0)         # Take values along the specified axis (axis 0)
        max_possible = torch.diagonal(selection_pool[max_indices, 0])
        print_metric(max_possible.ravel(),'max_possible: ')


    print_metric(res_baseline['chance'] , 'chance')
    baseline = res_baseline['acc']
    print_metric(baseline,'baseline: ')


    if args.fs_finetune!='':
        print_metric(res_fn['acc'],'finetuned: ')
    if args.cheated!='':
        print_metric(res_cheated['acc'],'cheated: ')
    print_metric(out['baseline'][0],'sanity check baseline: ')
    if N>0:
        _,_ = plot_norm_correlation(selection_pool, plot=False, proxy= proxy)
        backbones = eval(eval(args.competing_features))
    else:
        backbones = ""
    if save:
        save_results(out, dataset, proxy+'QR'*args.QR+'isotropic'*args.isotropic, res_baseline['chance'], episodes = episodes,backbones=backbones )
    return  out 



def save_results(out,dataset, proxy, chance, episodes,backbones):
    file = args.out_file
    lock = filelock.FileLock(file+".lock")
    with lock:
        if not os.path.isfile(file):
            d={'episodes': {}, 'hash_episode' : {}, 'backbones': {}}
            torch.save(d,file)
        else:
            d = torch.load(file)
        h = hashlib.md5(str(episodes).encode('utf-8')).hexdigest()
        print(h)
        d['episodes'][dataset] = episodes
        d['hash_episode'][dataset] = h
        d['backbones'][dataset]= str(backbones)
        if proxy in d.keys():
            d[proxy][dataset] = {'data' : out, 'info' : str(args), 'chance' : chance, 'hash_episode' : h}
        else:
            d[proxy] = {dataset:{'data' : out, 'info' : str(args),'chance' : chance, 'hash_episode' : h}}
        torch.save(d, file)




if __name__ == "__main__":
    try:
        _,_=compare(dataset = args.target_dataset, proxy = args.proxy, save = True  )
    except:
        ta = time()
        target_dataset = eval(args.target_dataset)
        for dat in  target_dataset:
            print(dat)
            _,_=compare(dataset = dat, proxy = args.proxy, save = True )
            print('\n')
        tb = time()
        print('TOTAL TIME : ', tb-ta)
