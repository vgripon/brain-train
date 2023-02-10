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

load_episode = args.load_episodes!=''
load_fs_fine = args.fs_finetune!=''
if load_episode ^ load_fs_fine:
    print('\n \n load_episode and fs_finetune work together you forgot one \n\n' )
    sys.exit(0)

real_dp = args.dataset_path
if 'omniglot' in args.target_dataset:
    data_info_omniglot_file = os.path.join(args.dataset_path, 'omniglot.json')
    with open(data_info_omniglot_file) as f:
        data_info_omniglot = json.load(f)
    data_info_omniglot=data_info_omniglot['metadataset_omniglot_{}'.format(args.valtest)]

def SNR(list_distrib):
    #print('n_ways distrib', len(list_distrib))
    n_ways = len(list_distrib)
    means = torch.stack([list_distrib[i].mean(0) for i in range(n_ways)])
    stds = [torch.norm(list_distrib[i].std(0)).item()  for i in range(n_ways)]
    noise = np.mean(stds)
    margin = torch.cdist(means, means).sum().item()/(n_ways*(n_ways-1))
    for i in range(n_ways):
        if list_distrib[i].shape[0]<=1: #make sure you have more than one shot
            return margin,margin,0
    return margin/noise , margin , noise

def SNR_mean_couple(list_distrib):
    n_ways = len(list_distrib)
    for i in range(n_ways):
        if list_distrib[i].shape[0]<=1: #make sure you have more than one shot
            return 0,0,0
    l_snr,l_margin,l_noise = [],[],[]
    for i in range(n_ways):
        for j in range(i+1, n_ways):
            snr , margin , noise = SNR([list_distrib[i],list_distrib[j]])
            l_snr.append(snr)
            l_margin.append(margin)
            l_noise.append(noise)
    snr= np.array(l_snr).mean()
    margin = np.array(margin).mean()
    noise = np.array(l_noise).mean()
    return margin/noise , margin , noise

def testFewShot_proxy(filename, datasets = None, n_shots = 0, proxy = [], tqdm_verbose = False, QR = args.QR ):
    if not os.path.isdir(filename):
        features = [torch.load(filename, map_location=args.device)]
    else:
        features = [torch.load(os.path.join(filename,'0metadataset_{0}_{1}_features.pt'.format(args.target_dataset, args.valtest)), map_location=args.device)]
        #first run just to get the genrator right
    for i in range(len(features)):
        accs = []
        fake_acc = []
        snr = []
        feature = features[i]
        chance = []
        loo = []
        soft,hard = [],[] 
        rankme, rankme_t = [],[]
        episodes = {'shots_idx' : [], 'queries_idx' : [], 'choice_classes' : []}
        if datasets=='omniglot':
            Generator = OmniglotGenerator
            generator = Generator(datasetName='omniglot', num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
            generator.dataset = data_info_omniglot
        else:
            args.dataset_path = None
            Generator = EpisodicGenerator
            generator = Generator(datasetName=None, num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        if args.load_episodes!='':
            episodes = torch.load(args.load_episodes)['episodes']
        for run in tqdm(range(args.few_shot_runs)) if tqdm_verbose else range(args.few_shot_runs):
            if args.load_episodes=='':
                shots = []
                queries = []
                init_seed(args.seed+run)
                episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=n_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries, max_queries = args.max_queries)
            else:
                if os.path.isdir(filename):
                    feature = torch.load(os.path.join(filename,str(run)+'metadataset_{0}_{1}_features.pt'.format(args.target_dataset, args.valtest)), map_location=args.device)
                episode = {'shots_idx' : episodes['shots_idx'][run], 'queries_idx' : episodes['queries_idx'][run], 'choice_classes' : episodes['choice_classes'][run]}
            #if run ==1:
            #    print('1st run shot', episode['shots_idx'][0])
            #    print('1st run classes', episode['choice_classes'])
            shots, queries = generator.get_features_from_indices(feature, episode)
            #print('1st shot', shots[0][0], '1st query' , queries[0][0])
            chance.append(1/len(shots)) # = 1/n_ways
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
            for epi in episode.items():
                episodes[epi[0]].append(epi[1])
        accs = 100 * torch.tensor(accs)
        fake_acc = 100 * torch.tensor(fake_acc)
        chance = 100 * torch.tensor(chance)
        snr = torch.tensor(snr)
        loo = 100*torch.tensor(loo)
        rankme , rankme_t = torch.tensor(rankme), torch.tensor(rankme_t)
        return {'episodes': episodes , 'acc' : accs,'snr': snr, 'fake_acc'+QR*'QR'+args.isotropic*'isotropic'  : fake_acc, 'chance' : chance, 'loo' : loo, 'soft' : soft, 'hard': hard, 'rankme' : rankme, 'rankme_t' : rankme_t}


def Rankme(shots , queries = None, centroids = args.centroids):
    if queries == None:
        if centroids:
            Z = torch.cat([shots[i].mean(0).unsqueeze(0) for i in range(len(shots))]).reshape(1,-1,640)
        else:
            Z = torch.cat([shots[i] for i in range(len(shots))]).reshape(1,-1,640)
    else:
        Z1 = torch.cat([shots[i] for i in range(len(shots))]).reshape(1,-1,640)
        Z2 = torch.cat([queries[i] for i in range(len(queries))]).reshape(1,-1,640)
        Z = torch.cat((Z1,Z2),dim =1)
    u, s, v = torch.svd(Z,compute_uv = False)
    pk = s/torch.sum(s) + 1e-7
    m = nn.Softmax(dim=1)
    pk = m(pk*args.temperature)
    pk = pk[:min(u.shape[1], v.shape[1])]
    entropy = -torch.sum(pk*torch.log(pk))
    return torch.exp(entropy).item()


def confidence(shots):
    n_ways = len(shots)
    centroids = torch.stack([shotClass.mean(dim = 0) for shotClass in shots])
    score = 0
    total = 0
    for i, queriesClass in enumerate(shots):
        distances = torch.norm(queriesClass.unsqueeze(1) - centroids.unsqueeze(0), dim = 2)
        sims = torch.softmax((- distances * args.temperature).reshape(-1, n_ways), dim = 1)
        score += torch.max(sims, dim = 1)[0].mean().cpu()
    return score


def QRsamplingtest(shots, queries_for_sanity_check, perf_for_sanity_check):
    n_ways = len(shots)
    means = torch.stack([shots[i].mean(0) for i in range(n_ways)])
    Q = dimReduction(means)
    reduced = [torch.einsum('nd, wd -> nw',shots[i], Q) for i in range(n_ways)]
    reduced_queries_for_sanity_check = [torch.einsum('nd, wd -> nw',queries_for_sanity_check[i], Q) for i in range(n_ways)]
    #print('after QR', [x.std(0).mean() for x in reduced_queries_for_sanity_check])
    #if not perf_for_sanity_check == classifiers.evalFewShotRun(reduced, reduced_queries_for_sanity_check):
    #    print(perf_for_sanity_check.item() ,classifiers.evalFewShotRun(reduced, reduced_queries_for_sanity_check).item(), 'should be same')
    fake_samples = fake_samples2(reduced)
    perf = classifiers.evalFewShotRun(reduced, fake_samples)
    return perf

def dimReduction(means):
    perm = torch.arange(means.shape[0])-1
    LDAdirections = (means-means[perm])[:-1]
    Q, R = torch.linalg.qr(LDAdirections.T)
    return Q.T

def fake_samples(list_distrib, n_sample = 100):
    n_ways = len(list_distrib)
    means = torch.stack([list_distrib[i].mean(0) for i in range(n_ways)])
    centered = [list_distrib[i]-means[i] for i in range(n_ways)]
    covs = []
    for i in range(n_ways):
        if centered[i].shape[0]!=1:
            covs.append(np.cov(centered[i].T.detach().numpy()))
        else:
            covs.append(np.eye(centered[i].shape[1]))
    fake_samples = [torch.from_numpy(np.random.multivariate_normal(means[i], covs[i], n_sample)) for i in range(n_ways)]
    return fake_samples

def fake_samples2(list_distrib, n_sample = 100 ):
    if args.isotropic and args.QR:
        alpha = 1.5
    elif args.QR:
        alpha = 1.3
    else:
        alpha = 1.7
    if args.num_clusters == 50:
        alpha*=0.7
    n_ways = len(list_distrib)
    means = torch.stack([list_distrib[i].mean(0) for i in range(n_ways)])
    centered = [list_distrib[i]-means[i] for i in range(n_ways)]
    covs=[]
    fake_samples  =[]
    for i in range(n_ways):
        x= centered[i]
        if x.shape[0]!=1:
            cov = (torch.matmul(x.T, x) + torch.eye(x.shape[1]).to(args.device) * 0.001) / (x.shape[0]-1)
            #check = torch.linalg.cholesky_ex(cov).info.eq(0).unsqueeze(0)
            if args.isotropic:
                #cov = torch.diag(x.std(dim=0))
                cov = torch.eye(x.shape[1])*alpha
            covs.append(cov)
        else:
            cov = torch.eye(x.shape[1])*alpha
            covs.append(cov)
        dist = torch.distributions.MultivariateNormal(loc  = means[i].float().to(device  = args.device), covariance_matrix= cov.float().to(device  = args.device))
        fake_samples.append(dist.rsample(torch.Size([n_sample])))
    return fake_samples

def init_seed(seed = args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def plot_norm_correlation(L, plot=True, proxy= ''):
    stds = np.std(L,axis = 0)
    means = np.mean(L,axis = 0)
    norm_L = (L-means)/stds
    norm_L = np.nan_to_num(norm_L, nan=0)
    y = norm_L[:,0].ravel()
    x = norm_L[:,1].ravel()
    rho = np.corrcoef(x,y)
    print('pearson correlation', np.round(rho[0,1],3))
    if plot:
        plt.figure()
        nbins=300
        k = kde.gaussian_kde([x,y])
        xi, yi = np.mgrid[x.min():x.max():nbins*1j, y.min():y.max():nbins*1j]
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        plt.xlabel(proxy)
        plt.ylabel('acc')
    return norm_L, rho

def print_metric(metric_tensor, name = ''):
    low,up = confInterval(metric_tensor)
    print(name, "\t{:.3f} ±{:.3f} (conf. [{:.3f}, {:.3f}])".format(metric_tensor.mean().item(), metric_tensor.std().item(), low, up))

def compare(dataset, seed = args.seed, n_shots = args.few_shot_shots, proxy = '', save = False):
    N = args.num_clusters
    shift = 0
    if args.fs_finetune!='':
        shift=1
    filename_baseline = os.path.join('/hpcfs/users/a1881717/work_dir/baseline/features/'+dataset+'/featmetadataset_'+ dataset+'_'+args.valtest+'_features.pt' )
    res_baseline = testFewShot_proxy(filename_baseline, datasets = dataset,n_shots = n_shots, proxy=proxy, tqdm_verbose = True)
    L = np.zeros((N+1+shift,2,len(res_baseline['acc'])))  #N+A and the two bottom lines are here to add the baseline amongst candidates
    L[N,0] = np.array(res_baseline['acc']) 
    L[N,1] = np.array(res_baseline[proxy+args.QR*'QR'+args.isotropic*'isotropic'])
    episodes = res_baseline['episodes']
    for i in tqdm(range(N)):
        filename = eval(eval(args.competing_features))[i]
        res = testFewShot_proxy(filename, datasets = dataset, n_shots = n_shots, proxy = [proxy])
        L[i,0] = np.array(res['acc'])
        L[i,1] = np.array(res[proxy+args.QR*'QR'+args.isotropic*'isotropic'])
    if args.fs_finetune!='':
        filename = args.fs_finetune
        res = testFewShot_proxy(filename, datasets = dataset, n_shots = n_shots, proxy = [proxy])
        L[N+1,0] = np.array(res_baseline['acc']) ### updated the position of the baseline
        L[N+1,1] = np.array(res_baseline[proxy+args.QR*'QR'+args.isotropic*'isotropic'])
        L[N,0] = np.array(res['acc'])  #custom finetune is before last
        L[N,1] = np.array(res[proxy+args.QR*'QR'+args.isotropic*'isotropic'])
    print(dataset, n_shots, 'n_shots', 'proxy', proxy)
    random_backbone = np.take_along_axis(L[:,0],np.random.randint(0, N+1, L.shape[2] ).reshape(1,-1), axis = 0)
    print_metric(random_backbone.ravel(), 'random_backbone')
    print_metric(res['chance'] , 'chance')
    opti = np.take_along_axis(L[:,0],L[:,1].argmax(0)[None,:], axis =0)
    print_metric(opti.ravel(), 'opti: ')
    baseline = res_baseline['acc']
    print_metric(baseline,'baseline: ')
    print_metric(L[N,0,:],'sanity check baseline: ')
    max_possible = np.take_along_axis(L[:,0],L[:,0].argmax(0)[None,:], axis =0)
    print_metric(max_possible.ravel(),'max_possible: ')
    _,_ = plot_norm_correlation(L, plot=False, proxy= proxy)
    if save:
        save_results(L, dataset, proxy+'QR'*args.QR+'isotropic'*args.isotropic, res['chance'], episodes = episodes, backbones = eval(eval(args.competing_features))+[args.fs_finetune]+[filename_baseline])
    return res_baseline, L 

def save_results(L,dataset, proxy, chance, episodes,backbones):
    N = args.num_clusters
    if args.fs_finetune=='':
        file = '/hpcfs/users/a1881717/work_dir/vis/dFS'+str(N)+'.pt'
    else:
        file = '/hpcfs/users/a1881717/work_dir/vis/d'+str(N)+'.pt'
    if not os.path.isfile(file):
        d={'episodes': {}, 'hash_episode' : {}}
        torch.save(d,file)
    else:
        d = torch.load(file)
    h = len(str(episodes))
    h = hashlib.md5(str(episodes).encode('utf-8')).hexdigest()
    print(h)
    if (dataset not in d['episodes'].keys()) or (dataset in  d['episodes'].keys() and len(str(d['episodes'][dataset])) != h) :
        d['episodes'][dataset] = episodes
        d['hash_episode'][dataset] = h
    if proxy in d.keys() and dataset in d[proxy].keys():
        print('overwriting',dataset, proxy)
        if args.fs_finetune=='':
            file2 = '/hpcfs/users/a1881717/work_dir/vis/dFS'+str(N)+'_2.pt'
        else:
            file2 = '/hpcfs/users/a1881717/work_dir/vis/d'+str(N)+'_2.pt'
        torch.save(d,file2)
    d['backbones']= backbones
    if proxy in d.keys():
        d[proxy][dataset] = {'data' : torch.from_numpy(L), 'info' : str(args), 'nb_runs' : args.few_shot_runs, 'chance' : chance, 'hash_episode' : h}
    else:
        d[proxy] = {dataset:{'data' : torch.from_numpy(L), 'info' : str(args), 'nb_runs' : args.few_shot_runs,'chance' : chance, 'hash_episode' : h}}
    torch.save(d, file)

def leave_one_out(shots):
    n_ways = len(shots)
    nb_shots =  np.array([shots[j].shape[0] for j in range(n_ways)])
    print('nb_shots' , nb_shots )
    max_shots = np.max(nb_shots)
    acc = 0
    for i in range(max_shots):
        pop_index = [i%nb_shots[j] for j in range(n_ways)]
        q = [shots[j][pop_index[j]].unsqueeze(0) for j in range(n_ways)]
        shots_loo = [ torch.cat((shots[j][:pop_index[j]],shots[j][pop_index[j]+1:]))  for j in range(n_ways) ] 
        print('shots_loo' , [len(x) for x in shots_loo] )
        acc += classifiers.evalFewShotRun(shots_loo, q)/max_shots
    return acc

def leave_one_out_2(shots):
    n_ways = len(shots)
    nb_shots =  np.array([shots[j].shape[0] for j in range(n_ways)])
    max_shots = np.max(nb_shots)
    acc = 0
    for i in range(max_shots-1):
        q,shots_loo=[],[]
        for j in range(n_ways):
            if i >= nb_shots[j]-1:
                q.append([])
                shots_loo.append(shots[j])
            else:
                q.append(shots[j][i].unsqueeze(0))
                shots_loo.append(torch.cat((shots[j][:i],shots[j][i+1:])))
        acc += classifiers.evalFewShotRun(shots_loo, q)/max_shots
    return acc

def loo_shuffle(shots,num_iterations=10):
    results = []
    if [len(shot) for shot in shots]==[1 for shot in shots]:
        print('loo cannot process this run')
        return np.random.random(1)[0]
    for i in range(num_iterations):
        new_shots = []
        val_query = []
        for shot in shots:
            n = shot.shape[0]
            shuffled_shot = shot[torch.randperm(n)] if n > 1 else shot
            if n>1:
                new_shots.append(shuffled_shot[1:])
                val_query.append(shuffled_shot[0].unsqueeze(0))
            if n==1:
                new_shots.append(shot)
                val_query.append([])
        results.append(classifiers.evalFewShotRun(new_shots, val_query).item())
    return np.array(results).mean()


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
