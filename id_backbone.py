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

real_dp = args.dataset_path
data_info_omniglot_file = os.path.join(args.dataset_path, 'omniglot_test.json')
with open(data_info_omniglot_file) as f:
    data_info_omniglot = json.load(f)

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

def testFewShot_proxy(filename, datasets = None, n_shots = 0, proxy = [], tqdm_verbose = False ):
    features = [torch.load(filename, map_location=args.device)]
    for i in range(len(features)):
        accs = []
        fake_acc = []
        snr = []
        feature = features[i]
        chance = []
        loo = []
        if datasets=='omniglot':
            Generator = OmniglotGenerator
            generator = Generator(datasetName=None, num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
            generator.dataset = data_info_omniglot
        else:
            args.dataset_path = None
            Generator = EpisodicGenerator
            generator = Generator(datasetName=None, num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        for run in tqdm(range(args.few_shot_runs)) if tqdm_verbose else range(args.few_shot_runs):
            shots = []
            queries = []
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=n_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
            #print('1st shot', episode['shots_idx'][0][0])
            shots, queries = generator.get_features_from_indices(feature, episode)
            #print('1st shot', shots[0][0], '1st query' , queries[0][0])
            chance.append(1/len(shots)) # = 1/n_ways
            accs.append(classifiers.evalFewShotRun(shots, queries))
            if 'snr' in proxy:
                snr.append(SNR(shots)[0])
            if 'fake_acc' in proxy:
                init_seed(args.seed)
                fake_data = fake_samples2(shots)
                fake_acc.append(classifiers.evalFewShotRun(shots, fake_data))
            if 'loo' in proxy:
                loo.append(leave_one_out(shots))
        accs = 100 * torch.tensor(accs)
        fake_acc = 100 * torch.tensor(fake_acc)
        chance = 100 * torch.tensor(chance)
        snr = torch.tensor(snr)
        loo = 100*torch.tensor(loo)
        return {'acc' : accs,'snr': snr, 'fake_acc' : fake_acc, 'chance' : chance, 'loo' : loo}


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

def fake_samples2(list_distrib, n_sample = 100):
    n_ways = len(list_distrib)
    means = torch.stack([list_distrib[i].mean(0) for i in range(n_ways)])
    centered = [list_distrib[i]-means[i] for i in range(n_ways)]
    covs=[]
    fake_samples  =[]
    for i in range(n_ways):
        x= centered[i]
        if x.shape[0]!=1:
            cov = (torch.matmul(x.T, x) + torch.eye(x.shape[1]) * 0.001) / (x.shape[0]-1)
            check = torch.linalg.cholesky_ex(cov).info.eq(0).unsqueeze(0)
            covs.append(cov)
        else:
            cov = torch.eye(x.shape[1])
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

def compare(dataset, seed = args.seed, n_shots = args.few_shot_shots, proxy = ''):
    N = 50
    init_seed(seed)
    filename_baseline = os.path.join(args.save_features_prefix,'baselinemetadataset_'+dataset+'_test_features.pt')
    res_baseline = testFewShot_proxy(filename_baseline, datasets = dataset,n_shots = n_shots, proxy=proxy, tqdm_verbose = True)
    L = np.zeros((N+1,2,len(res_baseline['acc'])))  #N+A and the two bottom lines are here to add the baseline amongst candidates
    L[N,0] = np.array(res_baseline['acc']) 
    L[N,1] = np.array(res_baseline[proxy])
    for i in tqdm(range(N)):
        init_seed(seed)
        filename = os.path.join(args.save_features_prefix,'50/'+str(i)+'metadataset_'+dataset+'_test_features.pt')
        res = testFewShot_proxy(filename, datasets = dataset, n_shots = n_shots, proxy = [proxy])
        L[i,0] = np.array(res['acc'])
        L[i,1] = np.array(res[proxy])
    print(dataset, n_shots, 'n_shots', 'proxy', proxy)
    random_backbone = np.take_along_axis(L[:,0],np.random.randint(0, N+1, L.shape[2] ).reshape(1,-1), axis = 0)
    print_metric(random_backbone, 'random_backbone')
    print_metric(res['chance'] , 'chance')
    opti = np.take_along_axis(L[:,0],L[:,1].argmax(0)[None,:], axis =0)
    print_metric(opti, 'opti: ')
    baseline = res_baseline['acc']
    print_metric(baseline,'baseline: ')
    print_metric(L[N,0,:],'sanity check baseline: ')
    max_possible = np.take_along_axis(L[:,0],L[:,0].argmax(0)[None,:], axis =0)
    print_metric(max_possible.ravel(),'max_possible: ')
    _,_ = plot_norm_correlation(L, plot=False, proxy= proxy)
    return res_baseline, L

def leave_one_out(shots):
    n_ways = len(shots)
    nb_shots =  np.array([shots[j].shape[0] for j in range(n_ways)])
    max_shots = np.max(nb_shots)
    acc = 0
    for i in range(max_shots):
        pop_index = [i%nb_shots[j] for j in range(n_ways)]
        q = [shots[j][pop_index[j]].unsqueeze(0) for j in range(n_ways)]
        shots_loo = [ torch.cat((shots[j][:pop_index[j]],shots[j][pop_index[j]+1:]))  for j in range(n_ways) ] 
        acc += classifiers.evalFewShotRun(shots_loo, q)/max_shots
    return acc


if __name__ == "__main__":
    try:
        _,_=compare(dataset = args.target_dataset, proxy = args.proxy )
    except:
        target_dataset = eval(args.target_dataset)
        for dat in  target_dataset:
            print(dat)
            _,_=compare(dataset = dat, proxy = args.proxy )
            print('\n')