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


def SNR(list_distrib):
    #print('n_ways distrib', len(list_distrib))
    n_ways = len(list_distrib)
    for i in range(n_ways):
        if list_distrib[i].shape[0]<=1: #make sure you have more than one shot
            return 0,0,0
    means = torch.stack([list_distrib[i].mean(0) for i in range(n_ways)])
    stds = [torch.norm(list_distrib[i].std(0)) for i in range(n_ways)]
    noise = np.mean(stds).item() 
    margin = torch.cdist(means, means).sum().item()/(n_ways*(n_ways-1))
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
    features = [torch.load(filename)]
    results = torch.zeros(len(features), 2)
    for i in range(len(features)):
        accs = []
        fake_acc = []
        snr = []
        feature = features[i]
        chance = []
        Generator = {'metadataset_omniglot':OmniglotGenerator, 'metadataset_imagenet':ImageNetGenerator}.get(datasets[i]['name'].replace('_train', '').replace('_test', '').replace('_validation', '') if datasets != None else datasets, EpisodicGenerator)
        generator = Generator(datasetName=None if datasets is None else datasets[i]["name"], num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        for run in tqdm(range(args.few_shot_runs)) if tqdm_verbose else range(args.few_shot_runs):
            shots = []
            queries = []
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=n_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
            shots, queries = generator.get_features_from_indices(feature, episode)
            chance.append(1/len(shots)) # = 1/n_ways
            accs.append(classifiers.evalFewShotRun(shots, queries))
            if 'snr' in proxy:
                snr.append(SNR(shots)[0])
            if 'fake_acc' in proxy:
                fake_data = fake_samples2(shots)
                fake_acc.append(classifiers.evalFewShotRun(shots, fake_data))
        accs = 100 * torch.tensor(accs)
        fake_acc = 100 * torch.tensor(fake_acc)
        chance = 100 * torch.tensor(chance)
        snr = torch.tensor(snr)
        return {'acc' : accs,'snr': snr, 'fake_acc' : fake_acc, 'chance' : chance}


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
        dist = torch.distributions.MultivariateNormal(loc  = means[i].float(), covariance_matrix= cov.float())
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

def get_baseline(filename,seed,n_shots, proxy):
    init_seed(seed)
    res_baseline = testFewShot_proxy(filename, datasets = None,n_shots = n_shots, proxy=proxy, tqdm_verbose = True)
    return res_baseline

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

def compare(dataset, seed = args.seed, n_shots = args.few_shot_shots, proxy = ''):
    N = 50
    init_seed(seed)
    filename_baseline = '/users2/libre/features/baselinemetadataset_'+dataset+'_test_features.pt'
    res_baseline = get_baseline(filename_baseline,seed,n_shots = n_shots, proxy = [proxy])
    L = np.zeros((N+1,2,len(res_baseline['acc'])))  #N+A and the two bottom lines are here to add the baseline amongst candidates
    L[N,0] = np.array(res_baseline['acc']) 
    L[N,1] = np.array(res_baseline[proxy])
    for i in tqdm(range(N)):
        init_seed(seed)
        filename = '/users2/libre/features/50/'+str(i)+'metadataset_'+dataset+'_test_features.pt'
        res = testFewShot_proxy(filename, datasets = None, n_shots = n_shots, proxy = [proxy] )
        L[i,0] = np.array(res['acc'])
        L[i,1] = np.array(res[proxy])
    print('dataset:',dataset, n_shots, 'shots', ';proxy:', proxy)
    print('random_backbone',np.take_along_axis(L[:,0],np.random.randint(0, N+1, L.shape[2] ).reshape(1,-1), axis = 0).mean().item())
    print('chance',res['chance'].mean().item()) 
    print('opti:', np.take_along_axis(L[:,0],L[:,1].argmax(0)[None,:], axis =0).mean())
    print('baseline:',res_baseline['acc'].mean().item())
    print('sanity check baseline',L[N,0,:].mean()) 
    print('max_possible', np.take_along_axis(L[:,0],L[:,0].argmax(0)[None,:], axis =0).mean())
    print('normalized proxy, normalized accuracy -> correlation')
    _,_ = plot_norm_correlation(L, plot=False, proxy= proxy)
    return res_baseline, L

if __name__ == "__main__":
    try:
        _,_=compare(dataset = args.target_dataset, proxy = args.proxy )
    except:
        target_dataset = eval(args.target_dataset)
        for dat in  target_dataset:
            print(dat)
            _,_=compare(dataset = dat, proxy = args.proxy )
            print('\n')