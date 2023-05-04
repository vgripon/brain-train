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
                val_query.append(shot)  # will give 100% accuracy but can't do any better in 1shot
        results.append(classifiers.evalFewShotRun(new_shots, val_query).item())
    return np.array(results).mean()


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


def SNR2(list_distrib):
    #print('n_ways distrib', len(list_distrib))
    n_ways = len(list_distrib)
    means = torch.stack([list_distrib[i].mean(0) for i in range(n_ways)])
    stds = [torch.norm(list_distrib[i].std(0)).item()  for i in range(n_ways)]
    noise = np.mean(stds)
    margin = torch.cdist(means, means)
    margin_noz=margin[margin!=0.0]
    d_min =  torch.mean(margin_noz)
    return d_min, d_min, d_min 

def torch_gaussian_kde(xy, bandwidth=0.1):
    xy = xy.T.unsqueeze(1)
    diff = xy - xy.T
    norm = torch.sqrt(torch.sum(diff * diff, dim=-1))
    kde = torch.mean(torch.exp(-0.5 * (norm / bandwidth) ** 2), dim=1) / (bandwidth * np.sqrt(2 * np.pi))
    return kde

def print_metric(metric_tensor, name = ''):
    low,up = confInterval(metric_tensor)
    print(name, "\t{:.3f} ±{:.3f} (conf. [{:.3f}, {:.3f}])".format(metric_tensor.mean().item(), metric_tensor.std().item(), low, up))


def plot_norm_correlation(L, plot=True, proxy= ''):
    stds = torch.std(L,axis = 0)
    means = torch.mean(L,axis = 0)
    norm_L = (L-means)/stds
    norm_L = torch.nan_to_num(norm_L, nan=0)
    norm_L[ abs(norm_L) > 1e10] = 0 
    index_best = norm_L[:,1].argmax(0)
    sigma = torch.diagonal(norm_L[index_best,0])
    print_metric(sigma, 'Sigma Selected Backbone :')
    y = norm_L[:,0].ravel()
    x = norm_L[:,1].ravel()
    rho = torch.corrcoef(torch.stack((x,y),dim=0))
    print('pearson correlation', np.round(rho[0,1],3))
    if plot:
        plt.figure()
        nbins=300
        xi, yi = torch.meshgrid(torch.linspace(x.min(), x.max(), nbins), torch.linspace(y.min(), y.max(), nbins))
        xy = torch.stack([xi.flatten(), yi.flatten()], dim=0)
        zi = torch_gaussian_kde(xy)
        plt.pcolormesh(xi, yi, zi.reshape(xi.shape), shading='auto')
        plt.xlabel(proxy)
        plt.ylabel('acc')
    return norm_L, rho

def init_seed(seed = args.seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def logit(shots, queries, classifier,episode, batch_size = 128):
    device = shots[0].device
    order_target = reassign_numbers([i.item() for i in episode['choice_classes']])
    target = torch.cat([torch.Tensor([c]*len(queries[i])) for i,c in enumerate(order_target)]).long().to(device)
    #target = torch.cat([torch.Tensor([c]*len(shots[c])) for c in range(len(shots))]).long().to(device)
    flat_queries = torch.cat(queries)
    predictions = torch.zeros(len(target)).to(device)
    for b in range(len(flat_queries)//batch_size +1):
        predictions[b*batch_size:(b+1)*batch_size] = classifier(flat_queries[b*batch_size:(b+1)*batch_size]).argmax(dim=1)
    acc = (target == predictions).float().mean()
    return acc

def reassign_numbers(lst):
    remap = dict(zip(set(lst), itertools.count()))
    return [remap[i] for i in lst]

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