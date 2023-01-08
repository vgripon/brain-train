### this file describes the criterions used to train architectures
### they have to be thought of as loss functions in a broad way, including a fully connected layer for some of them
### they are typically composed on top of a backbone, that is seen as a feature extractor
### it thus takes the form of a nn module, even if some do not contain any parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np
from args import args


### Logistic Regression module, which is the classic way to train deep models for classification
class LR(nn.Module):
    def __init__(self, inputDim, numClasses, backbone=None):
        super(LR, self).__init__()
        self.fc = nn.Linear(inputDim, numClasses)
        self.fcRotations = nn.Linear(inputDim, 4)
        self.criterion = nn.CrossEntropyLoss() if args.label_smoothing == 0 else LabelSmoothingLoss(numClasses, args.label_smoothing)
        self.backbone = backbone
    def forward(self, backbone, dataStep, y, lr=False, rotation=False, mixup=False, manifold_mixup=False):
        lbda, perm, mixupType = None, None, None
        loss,score, multiplier = 0., torch.zeros(1), 1
        if mixup or manifold_mixup:
            multiplier = 0.5
            perm = torch.randperm(dataStep.shape[0])
            if mixup:
                lbda = random.random()
                mixupType = "mixup"
            else:
                lbda = np.random.beta(2,2)
                mixupType = "manifold mixup"
        yRotations = None
        if rotation:
            bs = dataStep.shape[0] // 4
            targetRot = torch.LongTensor(dataStep.shape[0]).to(args.device)
            targetRot[:bs] = 0
            dataStep[bs:] = dataStep[bs:].transpose(3,2).flip(2)
            targetRot[bs:2*bs] = 1
            dataStep[2*bs:] = dataStep[2*bs:].transpose(3,2).flip(2)
            targetRot[2*bs:3*bs] = 2
            dataStep[3*bs:] = dataStep[3*bs:].transpose(3,2).flip(2)
            targetRot[3*bs:] = 3
            yRotations = targetRot
        if backbone !=None:
            x = backbone(dataStep, mixup = mixupType, lbda = lbda, perm = perm)
        else:
            x = dataStep
        if lr or mixup or manifold_mixup:
            output = self.fc(x)
            decision = output.argmax(dim = 1)
            score = (decision - y == 0).float().mean()
            loss = self.criterion(output, y)
            multiplier = 0.5
        if lbda is not None:
            loss = lbda * loss + (1 - lbda) * self.criterion(output, y[perm])
            score = lbda * score + (1 - lbda) * (decision - y[perm] == 0).float().mean()
        if yRotations is not None:
            outputRotations = self.fcRotations(x)
            loss = multiplier * (loss + (self.criterion(outputRotations, yRotations) if lbda == None else (lbda * self.criterion(outputRotations, yRotations) + (1 - lbda) * self.criterion(outputRotations, yRotations[perm]))))
        return loss, score

### MultiLabel BCE
class MultiLabelBCE(nn.Module):
    def __init__(self, inputDim, numClasses):
        super(MultiLabelBCE, self).__init__()
        self.fc = nn.Linear(inputDim, numClasses)
        if args.audio:
            weights = torch.load(args.dataset_path + "audioset/audioset/processed/weight.pt")
            weights = (1 - weights) / weights
        else:
            weights = torch.ones(numClasses)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight = weights)

    def forward(self, x, y, yRotations = None, lbda = None, perm = None):
        output = self.fc(x)
        score = 0.
        for b in range(output.shape[0]):
            decision = output[b].argsort(dim=0)[-y[b].sum().int():]
            gt = torch.where(y[b]==1)[0]
            score += sum([t in gt for t in decision]) / y[b].sum()
        score /= y.shape[0]
        loss = self.criterion(output, y) if lbda == None else (lbda * self.criterion(output, y) + (1 - lbda) * self.criterion(output, y[perm]))
        return loss, score


### with Euclidean distance
class L2(nn.Module):
    def __init__(self, inputDim, numClasses):
        super(L2, self).__init__()
        self.centroids = torch.nn.Parameter(torch.zeros(numClasses, inputDim))
        self.centroidsRotations = torch.nn.Parameter(torch.zeros(4, inputDim))
        self.criterion = nn.CrossEntropyLoss() if args.label_smoothing == 0 else LabelSmoothingLoss(numClasses, args.label_smoothing)
        self.numClasses = numClasses

    def forward(self, x, y, yRotations = None, lbda = None, perm = None):
        distances = -1 * torch.pow(torch.norm(x.unsqueeze(1) - self.centroids.unsqueeze(0), dim = 2), 2)
        decisions = distances.argmax(dim = 1)
        score = (decisions - y == 0).float().mean()
        loss = self.criterion(distances, y)
        if lbda is not None:
            loss = lbda * loss + (1 - lbda) * self.criterion(distances, y[perm])
            score = lbda * score + (1 - lbda) * (decisions - y[perm] == 0).float().mean()
        if yRotations is not None:
            distancesRotations = -1 * torch.pow(torch.norm(x.unsqueeze(1) - self.centroidsRotations.unsqueeze(0), dim = 2),2)
            loss = 0.5 * loss + 0.5 * (self.criterion(distancesRotations, yRotations) if lbda == None else (lbda * self.criterion(distancesRotations, yRotations) + (1 - lbda) * self.criterion(distancesRotations, yRotations[perm])))
        return loss, score

class LabelSmoothingLoss(nn.Module):
    def __init__(self, num_classes, smoothing):
        super(LabelSmoothingLoss, self).__init__()
        self.smoothing = smoothing
        self.cls = num_classes

    def forward(self, pred, target):
        assert 0 <= self.smoothing < 1
        pred = pred.log_softmax(dim=-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), 1 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * pred, dim=-1))

class ProtoNet(nn.Module):
    def __init__(self) -> None:
        super(ProtoNet, self).__init__()
        pass        
    def forward(self, backbone, dataStep):
        loss, score = 0, 0
        features = [] # forward everything through backbone using a batch_size
        for i in range(dataStep.shape[0]//args.batch_size + 1):
            features.append(backbone(dataStep[i*args.batch_size:(i+1)*args.batch_size]))
        features = torch.cat(features, dim = 0)
        shots = torch.stack([features[(args.few_shot_shots+args.few_shot_queries)*c:(args.few_shot_shots+args.few_shot_queries)*c+args.few_shot_shots] for c in range(args.few_shot_ways)]) # split into shots
        queries = torch.stack([features[(args.few_shot_shots+args.few_shot_queries)*c+args.few_shot_shots:(args.few_shot_shots+args.few_shot_queries)*(c+1)] for c in range(args.few_shot_ways)]) # split into queries
        prototypes = shots.mean(dim = 1) # compute prototypes
        distances = -1 * torch.pow(torch.norm(queries.reshape(-1, queries.shape[-1]).unsqueeze(1) - prototypes.unsqueeze(0), dim=2), 2) # compute distances        
        distances = distances.reshape(args.few_shot_ways, args.few_shot_queries, args.few_shot_ways)

        log_p_y = F.log_softmax(distances, dim=0)
        target_inds = torch.arange(0, args.few_shot_ways).to(args.device)
        target_inds = target_inds.view(args.few_shot_ways, 1, 1)
        target_inds = target_inds.expand(args.few_shot_ways, args.few_shot_queries, 1).long()
        loss = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
        _, y_hat = log_p_y.max(2)
        score = y_hat.eq(target_inds.squeeze(2)).float().mean().cpu()
        return loss, score

### NCM
def ncm(shots, queries):
    centroids = torch.stack([shotClass.mean(dim = 0) for shotClass in shots])
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = torch.norm(queriesClass.unsqueeze(1) - centroids.unsqueeze(0), dim = 2)
        score += (distances.argmin(dim = 1) - i == 0).float().sum()
        total += queriesClass.shape[0]
    return score / total

###  softkmeans
def softkmeans(shots, queries, T = 5):
    score, total = 0, 0
    centroids = torch.stack([shotClass.mean(dim=0) for shotClass in shots])
    support = centroids.clone()
    support_size = sum([shotClass.shape[0] for shotClass in shots])
    queriesFlat = torch.cat(queries)
    queries_size = queriesFlat.shape[0]
    # Compute means 
    for i in range(30):
        similarities = torch.cdist(queriesFlat, centroids)
        soft_allocations = F.softmax(-similarities.pow(2)*T, dim=1)
        soft_allocations = soft_allocations/soft_allocations.sum(dim=0, keepdim=True)
        centroids = torch.einsum('qp,qd->pd', soft_allocations, queriesFlat)
        centroids = support*support_size+centroids*queries_size
        centroids /= (support_size + queries_size)
        
    for i, queriesClass in enumerate(queries):
        distances = torch.cdist(queriesClass, centroids)
        winners = distances.argmin(dim=1)
        score += (winners == i).float().sum()
        total += queriesClass.shape[0]
    return score/total

### kNN
def knn(shots, queries):
    k = int(args.few_shot_classifier[:-2])
    anchors = torch.cat(shots)
    labels = []
    for i in range(len(shots)):
        labels += [i] * shots[i].shape[0]
    score = 0
    total = 0
    for i, queriesClass in enumerate(queries):
        distances = torch.norm(queriesClass.unsqueeze(1) - anchors.unsqueeze(0), dim = 2)
        sorting = distances.argsort(dim = 1)
        scores = torch.zeros(queriesClass.shape[0], len(shots))
        for j in range(queriesClass.shape[0]):
            for l in range(k):
                scores[j,labels[sorting[j,l]]] += 1
        score += (scores.argmax(dim = 1) - i == 0).float().sum()
        total += queriesClass.shape[0]
    return score / total

def evalFewShotRun(shots, queries):
    if args.few_shot_classifier.lower()[-2:] == "nn":
        search = "nn"
    else:
        search = args.few_shot_classifier.lower()
    with torch.no_grad():
        return {
            "ncm": ncm,
            "nn" : knn,
            "softkmeans": softkmeans, 
            }[search](shots, queries)

def prepareCriterion(outputDim, numClasses):
    return {
        "lr": lambda: LR(outputDim, numClasses),
        "l2": lambda: L2(outputDim, numClasses), 
        'multilabelbce': lambda : MultiLabelBCE(outputDim, numClasses), 
        }[args.classifier.lower()]()

print(" classifiers,", end="")
