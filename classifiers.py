### this file describes the criterions used to train architectures
### they have to be thought of as loss functions in a broad way, including a fully connected layer for some of them
### they are typically composed on top of a backbone, that is seen as a feature extractor
### it thus takes the form of a nn module, even if some do not contain any parameters

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from args import args

### Logistic Regression module, which is the classic way to train deep models for classification
class LR(nn.Module):
    def __init__(self, inputDim, numClasses):
        super(LR, self).__init__()
        self.fc = nn.Linear(inputDim, numClasses)
        self.fcRotations = nn.Linear(inputDim, 4)
        self.criterion = nn.CrossEntropyLoss() if args.label_smoothing == 0 else LabelSmoothingLoss(numClasses, args.label_smoothing)

    def forward(self, x, y, yRotations = None, lbda = None, perm = None):
        output = self.fc(x)
        decision = output.argmax(dim = 1)
        score = (decision - y == 0).float().mean()
        loss = self.criterion(output, y)
        if lbda is not None:
            loss = lbda * loss + (1 - lbda) * self.criterion(output, y[perm])
            score = lbda * score + (1 - lbda) * (decision - y[perm] == 0).float().mean()
        if yRotations is not None:
            outputRotations = self.fcRotations(x)
            loss = 0.5 * loss + 0.5 * (self.criterion(outputRotations, yRotations) if lbda == None else (lbda * self.criterion(outputRotations, yRotations) + (1 - lbda) * self.criterion(outputRotations, yRotations[perm])))
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
        self.inputDim = inputDim

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

class Multihead(L2):
    def __init__(self, inputDim, numClasses):
        self.head_size = inputDim // numClasses
        self.numClasses = numClasses
        super(Multihead, self).__init__(self.head_size,numClasses-1)
        self.L2= nn.Sequential(*[L2(self.head_size,numClasses-1).to(args.device) for i in range(numClasses)])
        if args.culsters!='':
            list_l2 = []
            with open(args.clusters, 'r') as f:
                self.clusters = [line.rstrip('\n') for line in f]
            self.head_size = inputDim // len(self.clusters)
            for i,x in enumerate(self.clusters):
                self.clusters[i]=eval(x)
                list_l2.append(L2(self.head_size,len(eval)).to(args.device) for i in range(len(self.clusters)))
            self.L2= nn.Sequential(*list_l2)
            self.LUT=[]
            for i,c in enumerate(self.clusters):
                    self.LUT.append(np.array(c).argsort())
    
    def forward(self, x, y, yRotations = None, lbda = None, perm = None):
        loss, score = 0,0
        if args.clusters=='':
            for i in range(self.numClasses):
                x_head = x[y!=i,i*self.head_size:(i+1)*self.head_size]   # #ignore targets corresponding to the head i & select a chunk of features
                y_head = y[y!=i]                                         #ignore targets corresponding to the head i 
                y_head[y_head>i] -= 1                                      #realign targets corresponding so as to have the same number of targets.
                loss_i, score_i = self.L2[i].forward(x_head,y_head, yRotations=yRotations,lbda =lbda,perm=perm)
                loss += loss_i
                score += score_i
            return loss/self.numClasses , score/self.numClasses 
        else:
            for i,c in enumerate(self.clusters):
                y_head = []
                x_head = torch.tensor()
                for i in range(len(y)):
                    if y[i] in c:
                        y_head.append(self.LUT[i][c.index(y[i])])
                        x_head.append(x[i,i*self.head_size:(i+1)*self.head_size])
                x_head = torch.stack(x_head)
                loss_i, score_i = self.L2[i].forward(x_head,y_head, yRotations=yRotations,lbda =lbda,perm=perm)
                loss += loss_i
                score += score_i
            return loss/len() , score/self.numClasses 
    

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
def softkmeans(shots, queries):
    T = 5
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
        'multihead' : lambda : Multihead(outputDim,numClasses)
        }[args.classifier.lower()]()

print(" classifiers,", end="")
