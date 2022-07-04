### this file describes the criterions used to train architectures
### they have to be thought of as loss functions in a broad way, including a fully connected layer for some of them
### they are typically composed on top of a backbone, that is seen as a feature extractor
### it thus takes the form of a nn module, even if some do not contain any parameters

import torch
import torch.nn as nn
from args import args

### Logistic Regression module, which is the classic way to train deep models for classification
class LR(nn.Module):
    def __init__(self, inputDim, numClasses):
        super(LR, self).__init__()
        self.fc = nn.Linear(inputDim, numClasses)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, y):
        output = self.fc(x)
        decision = output.argmax(dim = 1)
        score = (decision - y == 0).float().mean()
        return self.criterion(output, y), score

### with Euclidean distance
class L2(nn.Module):
    def __init__(self, inputDim, numClasses):
        super(L2, self).__init__()
        self.centroids = torch.nn.Parameter(torch.zeros(numClasses, inputDim))
        self.criterion = nn.CrossEntropyLoss()
        self.numClasses = numClasses

    def forward(self, x, y):
        distances = -1 * torch.norm(x.unsqueeze(1) - self.centroids.unsqueeze(0), dim = 2)
        decisions = distances.argmax(dim = 1)
        score = (decisions - y == 0).float().mean()
        return self.criterion(distances, y), score

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
    return {
        "ncm": ncm,
        "nn" : knn
        }[search](shots, queries)

def prepareCriterion(outputDim, numClasses):
    return {
        "lr": lambda: LR(outputDim, numClasses),
        "l2": lambda: L2(outputDim, numClasses)
        }[args.classifier.lower()]()

print(" classifiers,", end="")
