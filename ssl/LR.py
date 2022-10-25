import torch
import torch.nn as nn 
from args import args
import random
import numpy as np

### Logistic Regression module, which is the classic way to train deep models for classification
class LR_Rotations_Mixup(nn.Module):
    def __init__(self, inputDim, numClasses, backbone=None):
        super(LR_Rotations_Mixup, self).__init__()
        self.fc = nn.Linear(inputDim, numClasses)
        self.fcRotations = nn.Linear(inputDim, 4)
        self.criterion = nn.CrossEntropyLoss() if args.label_smoothing == 0 else LabelSmoothingLoss(numClasses, args.label_smoothing)
        self.backbone = backbone
    def forward(self, backbone, dataStep, y, rotation=False, mixup=False, manifold_mixup=False):
        lbda, perm, mixupType = None, None, None
        if mixup or manifold_mixup:
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

        x = backbone(dataStep, mixup = mixupType, lbda = lbda, perm = perm)
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