import torch
import torch.nn as nn
import random

class RandomApply(nn.Module):
    def __init__(self, fn, p):
        super().__init__()
        self.fn = fn
        self.p = p

    def forward(self, x):
        if random.random() > self.p:
            return x
        return self.fn(x)

class DINO(nn.Module):
    def __init__(self, backbone, in_dim, out_dim, temperature_student, temperature_teacher, norm_last_layer=True, moving_average_decay=0.999, head_hidden_dim=2048, bottleneck_dim=256):
        super(DINO, self).__init__()
        self.teacher = backbone.clone().detach()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.moving_average_decay = moving_average_decay
        self.centering = nn.Parameter(torch.zeros(in_dim))
        self.temperature_student = temperature_student
        self.temperature_teacher = temperature_teacher

        self.projector = self.build_projector(in_dim, out_dim, norm_last_layer, head_hidden_dim, bottleneck_dim)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
                self.last_layer.weight_g.requires_grad = False
        self.augment = torch.nn.Sequential(
            RandomApply(
                T.ColorJitter(0.8, 0.8, 0.8, 0.2),
                p = 0.3
            ),
            T.RandomGrayscale(p=0.2),
            T.RandomHorizontalFlip(),
            RandomApply(
                T.GaussianBlur((3, 3), (1.0, 2.0)),
                p = 0.2
            ),
            T.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]),
                std=torch.tensor([0.229, 0.224, 0.225])),
        )
    def build_projector(self, in_dim, out_dim, norm_last_layer, head_hidden_dim, bottleneck_dim):
        projector = [nn.Linear(in_dim, head_hidden_dim), nn.BatchNorm1d(head_hidden_dim)]
        for i in range(2):
            projector.append(nn.Linear(head_hidden_dim, head_hidden_dim))
            projector.append(nn.BatchNorm1d(head_hidden_dim))
            projector.append(nn.GELU())
        projector.append(nn.Linear(head_hidden_dim, bottleneck_dim))
        projector = nn.Sequential(*projector)
        return projector
        
    def forward(self, student, dataStep, target):
        pass


