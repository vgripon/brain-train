import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
import random
from args import args
from torchvision import transforms

DEFAULT_NCROPS = 2
DEFAULT_LOSS_TRADEOFF = 0.0051

class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class BARLOWTWINSAugmentation(object):
    def __init__(self,
                 image_size, normalization, s=1.0):
        #s is the color distorsion strength
        random_resized_crop = transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=Image.BICUBIC)
        rnd_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        rnd_gaussian_blur1 = GaussianBlur(p=1.0)
        rnd_gaussian_blur2 = GaussianBlur(p=0.1)
        rnd_solarization2 = Solarization(p=0.2)

        # first view
        self.global_transform1 = transforms.Compose([random_resized_crop, 
                                                    rnd_horizontal_flip, 
                                                    color_distort, 
                                                    rnd_gaussian_blur1,
                                                    transforms.ToTensor(), 
                                                    normalization,])
        # second view
        self.global_transform2 = transforms.Compose([random_resized_crop, 
                                                    rnd_horizontal_flip, 
                                                    color_distort, 
                                                    rnd_gaussian_blur2, 
                                                    rnd_solarization2,
                                                    transforms.ToTensor(),
                                                    normalization,])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        return crops

class BARLOWTWINSHead(nn.Module):
    def __init__(self, in_dim):
        super(BARLOWTWINSHead, self).__init__()
        layers = []
        layers.extend([nn.Linear(in_dim, 4*in_dim, bias=False), nn.BatchNorm1d(4*in_dim), nn.ReLU()])
        layers.extend([nn.Linear(4*in_dim, 4*in_dim, bias=False), nn.BatchNorm1d(4*in_dim), nn.ReLU()])
        layers.append(nn.Linear(4*in_dim, 4*in_dim, bias=False))
        # for debugging purposes, reducing mlp size
        #layers.extend([nn.Linear(in_dim, in_dim, bias=False), nn.BatchNorm1d(in_dim), nn.ReLU()])
        #layers.append(nn.Linear(in_dim, 128, bias=False))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x # standardization done in the loss function

class BARLOWTWINS(nn.Module):
    def __init__(self, in_dim, ncrops=DEFAULT_NCROPS):
        super(BARLOWTWINS, self).__init__()
        self.ncrops = ncrops
        self.head = BARLOWTWINSHead(in_dim)
        self.barlowtwins_loss_fn = BarlowTwinsLoss(dim=in_dim, lambd=DEFAULT_LOSS_TRADEOFF)

    def forward_pass(self, backbone, head, x):
        return head(backbone(x))

    def forward(self, backbone, dataStep):
        proj1 = self.forward_pass(backbone, self.head, dataStep[0])
        proj2 = self.forward_pass(backbone, self.head, dataStep[1])
        barlowtwins_loss = self.barlowtwins_loss_fn(proj1, proj2.detach())
        #print(barlowtwins_loss)
        return barlowtwins_loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, dim, lambd=0.0051):
        super().__init__()

        self.bn = nn.BatchNorm1d(128, affine=False)
        self.lambd = lambd

    def forward(self, z1, z2):

        c = self.bn(z1).T @ self.bn(z2) #normalization inside the loss function
        c = z1.T @ z2
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss
        # for debugging purpose using cosine sim
        #return nn.CosineSimilarity(dim=1)(z1, z2).mean()