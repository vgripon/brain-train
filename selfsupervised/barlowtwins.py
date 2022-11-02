import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps, ImageFilter
import random
from args import args
from torchvision import transforms

DEFAULT_NCROPS = 2
DEFAULT_HEAD_HIDDEN_DIM = 2048
DEFAULT_OUT_DIM = 128
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

class Transform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=1.0),
            Solarization(p=0.0),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2

class BARLOWTWINSAugmentation(object):
    def __init__(self,
                 image_size, normalization, s=1.0):
        #s is the color distorsion strength
        random_resized_crop = transforms.RandomResizedCrop(image_size, scale=image_size, interpolation=Image.BICUBIC)
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
                                                    normalization,])
        # second view
        self.global_transform1 = transforms.Compose([random_resized_crop, 
                                                    rnd_horizontal_flip, 
                                                    color_distort, 
                                                    rnd_gaussian_blur2, 
                                                    rnd_solarization2,
                                                    normalization,])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        return crops

class BARLOWTWINSHead(nn.Module):
    def __init__(self, in_dim, out_dim, head_hidden_dim):
        super(BARLOWTWINSHead, self).__init__()
        layers = []
        layers.extend([nn.Linear(in_dim, head_hidden_dim), nn.BatchNorm1d(head_hidden_dim), nn.ReLU()])
        layers.extend([nn.Linear(head_hidden_dim, head_hidden_dim), nn.BatchNorm1d(head_hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(head_hidden_dim, out_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        x = self.mlp(x)
        return x # standardization done in the loss function

class BARLOWTWINS(nn.Module):
    def __init__(self, in_dim, epochs, nSteps, head_hidden_dim=DEFAULT_HEAD_HIDDEN_DIM, out_dim=DEFAULT_OUT_DIM, ncrops=DEFAULT_NCROPS):
        super(BARLOWTWINS, self).__init__()
        self.ncrops = ncrops
        self.nSteps = nSteps
        self.head = BARLOWTWINSHead(in_dim, out_dim, head_hidden_dim)
        self.barlowtwins_loss_fn = BarlowTwinsLoss(projector_out=DEFAULT_OUT_DIM, lambd=DEFAULT_LOSS_TRADEOFF)

    def forward_pass(self, backbone, head, x):
        return backbone(head(x))

    def forward(self, backbone, dataStep, target):
        proj1 = self.forward_pass(backbone, self.head, dataStep[0])
        proj2 = self.forward_pass(backbone, self.head, dataStep[1])
        barlowtwins_loss = self.barlowtwins_loss_fn(proj1, proj2)
        return barlowtwins_loss

def off_diagonal(x):
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

class BarlowTwinsLoss(nn.Module):
    def __init__(self, projector_out=128, lambd=0.0051):
        super().__init__()

        self.bn = nn.BatchNorm1d(projector_out, affine=False)
        self.lambd = lambd

    def forward(self, z1, z2):

        c = self.bn(z1).T @ self.bn(z2) #normalization outside the loss function
        c = z1.T @ z2
        c.div_(z1.shape[0])

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        return loss