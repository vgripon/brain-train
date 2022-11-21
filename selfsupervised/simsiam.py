import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter
import random
from args import args

DEFAULT_NCROPS = 2
DEFAULT_IN_DIM = 512 #2048 if resnet50 or 512 if resnet18

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

class SIMSIAMAugmentation(object):
    def __init__(self,
                 image_size, normalization):
        random_resized_crop = transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0), ratio=(3/4, 4/3), interpolation=Image.BILINEAR)  
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        rnd_gaussian_blur = transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5)
        rnd_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        

        rrc_color_distort_blur_flip = transforms.Compose([random_resized_crop, 
                                                        color_distort, 
                                                        rnd_gaussian_blur, 
                                                        rnd_horizontal_flip,])

        # first view
        self.global_transform1 = transforms.Compose([rrc_color_distort_blur_flip, transforms.ToTensor(), normalization])

        # second view
        self.global_transform2 = transforms.Compose([rrc_color_distort_blur_flip, transforms.ToTensor(), normalization])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        return crops

class SIMSIAM(nn.Module):
    def __init__(self, in_dim=DEFAULT_IN_DIM, pred_dim=DEFAULT_IN_DIM//4):
        super(SIMSIAM, self).__init__()
        # PROJECTOR (3-layer MLP)
        self.projector = nn.Sequential(nn.Linear(in_dim, in_dim, bias=False),
                                        nn.BatchNorm1d(in_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(in_dim, in_dim, bias=False),
                                        nn.BatchNorm1d(in_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        nn.Linear(in_dim, in_dim, bias=False),               
                                        nn.BatchNorm1d(in_dim, affine=False)) # output layer

        # PREDICTOR (2-layer MLP)
        self.predictor = nn.Sequential(nn.Linear(in_dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, in_dim)) # output layer

        self.loss = SIMSIAMLoss()

    def forward(self, backbone, dataStep):
        z1, z2 = backbone(dataStep[0]), backbone(dataStep[1])
        h1, h2 = self.projector(z1), self.projector(z2)
        p1, p2 = self.predictor(h1), self.predictor(h2)
        simsiam_loss = self.loss(p1, p2, h1, h2)
        return simsiam_loss

class SIMSIAMLoss(nn.Module):
    def __init__(self):
        super(SIMSIAMLoss, self).__init__()
        self.criterion = nn.CosineSimilarity(dim=1)#.to(args.device)

    def forward(self, p1, p2, h1, h2):
        return -(self.criterion(p1, h2.detach()).mean() + self.criterion(p2, h1.detach()).mean()) * 0.5