import torch
import torch.nn as nn
import random
from args import args
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter, ImageOps

DEFAULT_NORMALIZATION = transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )   

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img

class DINOAugmentation(object):
    def __init__(self, local_crops_number,
                 image_size, supervised_transform, normalization=DEFAULT_NORMALIZATION, global_crops_scale=(0.5,1), local_crops_scale=(0.05, 0.4)):
        if normalization == None:
            normalization = DEFAULT_NORMALIZATION
        self.supervised_transform = supervised_transform
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ])
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)],p = 0.3),
            transforms.RandomGrayscale(p=0.2),
        ])
        # first global crop
        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(1.0),
            transforms.ToTensor(),
            normalization,
        ])
        # second global crop
        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(0.1),
            Solarization(0.2),
            transforms.ToTensor(),
            normalization,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(int(image_size*96/224), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            GaussianBlur(p=0.5),
            transforms.ToTensor(),
            normalization,
        ])

    def __call__(self, image):
        crops = []
        if self.supervised_transform is not None:
            crops.append(self.supervised_transform(image))
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(image))
        return crops

def get_ssl_transform(image_size, supervised_transform, normalization):
    all_steps = [item for sublist in eval(args.steps) for item in sublist]
    for step in all_steps:
        if 'dino' in step:
            local_crops_number = 8
            return DINOAugmentation(local_crops_number, image_size, supervised_transform, normalization=normalization, global_crops_scale=(0.5,1), local_crops_scale=(0.05, 0.4))
    return supervised_transform

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


