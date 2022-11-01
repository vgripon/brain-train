import torch
import torch.nn as nn
import torch.nn.functional as F
from args import args
from torchvision import transforms
from losses import SupConLoss

DEFAULT_NCROPS = 2
DEFAULT_HEAD_HIDDEN_DIM = 2048 #if resnet50
DEFAULT_OUT_DIM = 128
DEFAULT_TEMPERATURE = 0.07

class SIMCLRAugmentation(object):
    def __init__(self,
                 image_size, normalization, s=1.0):
        #s is the color distorsion strength
        random_resized_crop = transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC)
        rnd_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        gaussian_blur = transforms.GaussianBlur(kernel_size=image_size*0.1, sigma=(0.1, 2.0))
        rnd_gaussian_blur = transforms.RandomApply([gaussian_blur], p=0.5)

        rrc_flip_color_distort_blur = transforms.Compose([random_resized_crop, 
                                                        rnd_horizontal_flip, 
                                                        color_distort, 
                                                        rnd_gaussian_blur,])

        # first view
        self.global_transform1 = transforms.Compose([rrc_flip_color_distort_blur, normalize])
        # second view
        self.global_transform2 = transforms.Compose([rrc_flip_color_distort_blur, normalize])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        return crops

class SIMCLRHead(nn.Module):
    def __init__(self, in_dim, out_dim, norm_last_layer, head_hidden_dim):
        super(SIMCLRHead, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, head_hidden_dim), 
                                nn.BatchNorm1d(head_hidden_dim), nn.ReLU(), 
                                nn.Linear(head_hidden_dim, out_dim),)
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x

class SIMCLR(nn.Module):
    def __init__(self, in_dim, epochs, nSteps, head_hidden_dim=DEFAULT_HEAD_HIDDEN_DIM, out_dim=DEFAULT_OUT_DIM, temperature=DEFAULT_TEMPERATURE, ncrops=DEFAULT_NCROPS):
        super(SIMCLR, self).__init__()
        self.ncrops = ncrops
        self.nSteps = nSteps
        self.temperature = temperature
        self.head = SIMCLRHead(in_dim, out_dim, head_hidden_dim)
        self.simclr_loss_fn = SupConLoss()

    def forward_pass(self, backbone, head, x):
        return backbone(head(x))

    def forward(self, backbone, dataStep, target):
        proj1 = self.forward_pass(backbone, self.head, dataStep[0])
        proj2 = self.forward_pass(backbone, self.head, dataStep[1])
        if target is not None:
            target = torch.cat([target, target], dim=0)
            simclr_loss = self.simclr_loss_fn(proj1, proj2, target) #supervised contrastive if label
        else:
            simclr_loss = self.simclr_loss_fn(proj1, proj2) #simclr if unlabel
        return simclr_loss