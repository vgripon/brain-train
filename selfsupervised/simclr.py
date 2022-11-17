import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from args import args
from torchvision import transforms

DEFAULT_NCROPS = 2
DEFAULT_IN_DIM = 512 #2048 if resnet50 or 512 if resnet18
DEFAULT_OUT_DIM = 128
DEFAULT_TEMPERATURE = 0.07
DEFAULT_SUPERVISED = False

class SIMCLRAugmentation(object):
    def __init__(self,
                 image_size, normalization, s=1.0):
        #s is the color distorsion strength
        random_resized_crop = transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0), ratio=(3/4, 4/3), interpolation=Image.BICUBIC)
        rnd_horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
        color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
        rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.2)
        color_distort = transforms.Compose([rnd_color_jitter, rnd_gray])
        gaussian_blur = transforms.GaussianBlur(kernel_size=(int(image_size*0.1)//2)*2+1, sigma=(0.1, 2.0))
        rnd_gaussian_blur = transforms.RandomApply([gaussian_blur], p=0.5)

        rrc_flip_color_distort_blur = transforms.Compose([random_resized_crop, 
                                                        rnd_horizontal_flip, 
                                                        color_distort, 
                                                        rnd_gaussian_blur,])

        # first view
        self.global_transform1 = transforms.Compose([rrc_flip_color_distort_blur, transforms.ToTensor(), normalization])
        # second view
        self.global_transform2 = transforms.Compose([rrc_flip_color_distort_blur, transforms.ToTensor(), normalization])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transform1(image))
        crops.append(self.global_transform2(image))
        return crops

class SIMCLRHead(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(SIMCLRHead, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(in_dim, in_dim), 
                                nn.BatchNorm1d(in_dim), nn.ReLU(), 
                                nn.Linear(in_dim, out_dim),)
    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        return x

class SIMCLR(nn.Module):
    def __init__(self, in_dim=DEFAULT_IN_DIM, out_dim=DEFAULT_OUT_DIM, temperature=DEFAULT_TEMPERATURE, ncrops=DEFAULT_NCROPS, supervised=DEFAULT_SUPERVISED):
        super(SIMCLR, self).__init__()
        self.ncrops = ncrops
        self.temperature = temperature
        self.head = SIMCLRHead(in_dim, out_dim)
        self.simclr_loss_fn = SupConLoss(temperature=DEFAULT_TEMPERATURE, device=args.device)
        self.supervised = supervised

    def forward_pass(self, backbone, head, x):
        return head(backbone(x))

    def forward(self, backbone, dataStep, target):
        proj1 = self.forward_pass(backbone, self.head, dataStep[0])
        proj2 = self.forward_pass(backbone, self.head, dataStep[1])
        if self.supervised:
            simclr_loss = self.simclr_loss_fn(proj1, proj2, target) #supervised contrastive if label
        else:
            simclr_loss = self.simclr_loss_fn(proj1, proj2) #simclr if unlabel
        return simclr_loss

class SupConLoss(nn.Module): # inspired by : https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    def __init__(self, temperature=0.07, base_temperature=0.07, device="cpu"):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.device = device

    def forward(self, proj1, proj2, labels=None):
        features = torch.cat([proj1.unsqueeze(1), proj2.unsqueeze(1)], dim=1)
        batch_size = features.shape[0]

        if labels is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(self.device)
        else:
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(self.device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_dot_contrast = torch.div(torch.matmul(contrast_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach() # for numerical stability

        mask = mask.repeat(contrast_count, contrast_count)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size * contrast_count).view(-1, 1).to(self.device), 0)
        # or simply : logits_mask = torch.ones_like(mask) - torch.eye(50)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(contrast_count, batch_size).mean()

        return loss