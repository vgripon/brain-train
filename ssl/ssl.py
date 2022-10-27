import torch
import torch.nn as nn
import random
from args import args
from torchvision import transforms
from PIL import Image
from PIL import ImageFilter, ImageOps
from ssl.dino import DINOAugmentation
DEFAULT_NORMALIZATION = transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))
class SSLTransform(object):
    """
    Wrapper for different transforms.
    """
    def __init__(self, all_transforms):
        self.all_transforms = all_transforms
    def __call__(self, image):
        out = {}
        for name, T in self.all_transforms.items():
            out[name] = T(image)
        return out
def get_ssl_transform(image_size, supervised_transform, normalization=DEFAULT_NORMALIZATION):
    all_steps = [item for sublist in eval(args.steps) for item in sublist]
    all_transforms = {'supervised':supervised_transform}
    for step in all_steps:
        if 'dino' in step:
            local_crops_number = 8
            all_transforms['dino'] = DINOAugmentation(local_crops_number, image_size, normalization=normalization, global_crops_scale=(0.5,1), local_crops_scale=(0.05, 0.4))
    return SSLTransform(all_transforms)
