import torch
from args import args
from torchvision import transforms
from selfsupervised.dino import DINOAugmentation
from selfsupervised.simclr import SIMCLRAugmentation
from selfsupervised.simsiam import SIMSIAMAugmentation
from selfsupervised.barlowtwins import BARLOWTWINSAugmentation
DEFAULT_NORMALIZATION = transforms.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]),std=torch.tensor([0.229, 0.224, 0.225]))

def get_ssl_transform(image_size, normalization=DEFAULT_NORMALIZATION):
    all_steps = [item for sublist in eval(args.steps) for item in sublist]
    all_transforms = {}
    for step in all_steps:
        if 'dino' in step:
            all_transforms['dino'] = DINOAugmentation(image_size, normalization=normalization)
        if 'simclr' in step:
            all_transforms['simclr'] = SIMCLRAugmentation(image_size, normalization=normalization)
        if 'simclr_supervised' in step:
            all_transforms['simclr_supervised'] = SIMCLRAugmentation(image_size, normalization=normalization)
        if 'simsiam' in step:
            all_transforms['simsiam'] = SIMSIAMAugmentation(image_size, normalization=normalization)
        if 'barlowtwins' in step:
            all_transforms['barlowtwins'] = BARLOWTWINSAugmentation(image_size, normalization=normalization)
    return all_transforms