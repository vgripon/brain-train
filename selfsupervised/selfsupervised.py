import torch
from args import args
from torchvision import transforms
from selfsupervised.dino import DINOAugmentation
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
    if 'lr' in all_steps or 'rotations' in all_steps or 'mixup' in all_steps or 'manifold mixup' in all_steps:
        all_transforms = {'supervised':supervised_transform}
    else:
        all_transforms = {}
    for step in all_steps:
        if 'dino' in step:
            all_transforms['dino'] = DINOAugmentation(image_size, normalization=normalization)
    return SSLTransform(all_transforms)
