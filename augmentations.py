# Define all the transformations used for data augmentation
import torch
from torchvision import transforms
import numpy as np
# Define GaussianNoise since it's not part of torch.transforms
class GaussianNoise(object):
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, img):
        img += self.noise_std * torch.randn(img.size())
        return img

class norm(object):
    def __init__(self, max_val=255, change_sign = 1):
        self.max_val = max_val
        self.change_sign = change_sign
    def __call__(self, img):
        img /= self.max_val
        img-= 0.5
        img*=2
        return self.change_sign * img
    
class bi_resize(object):
    def __init__(self, align_corners=True,target_size =126):
        self.align_corners = align_corners
        self.target_size = target_size

    def __call__(self, img):
        img = torch.nn.functional.interpolate(img.unsqueeze(0),size=(self.target_size,self.target_size),mode='bilinear',align_corners=self.align_corners).squeeze(0)
        return img

class totensor(object):
    def __init__(self, normalize=True):
        self.normalize = normalize
    def __call__(self, img):
        img = torch.tensor(np.array(img).astype(np.float32)).permute(2,0,1)
        return img
imagenetnorm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
miniimagenetnorm = transforms.Normalize([125.3/255, 123.0/255, 113.9/255], [63.0/255, 62.1/255, 66.7/255])
cifar10norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
cifar100norm = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))
mnistnorm = transforms.Normalize((0.1302,), (0.3069,))
def parse_transforms(transforms_str, image_size):
    """
    Define the transformations to be applied to the images
    """
    transforms_list = []
    for transform in transforms_str:
        if 'gaussiannoise' in transform.lower():
            if '_' in transform:
                noise_std = float(transform.split('_')[-1])
            else:
                noise_std = 0.1
            transforms_list.append(GaussianNoise(noise_std))
        elif 'metadatasettotensor' in transform.lower():
            transforms_list.append(totensor())
        elif 'totensor' in transform.lower() and 'metadataset' not in transform.lower():
            transforms_list.append(transforms.ToTensor())
        elif 'metadatasetnorm' in transform.lower():
            if 'change_sign' in transform.lower():
                change_sign = -1
            else:
                change_sign = 1
            transforms_list.append(norm(change_sign=change_sign))
        elif 'imagenetnorm' in transform.lower() and 'miniimagenetnorm' not in transform.lower():
            transforms_list.append(imagenetnorm)
        elif 'miniimagenetnorm' in transform.lower():
            transforms_list.append(miniimagenetnorm)
        elif 'cifar10norm' in transform.lower():
            transforms_list.append(cifar10norm)
        elif 'cifar100norm' in transform.lower():
            transforms_list.append(cifar100norm)
        elif 'mnistnorm' in transform.lower():
            transforms_list.append(mnistnorm)
        elif 'norm_' in transform.lower():
            values = transform.lower().split('_')[1:]
            mean = [float(value) for value in values[:3]]
            std = [float(value) for value in values[3:]]
            transforms_list.append(transforms.Normalize(mean,std))
        elif 'resize' in transform.lower() and 'random' not in transform.lower() and 'biresize' not in transform.lower():
            if '_' in transform:
                ratio = transform.split('_')[-1]
                if '/' in transform:
                    ratio = eval(ratio)
                else:
                    ratio = float(ratio)
            else:
                ratio = 1
            transforms_list.append(transforms.Resize(int(image_size*ratio)))
        elif 'biresize' in transform.lower():
            transforms_list.append(bi_resize(target_size=image_size))
        elif 'randomresizedcrop' in transform.lower():
            transforms_list.append(transforms.RandomResizedCrop(image_size))
        elif 'centercrop' in transform.lower():
            transforms_list.append(transforms.CenterCrop(image_size))
        elif 'randomhorizontalflip' in transform.lower():
            if '_' in transform:
                p = float(transform.split('_')[-1])
            else:
                p = 0.5
            transforms_list.append(transforms.RandomHorizontalFlip(p=p))
        elif 'randomverticalflip' in transform.lower():
            if '_' in transform:
                p = float(transform.split('_')[-1])
            else:
                p = 0.5
            transforms_list.append(transforms.RandomVerticalFlip(p=p))
        elif 'colorjitter' in transform.lower():
            if '_' in transform:
                brightness = transform.split('_')[1:]
                if len(brightness) == 1:
                    brightness = float(brightness[0])
                    contrast, saturation = brightness, brightness
                else:
                    brightness, contrast, saturation = float(brightness[0]), float(brightness[1]), float(brightness[2])
            else:
                brightness, contrast, saturation = 0.4, 0.4, 0.4
            transforms_list.append(transforms.ColorJitter(brightness=brightness, contrast=contrast, saturation=saturation))     
        else:
            raise ValueError('Unknown transformation: {}'.format(transform))
    return transforms_list