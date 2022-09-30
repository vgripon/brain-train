### file to generate dataloaders
### for simplicity, we do not offer the choice to load the whole dataset in VRAM anymore

from torchvision import transforms, datasets
import random
from args import args
import torch
import os
import json
import numpy as np
from PIL import Image
from utils import *
### first define dataholder, which will be used as an argument to dataloaders
class DataHolder():
    def __init__(self, data, targets, transforms, target_transforms=lambda x:x, opener=lambda x: Image.open(x).convert('RGB')):
        self.data = data
        if torch.is_tensor(data):
            self.length = data.shape[0]
        else:
            self.length = len(self.data)
        self.targets = targets
        assert(self.length == len(targets))
        self.transforms = transforms
        self.target_transforms = target_transforms
        self.opener = opener
    def __getitem__(self, idx):
        if isinstance(self.data[idx], str):
            elt = self.opener(args.dataset_path + self.data[idx])
        else:
            elt = self.data[idx]
        return self.transforms(elt), self.target_transforms(self.targets[idx])
    def __len__(self):
        return self.length

def dataLoader(dataholder, shuffle):
    return torch.utils.data.DataLoader(dataholder, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(os.cpu_count(), 8))

# Define GaussianNoise since it's not part of torch.transforms
class GaussianNoise(object):
    def __init__(self, noise_std):
        self.noise_std = noise_std

    def __call__(self, img):
        img += self.noise_std * torch.randn(img.size())
        return img

class norm(object):
    def __init__(self, max_val=255):
        self.max_val = max_val

    def __call__(self, img):
        img /= self.max_val
        img-= 0.5
        img*=2
        return img
    
class bi_resize(object):
    def __init__(self, align_corners=True,target_size =126):
        self.align_corners = align_corners
        self.target_size = target_size

    def __call__(self, img):
        print(img.shape)
        img = torch.nn.functional.interpolate(img.unsqueeze(0),size=(self.target_size,self.target_size),mode='bilinear',align_corners=self.align_corners).squeeze(0)
        return img

class totensor(object):
    def __init__(self, normalize=True):
        self.normalize = normalize
    def __call__(self, img):

        img = torch.tensor(np.array(img).astype(np.float32)).permute(2,0,1)
        return img

def cifar10(dataset):
    pytorchDataset = datasets.CIFAR10(args.dataset_path, train = dataset != "test", download = 'cifar-10-python.tar.gz' not in os.listdir(args.dataset_path))
    data = torch.tensor(pytorchDataset.data).transpose(1,3).transpose(2,3).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))

    if dataset == 'train':
        trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), normalization]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([normalization])
        else:
            trans = torch.nn.Sequential(transforms.RandomResizedCrop(32), normalization)

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset == "train"), "name":"cifar10_" + dataset, "num_classes":10, "name_classes": pytorchDataset.classes}

def cifar100(dataset):
    pytorchDataset = datasets.CIFAR100(args.dataset_path, train = dataset != "test", download = 'cifar-100-python.tar.gz' not in os.listdir(args.dataset_path))
    data = torch.tensor(pytorchDataset.data).transpose(1,3).transpose(2,3).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762))

    if dataset == 'train':
        trans = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), normalization]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([normalization])
        else:
            trans = torch.nn.Sequential(transforms.RandomResizedCrop(32), normalization)

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = dataset == "train"), "name":"cifar100_" + dataset, "num_classes":100, "name_classes": pytorchDataset.classes}

def miniimagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["miniimagenet_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]

    normalization = transforms.Normalize([125.3/255, 123.0/255, 113.9/255], [63.0/255, 62.1/255, 66.7/255])

    if datasetName == 'train':
        trans = transforms.Compose([transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalization])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([transforms.Resize(92), transforms.CenterCrop(84), transforms.ToTensor(), normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(84), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def tieredimagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["tieredimagenet_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]

    normalization = transforms.Normalize([125.3/255, 123.0/255, 113.9/255], [63.0/255, 62.1/255, 66.7/255])

    if datasetName == 'train':
        trans = transforms.Compose([transforms.RandomResizedCrop(84), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalization])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([transforms.Resize(92), transforms.CenterCrop(84), transforms.ToTensor(), normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(84), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def cifarfs(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["cifarfs_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]

    normalization = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))

    if datasetName == 'train':
        trans = transforms.Compose([transforms.RandomResizedCrop(32), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalization])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([transforms.Resize([int(1.15*32), int(1.15*32)]), transforms.CenterCrop(32), transforms.ToTensor(), normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(32), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_imagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_imagenet_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def imagenet(datasetName):
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if datasetName == 'train':
        trans = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalization])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(256), transforms.ToTensor(), normalization])

    pytorchDataset = datasets.ImageNet(args.dataset_path + "/imagenet", split = "train" if datasetName != "test" else "val", transform = trans)
        
    return {"dataloader": dataLoader(pytorchDataset, shuffle = datasetName == "train"), "name":"imagenet_" + datasetName, "num_classes":1000, "name_classes": pytorchDataset.classes}

def mnist(datasetName):
    pytorchDataset = datasets.MNIST(args.dataset_path, train = datasetName != "test", download = 'MNIST' not in os.listdir(args.dataset_path))
    data = pytorchDataset.data.clone().unsqueeze(1).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.1302,), (0.3069,))

    trans = transforms.Compose([normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name": "mnist_" + datasetName, "num_classes": 10, "name_classes": list(range(10))}

def fashionMnist(datasetName):
    pytorchDataset = datasets.FashionMNIST(args.dataset_path, train = datasetName != "test", download = 'FashionMNIST' not in os.listdir(args.dataset_path))
    data = pytorchDataset.data.clone().unsqueeze(1).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.2849,), (0.3516,))

    if datasetName == 'train':
        trans = transforms.Compose([transforms.RandomCrop(28, padding=4), transforms.RandomHorizontalFlip(), normalization])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(28), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name": "fashion-mnist_" + datasetName, "num_classes": 10, "name_classes": pytorchDataset.classes}


def metadataset_dtd(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_dtd_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_cub(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_cub_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_fungi(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_fungi_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()])  
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_aircraft(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_aircraft_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_mscoco(datasetName):
    if datasetName=='train':
        return 'mscoco is not a train dataset in metadataset (see official split)'
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_mscoco_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_vggflower(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_vgg_flower_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
 
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_quickdraw(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_quickdraw_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(28), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_omniglot(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_omniglot_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()])

    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def metadataset_traffic_signs(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metadataset_traffic_signs_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    if datasetName == 'train':
        trans = transforms.Compose([ totensor(), norm(), bi_resize()])
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([ totensor(), norm(), bi_resize()])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(126), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def audioset(datasetName):
    def randcrop(tensor, duration):
        freq = 32000 * duration
        N = tensor.shape[0]
        if N<freq:
            new_tensor = torch.zeros(freq)
            new_tensor[:N] = tensor
            return new_tensor
        
        if N > freq:
            i = random.randint(0,N - freq - 1)
        else:
            i = 0
        return tensor[i:i+freq]

    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["audioset_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    
    trans = transforms.Compose([lambda x : randcrop(x.mean(dim=0), duration = 1).unsqueeze(0).to(dtype=torch.float), lambda x: x + 0.1 * torch.randn_like(x), lambda x: -1 * x if random.random() < 0.5 else x])
    test_trans = lambda x : randcrop(x.mean(dim=0), duration = 1).unsqueeze(0).to(dtype=torch.float)
    target_trans = lambda x: torch.zeros(dataset['num_classes']).scatter_(0,torch.Tensor(x).long(), 1.)
    opener = lambda x: torch.load(x, map_location='cpu')

    return {"dataloader": dataLoader(DataHolder(data, targets, trans if datasetName == "train" else test_trans, target_transforms=target_trans, opener=opener), shuffle = datasetName == "train"), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def esc50(datasetName):
    def randcrop(tensor, duration,sr=44100):
        nsamples = sr * duration
        N = tensor.shape[0]
        if N<nsamples:
            new_tensor = torch.zeros(nsamples)
            new_tensor[:N] = tensor
            return new_tensor
        
        if N > nsamples:
            i = random.randint(0,N - nsamples - 1)
        else:
            i = 0
        return tensor[i:i+nsamples]

    f = open(args.dataset_path + "datasets.json")
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["esc50fs_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    
    #trans = transforms.Compose([lambda x : randcrop(x, duration = 3).unsqueeze(0), lambda x: x + 0.1 * torch.randn_like(x), lambda x: -1 * x if random.random() < 0.5 else x])
    trans = lambda x : x.unsqueeze(0)
    test_trans = lambda x : x.unsqueeze(0)
    opener = lambda x: torch.load(x, map_location='cpu')

    return {"dataloader": dataLoader(DataHolder(data, targets, trans if datasetName == "train" else test_trans, opener=opener), shuffle = datasetName == "train"), 
    "name":dataset['name'], 
    "num_classes":dataset["num_classes"], 
    "name_classes": dataset["name_classes"]}

def metaalbum(source, is_train=False):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["metaalbum_"+source]
    data = dataset["data"]
    targets = dataset["targets"]
    normalization = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    image_size = 224 if args.backbone == "resnet50" else 126
    if is_train:
        trans = transforms.Compose([
            transforms.RandomResizedCrop(image_size), transforms.ToTensor(), normalization, GaussianNoise(0.1533), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip()]) 
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(image_size), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = is_train), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def prepareDataLoader(name, is_train=False):
    if isinstance(name, str):
        name = [name]
    result = []
    dataset_options = {
            "cifar10_train": lambda: cifar10("train"),
            "cifar10_validation": lambda: cifar10("validation"),
            "cifar10_test": lambda: cifar10("test"),
            "cifar100_train": lambda: cifar100("train"),
            "cifar100_validation": lambda: cifar100("validation"),
            "cifar100_test": lambda: cifar100("test"),
            "mnist_train": lambda: mnist("train"),
            "mnist_validation": lambda: mnist("validation"),
            "mnist_test": lambda: mnist("test"),
            "fashion-mnist_train": lambda: fashionMnist("train"),
            "fashion-mnist_validation": lambda: fashionMnist("validation"),
            "fashion-mnist_test": lambda: fashionMnist("test"),
            "imagenet_train": lambda: imagenet("train"),
            "imagenet_validation": lambda: imagenet("validation"),
            "imagenet_test": lambda: imagenet("test"),
            "miniimagenet_train": lambda: miniimagenet("train"),
            "miniimagenet_validation": lambda: miniimagenet("validation"),
            "miniimagenet_test": lambda: miniimagenet("test"),
            "tieredimagenet_train": lambda: tieredimagenet("train"),
            "tieredimagenet_validation": lambda: tieredimagenet("validation"),
            "tieredimagenet_test": lambda: tieredimagenet("test"),
            "cifarfs_train": lambda: cifarfs("train"),
            "cifarfs_validation": lambda: cifarfs("validation"),
            "cifarfs_test": lambda: cifarfs("test"),
            "metadataset_imagenet_train": lambda: metadataset_imagenet("train"),
            "metadataset_imagenet_validation": lambda: metadataset_imagenet("validation"),
            "metadataset_imagenet_test": lambda: metadataset_imagenet("test"),
            "metadataset_cub_train": lambda: metadataset_cub("train"),
            "metadataset_cub_validation": lambda: metadataset_cub("validation"),
            "metadataset_cub_test": lambda: metadataset_cub("test"),
            "metadataset_dtd_train": lambda: metadataset_dtd("train"),
            "metadataset_dtd_validation": lambda: metadataset_dtd("validation"),
            "metadataset_dtd_test": lambda: metadataset_dtd("test"),
            "metadataset_fungi_train": lambda: metadataset_fungi("train"),
            "metadataset_fungi_validation": lambda: metadataset_fungi("validation"),
            "metadataset_fungi_test": lambda: metadataset_fungi("test"),
            "metadataset_aircraft_train": lambda: metadataset_aircraft("train"),
            "metadataset_aircraft_validation": lambda: metadataset_aircraft("validation"),
            "metadataset_aircraft_test": lambda: metadataset_aircraft("test"),
            "metadataset_mscoco_train": lambda: metadataset_mscoco("train"),
            "metadataset_mscoco_validation": lambda: metadataset_mscoco("validation"),
            "metadataset_mscoco_test": lambda: metadataset_mscoco("test"),
            "metadataset_cub_train": lambda: metadataset_cub("train"),
            "metadataset_cub_validation": lambda: metadataset_cub("validation"),
            "metadataset_cub_test": lambda: metadataset_cub("test"),
            "metadataset_omniglot_train": lambda: metadataset_omniglot("train"),
            "metadataset_omniglot_validation": lambda: metadataset_omniglot("validation"),
            "metadataset_omniglot_test": lambda: metadataset_omniglot("test"),
            "metadataset_quickdraw_train": lambda: metadataset_quickdraw("train"),
            "metadataset_quickdraw_validation": lambda: metadataset_quickdraw("validation"),
            "metadataset_quickdraw_test": lambda: metadataset_quickdraw("test"),
            "metadataset_vgg_flower_train": lambda: metadataset_vggflower("train"),
            "metadataset_vgg_flower_validation": lambda: metadataset_vggflower("validation"),
            "metadataset_vgg_flower_test": lambda: metadataset_vggflower("test"),
            "metadataset_traffic_signs_train": lambda: metadataset_traffic_signs("train"),
            "metadataset_traffic_signs_validation": lambda: metadataset_traffic_signs("validation"),
            "metadataset_traffic_signs_test": lambda: metadataset_traffic_signs("test"),
            "audioset_train":lambda: audioset("train"),
            "audioset_test":lambda: audioset("test"), 
            "esc50fs_train":lambda: esc50("train"),
            "esc50fs_val":lambda: esc50("validation"),
            "esc50fs_test":lambda: esc50("test"),
            "metaalbum_micro":lambda: metaalbum("Micro", is_train=is_train),
            "metaalbum_mini":lambda: metaalbum("Mini", is_train=is_train),
            "metaalbum_extended":lambda: metaalbum("Extended", is_train=is_train),
        }
    # Meta albums
    for setting in ['Micro', 'Macro', 'Extended']:
        for album in ['BCT', 'BRD', 'CRS', 'FLW', 'MD_MIX', 'PLK', 'PLT_VIL', 'RESISC', 'SPT', 'TEX']:
            dataset_options[f'metaalbum_{album.lower()}_{setting.lower()}'] = lambda: metaalbum(f'{album}_{setting}', is_train=is_train)    
                 
    for elt in name:
        assert elt.lower() in dataset_options.keys(), f'The chosen dataset "{elt}" is not existing, please provide a valid option: \n {list(dataset_options.keys())}'
        result.append(dataset_options[elt.lower()]())
    return result
    
if args.training_dataset != "":
    try:
        eval(args.training_dataset)
        trainSet = prepareDataLoader(eval(args.training_dataset), is_train=True)
    except NameError:
        trainSet = prepareDataLoader(args.training_dataset, is_train=True)
else:
    trainSet = []

if args.validation_dataset != "":
    try:
        eval(args.validation_dataset)
        validationSet = prepareDataLoader(eval(args.validation_dataset), is_train=False)
    except NameError:
        validationSet = prepareDataLoader(args.validation_dataset, is_train=False)
else:
    validationSet = []

if args.test_dataset != "":
    try:
        eval(args.test_dataset)
        testSet = prepareDataLoader(eval(args.test_dataset), is_train=False)
    except NameError:
        testSet = prepareDataLoader(args.test_dataset, is_train=False)
else:
    testSet = []
    
print(" dataloaders,", end='')
