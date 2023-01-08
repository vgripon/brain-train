### file to generate dataloaders
### for simplicity, we do not offer the choice to load the whole dataset in VRAM anymore

from torchvision import transforms, datasets
import random
from args import args
import torch
import torch.nn as nn
import os
import json
import numpy as np
from PIL import Image
import copy
from selfsupervised.selfsupervised import get_ssl_transform
from utils import *
from few_shot_evaluation import EpisodicGenerator
from augmentations import parse_transforms
### first define dataholder, which will be used as an argument to dataloaders
all_steps = [item for sublist in eval(args.steps) for item in sublist]
supervised = 'lr' in all_steps or 'rotations' in all_steps or 'mixup' in all_steps or 'manifold mixup' in all_steps or (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "" or args.episodic
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
class CategoriesSampler():
    """
        Sampler for episodic training
    """
    def __init__(self, datasetName):
        self.batch_size = args.batch_size
        self.generator = EpisodicGenerator(datasetName=datasetName, dataset_path=args.dataset_path)
        self.n_ways = args.few_shot_ways
        self.n_shots = args.few_shot_shots
        self.n_queries = args.few_shot_queries
        self.episodic_iterations_per_epoch = args.episodic_iterations_per_epoch
    def __len__(self):
        return self.episodic_iterations_per_epoch
    
    def __iter__(self):
        """
            Return indices used in one batch
            data is returned in a sequence of c1c1c1c1c2c2c2c2c3c3c3c3 with shots first then queries
        """
        for _ in range(self.episodic_iterations_per_epoch):
            episode = self.generator.sample_episode(ways=self.n_ways, n_shots=self.n_shots, n_queries=self.n_queries)
            batch = []
            for c, class_idx in enumerate(episode['choice_classes']):
                offset = sum(self.generator.num_elements_per_class[:class_idx])
                batch = batch + [offset+s for s in episode['shots_idx'][c]+episode['queries_idx'][c]]
            batch = torch.tensor(batch)
            yield batch

def dataLoader(dataholder, shuffle, datasetName, episodic):
    if episodic : 
        sampler = CategoriesSampler(datasetName=datasetName)
        return torch.utils.data.DataLoader(dataholder, num_workers = min(os.cpu_count(), 8), batch_sampler=sampler)
    return torch.utils.data.DataLoader(dataholder, batch_size = args.batch_size, shuffle = shuffle, num_workers = min(os.cpu_count(), 8))

class TransformWrapper(object):
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

def get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms):
    if datasetName == 'train':
        supervised_transform_str = args.training_transforms if len(args.training_transforms) > 0 else default_train_transforms
        supervised_transform = parse_transforms(supervised_transform_str, image_size) 
        all_transforms = {}
        if supervised:
            all_transforms['supervised'] = transforms.Compose(supervised_transform)
        all_transforms.update(get_ssl_transform(image_size, normalization=supervised_transform[-1]))
        trans = TransformWrapper(all_transforms)
    else:
        trans = transforms.Compose(parse_transforms(args.test_transforms if len(args.test_transforms) > 0 else default_test_transforms, image_size))
    return trans

def cifar10(datasetName):
    pytorchDataset = datasets.CIFAR10(args.dataset_path, train = datasetName != "test", download = 'cifar-10-python.tar.gz' not in os.listdir(args.dataset_path))
    data = torch.tensor(pytorchDataset.data).transpose(1,3).transpose(2,3).float() / 256.
    targets = pytorchDataset.targets
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 32
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 32
    default_train_transforms = ['randomcrop','randomhorizontalflip', 'totensor', 'cifar10norm']
    if args.sample_aug == 1:
        default_test_transforms = ['centercrop','totensor', 'cifar10norm']
    else:
        default_test_transforms = ['randomresizecrop', 'totensor', 'cifar10norm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)
   
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":"cifar10_" + datasetName, "num_classes":10, "name_classes": pytorchDataset.classes}

def cifar100(datasetName):
    pytorchDataset = datasets.CIFAR100(args.dataset_path, train = datasetName != "test", download = 'cifar-100-python.tar.gz' not in os.listdir(args.dataset_path))
    data = torch.tensor(pytorchDataset.data).transpose(1,3).transpose(2,3).float() / 256.
    targets = pytorchDataset.targets

    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 32
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 32
    default_train_transforms = ['randomcrop','randomhorizontalflip', 'totensor', 'cifar100norm']
    if args.sample_aug == 1:
        default_test_transforms = ['centercrop','totensor', 'cifar100norm']
    else:
        default_test_transforms = ['randomresizecrop', 'totensor', 'cifar100norm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)

    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train"), "name":"cifar100_" + datasetName, "num_classes":100, "name_classes": pytorchDataset.classes}

def miniimagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["miniimagenet_" + datasetName]
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 84
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 84
    default_train_transforms = ['randomresizedcrop','colorjitter', 'randomhorizontalflip', 'totensor', 'miniimagenetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['resize_92/84', 'centercrop', 'totensor', 'miniimagenetnorm']
    else:
        default_test_transforms = ['randomresizecrop', 'totensor', 'miniimagenetnorm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)

    return {"dataloader": dataLoader(DataHolder(dataset["data"], dataset["targets"], trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="miniimagenet_"+datasetName), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def tieredimagenet(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["tieredimagenet_" + datasetName]
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 84
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 84
    default_train_transforms = ['randomresizedcrop','colorjitter', 'randomhorizontalflip', 'totensor', 'miniimagenetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['resize_92/84', 'centercrop', 'totensor', 'miniimagenetnorm']
    else:
        default_test_transforms = ['randomresizecrop', 'totensor', 'miniimagenetnorm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)

    return {"dataloader": dataLoader(DataHolder(dataset["data"], dataset["targets"], trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="tieredimagenet_"+datasetName), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def cifarfs(datasetName):
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["cifarfs_" + datasetName]
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 32
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 32

    default_train_transforms = ['randomresizedcrop','colorjitter', 'randomhorizontalflip', 'totensor', 'imagenetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['resize_115/100', 'centercrop', 'totensor', 'imagenetnorm']
    else:
        default_test_transforms = ['randomresizecrop', 'totensor', 'imagenetnorm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)
    
    return {"dataloader": dataLoader(DataHolder(dataset["data"], dataset["targets"], trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="cifarfs_"+datasetName), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def imagenet(datasetName):
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 224
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 224
    default_train_transforms = ['randomresizedcrop','randomhorizontalflip', 'totensor', 'imagenetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['resize_256/224', 'centercrop', 'totensor', 'imagenetnorm']
    else:
        default_test_transforms = ['randomresizedcrop', 'totensor', 'imagenetnorm'] 
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)
    pytorchDataset = datasets.ImageNet(args.dataset_path + "/imagenet", split = "train" if datasetName != "test" else "val", transform = trans)
        
    return {"dataloader": dataLoader(pytorchDataset, shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="imagenet_"+datasetName), "name":"imagenet_" + datasetName, "num_classes":1000, "name_classes": pytorchDataset.classes}

def metadataset(datasetName, name):
    """
    Generic function to load a dataset from the Meta-Dataset v1.0
    """
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets[name+"_" + datasetName]
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 126
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 126
    default_train_transforms = ['metadatasettotensor', 'biresize', 'metadatasetnorm']
    if args.sample_aug == 1:
        default_test_transforms = ['metadatasettotensor', 'biresize', 'metadatasetnorm']
    else:
        default_test_transforms = ['metadatasettotensor', 'randomresizedcrop', 'biresize', 'metadatasetnorm']
    trans = get_transforms(image_size, datasetName, default_train_transforms, default_test_transforms)
    return {"dataloader": dataLoader(DataHolder(dataset["data"], dataset["targets"], trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName=name+"_"+datasetName), "name":dataset["name"], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}

def metadataset_imagenet_v2():
    f = open(args.dataset_path + "datasets.json")    
    all_datasets = json.loads(f.read())
    f.close()
    dataset_train = all_datasets["metadataset_imagenet_train"]
    dataset_validation = all_datasets["metadataset_imagenet_validation"]
    dataset_test = all_datasets["metadataset_imagenet_test"]
    data = dataset_train["data"] + dataset_validation["data"] + dataset_test["data"]
    train_classes = dataset_train["num_classes"]
    validation_classes = dataset_validation["num_classes"]
    test_classes = dataset_test["num_classes"]
    num_classes = train_classes + validation_classes + test_classes
    targets = dataset_train["targets"] + [t + train_classes for t in dataset_validation["targets"]] + [t + train_classes + validation_classes for t in dataset_test["targets"]]

    image_size = args.training_image_size if args.training_image_size>0 else 126
    default_train_transforms = ['metadatasettotensor','metadatasetnorm', 'beresize']
    trans = get_transforms(image_size, "train", default_train_transforms, [])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = True, episodic=args.episodic, datasetName="metadataset_imagenet_train_v2"), "name":"metadataset_imagenet_v2_train", "num_classes":num_classes, "name_classes": dataset_train["name_classes"]+dataset_validation["name_classes"]+dataset_test["name_classes"]}

def mnist(datasetName):
    pytorchDataset = datasets.MNIST(args.dataset_path, train = datasetName != "test", download = 'MNIST' not in os.listdir(args.dataset_path))
    data = pytorchDataset.data.clone().unsqueeze(1).float() / 256.
    targets = pytorchDataset.targets
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 28
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 28
    default_transform = ['totensor', 'mnistnorm']
    trans = get_transforms(image_size, datasetName, default_transform, default_transform)
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="mnist_"+datasetName), "name": "mnist_" + datasetName, "num_classes": 10, "name_classes": list(range(10))}

def fashionMnist(datasetName):
    pytorchDataset = datasets.FashionMNIST(args.dataset_path, train = datasetName != "test", download = 'FashionMNIST' not in os.listdir(args.dataset_path))
    data = pytorchDataset.data.clone().unsqueeze(1).float() / 256.
    targets = pytorchDataset.targets

    normalization = transforms.Normalize((0.2849,), (0.3516,))
    if datasetName == "train":
        image_size = args.training_image_size if args.training_image_size>0 else 28
    else:
        image_size = args.test_image_size if args.test_image_size>0 else 28
    default_transform = ['totensor', 'mnistnorm']
    trans = get_transforms(image_size, datasetName, default_transform, default_transform)
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = datasetName == "train", episodic=args.episodic and datasetName == "train", datasetName="fashionmnist_"+datasetName), "name": "fashion-mnist_" + datasetName, "num_classes": 10, "name_classes": pytorchDataset.classes}

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
    f = open(args.dataset_path + "datasets.json")
    all_datasets = json.loads(f.read())
    f.close()
    dataset = all_datasets["esc50fs_" + datasetName]
    data = dataset["data"]
    targets = dataset["targets"]
    
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
        supervised_transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size), transforms.ToTensor(), normalization, GaussianNoise(0.1533), transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4), transforms.RandomHorizontalFlip()]) 
        all_transforms = {}
        if 'lr' in all_steps or 'rotations' in all_steps or 'mixup' in all_steps or 'manifold mixup' in all_steps or (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "":
            all_transforms['supervised'] = supervised_transform
        all_transforms.update(get_ssl_transform(image_size, normalization))
        trans = TransformWrapper(all_transforms)
    else:
        if args.sample_aug == 1:
            trans = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size), transforms.ToTensor(), normalization])
        else:
            trans = transforms.Compose([transforms.RandomResizedCrop(image_size), transforms.ToTensor(), normalization])
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = is_train), "name":dataset['name'], "num_classes":dataset["num_classes"], "name_classes": dataset["name_classes"]}


def load_features(path = args.test_features, is_train = True):
    trans = transforms.Lambda(lambda x:x)
    features = torch.load(path)
    num_classes = len(features)
    name_classes = [x['name_class'] for x in features]
    data = features[0]['features']
    targets = [0] * data.shape[0]
    for i,x in enumerate(features[1:]):
        feat = x['features']
        targets+=[i]*feat.shape[0]
        data = torch.cat((data,feat), dim = 0)
    return {"dataloader": dataLoader(DataHolder(data, targets, trans), shuffle = is_train ,episodic=args.episodic, datasetName="loaded_features"), "name":path, "num_classes":num_classes, "name_classes": name_classes}

def prepareDataLoader(name, is_train=False):
    if isinstance(name, str):
        name = [name]
    result = []
    train_trans_results = []
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
            "metadataset_imagenet_train": lambda: metadataset("train", "metadataset_imagenet"),
            "metadataset_imagenet_validation": lambda: metadataset("validation", "metadataset_imagenet"),
            "metadataset_imagenet_test": lambda: metadataset("test", "metadataset_imagenet"),
            "metadataset_cub_train": lambda: metadataset("train", "metadataset_cub"),
            "metadataset_cub_validation": lambda: metadataset("validation", "metadataset_cub"),
            "metadataset_cub_test": lambda: metadataset("test", "metadataset_cub"),
            "metadataset_dtd_train": lambda: metadataset("train", "metadataset_dtd"),
            "metadataset_dtd_validation": lambda: metadataset("validation", "metadataset_dtd"),
            "metadataset_dtd_test": lambda: metadataset("test", "metadataset_dtd"),
            "metadataset_fungi_train": lambda: metadataset("train", "metadataset_fungi"),
            "metadataset_fungi_validation": lambda: metadataset("validation", "metadataset_fungi"),
            "metadataset_fungi_test": lambda: metadataset("test", "metadataset_fungi"),
            "metadataset_aircraft_train": lambda: metadataset("train", "metadataset_aircraft"),
            "metadataset_aircraft_validation": lambda: metadataset("validation", "metadataset_aircraft"),
            "metadataset_aircraft_test": lambda: metadataset("test", "metadataset_aircraft"),
            "metadataset_mscoco_train": lambda: metadataset("train", "metadataset_mscoco"),
            "metadataset_mscoco_validation": lambda: metadataset("validation", "metadataset_mscoco"),
            "metadataset_mscoco_test": lambda: metadataset("test", "metadataset_mscoco"),
            "metadataset_cub_train": lambda: metadataset("train", "metadataset_cub"),
            "metadataset_cub_validation": lambda: metadataset("validation", "metadataset_cub"),
            "metadataset_cub_test": lambda: metadataset("test", "metadataset_cub"),
            "metadataset_omniglot_train": lambda: metadataset("train", "metadataset_omniglot"),
            "metadataset_omniglot_validation": lambda: metadataset("validation", "metadataset_omniglot"),
            "metadataset_omniglot_test": lambda: metadataset("test", "metadataset_omniglot"),
            "metadataset_quickdraw_train": lambda: metadataset("train", "metadataset_quickdraw"),
            "metadataset_quickdraw_validation": lambda: metadataset("validation", "metadataset_quickdraw"),
            "metadataset_quickdraw_test": lambda: metadataset("test", "metadataset_quickdraw"),
            "metadataset_vgg_flower_train": lambda: metadataset("train", "metadataset_vgg_flower"),
            "metadataset_vgg_flower_validation": lambda: metadataset("validation", "metadataset_vgg_flower"),
            "metadataset_vgg_flower_test": lambda: metadataset("test", "metadataset_vgg_flower"),
            "metadataset_traffic_signs_train": lambda: metadataset("train", "metadataset_traffic_signs"),
            "metadataset_traffic_signs_validation": lambda: metadataset("validation", "metadataset_traffic_signs"),
            "metadataset_traffic_signs_test": lambda: metadataset("test", "metadataset_traffic_signs"),
            "metadataset_imagenet_v2_train": lambda: metadataset_imagenet_v2(),
            "audioset_train":lambda: audioset("train"),
            "audioset_test":lambda: audioset("test"), 
            "esc50fs_train":lambda: esc50("train"),
            "esc50fs_val":lambda: esc50("validation"),
            "esc50fs_test":lambda: esc50("test"),
            "metaalbum_micro":lambda: metaalbum("Micro", is_train=is_train),
            "metaalbum_mini":lambda: metaalbum("Mini", is_train=is_train),
            "metaalbum_extended":lambda: metaalbum("Extended", is_train=is_train),
            "load_features": lambda: load_features(args.test_features, is_train = is_train)
        }
    # Adding Meta albums
    for setting in ['Micro', 'Macro', 'Extended']:
        for album in ['BCT', 'BRD', 'CRS', 'FLW', 'MD_MIX', 'PLK', 'PLT_VIL', 'RESISC', 'SPT', 'TEX']:
            dataset_options[f'metaalbum_{album.lower()}_{setting.lower()}'] = lambda: metaalbum(f'{album}_{setting}', is_train=is_train)    
                 
    for elt in name:
        assert elt.lower() in dataset_options.keys(), f'The chosen dataset "{elt}" is not existing, please provide a valid option: \n {list(dataset_options.keys())}'
        result.append(dataset_options[elt.lower()]())
    return result
def checkSize(dataset):
        if 'cifar' in dataset:
            image_size = 32
        elif 'mnist' in dataset or 'omniglot' in dataset:
            image_size = 28
        elif 'imagenet' in dataset and 'metadataset' not in dataset and 'miniimagenet' not in dataset:
            image_size = 224
        elif 'metadataset' in dataset: 
            image_size = 126
        elif 'miniimagenet' in dataset or 'tieredimagenet' in dataset or 'cub' in dataset:
            image_size = 84
        return image_size

if args.training_dataset != "":
    try:
        eval(args.training_dataset)
        trainSet = prepareDataLoader(eval(args.training_dataset), is_train=True)
        if args.training_image_size == -1:
            args.training_image_size = checkSize(eval(args.training_dataset)[0])
    except NameError:
        trainSet = prepareDataLoader(args.training_dataset, is_train=True)
        if args.training_image_size == -1:
            args.training_image_size = checkSize(args.training_dataset)
else:
    trainSet = []
if args.validation_dataset != "":
    try:
        eval(args.validation_dataset)
        validationSet = prepareDataLoader(eval(args.validation_dataset), is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(eval(args.validation_dataset)[0])
    except NameError:
        validationSet = prepareDataLoader(args.validation_dataset, is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(args.validation_dataset)
else:
    validationSet = []

if args.test_dataset != "":
    try:
        eval(args.test_dataset)
        testSet = prepareDataLoader(eval(args.test_dataset), is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(eval(args.test_dataset)[0])
    except NameError:
        testSet = prepareDataLoader(args.test_dataset, is_train=False)
        if args.test_image_size == -1:
            args.test_image_size = checkSize(args.test_dataset)
else:
    testSet = []


if args.training_dataset == "" and args.test_dataset == "":
    trainSet = prepareDataLoader("load_features", is_train=True)
    cleanSet = prepareDataLoader("load_features", is_train=False)

print(" dataloaders,", end='')
