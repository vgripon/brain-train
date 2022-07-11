import argparse
import os
import random # for seed

parser = argparse.ArgumentParser(description="BRAIn team routines for training on standard benchmarks")

### generic args
parser.add_argument("--device", type=str, default="cuda:0", help="device to use")
parser.add_argument("--runs", type=int, default=1, help="how many times to repeat experiments for increased statistical robustness")
parser.add_argument("--csv", type=str, default="", help="name of csv file to write as output, will erase existing file if it exists")
parser.add_argument("--save-features-prefix", type=str, default="", help="save features of validation and test sets to hard drive, use this parameter as prefix to file names")
parser.add_argument("--save-backbone", type=str, default="", help="save backbone to hard drive at the specified location")
parser.add_argument("--load-backbone", type=str, default="", help="load backbone from hard drive at the specified location")
parser.add_argument("--freeze-backbone", action="store_true", help="freeze the backbone during training, can be useful in conjonction with load-backbone")
parser.add_argument("--skip-epochs", type=int, default=0, help="number of epochs for which validation and test are ignored")
parser.add_argument("--seed", type=int, default=random.randint(0, 1000000000), help="initial random seed")
parser.add_argument("--deterministic", action="store_true", help="force deterministic mode for cuda")

### optimizer args
parser.add_argument("--optimizer", type=str, default="SGD", help="can be SGD or Adam")
parser.add_argument("--lr", type=float, default=-1., help="initial learning rate, defaut to 0.1 for SGD and 0.001 for Adam")
parser.add_argument("--wd", type=float, default=-1., help="weight decay, default to 5e-4 for SGD and 0 for Adam")
parser.add_argument("--steps", type=str, default="[[]]", help="describe what steps during training are made of, is a list of lists containing 'rotations', 'mixup' or 'manifold mixup', for example \"[['manifold mixup'],['rotations']]\" does two steps: first with manifold mixup then with rotations as additional self-supervision. Last list is used to compute losses and scores")
parser.add_argument("--label-smoothing", type=float, default=0, help="use label smoothing with given smoothing factor. 0 means no smoothing")

### dataloaders args
# list of datasets, which automatically define a train, a validation and a test set
datasets = {
    "cifar10": ("cifar10_train", "", "cifar10_test"),
    "cifar100": ("cifar100_train", "", "cifar100_test"),
    "mnist": ("mnist_train", "", "mnist_test"),
    "fashion-mnist": ("fashion-mnist_train", "", "fashion-mnist_test"),
    "imagenet": ("imagenet_train", "", "imagenet_test"),
    "miniimagenet": ("miniimagenet_train", "miniimagenet_validation", "miniimagenet_test"),
    "tieredimagenet": ("tieredimagenet_train", "tieredimagenet_validation", "tieredimagenet_test"),
    "cifarfs": ("cifarfs_train", "cifarfs_validation", "cifarfs_test"),
    "metadataset_imagenet": ("metadataset_imagenet_train", "metadataset_imagenet_validation", "metadataset_imagenet_test"),
}

parser.add_argument("--dataset-path", type=str, default=os.environ.get("DATASETS"), help="path to dataset files")
parser.add_argument("--batch-size", type=int, default=64, help="batch size")
parser.add_argument("--dataset", type=str, default="", help="dataset to use, can be any of " + str(datasets.keys()) + " or blank if using a direct choice for training-dataset, validation-dataset and test-dataset")
parser.add_argument("--training-dataset", type=str, default="", help="training dataset, overriden by --dataset")
parser.add_argument("--validation-dataset", type=str, default="", help="validation dataset, overriden by --dataset")
parser.add_argument("--test-dataset", type=str, default="", help="test dataset, overriden by --dataset")
parser.add_argument("--dataset-size", type=int, default=0, help="defines a maximum of samples considered at each epoch, 0 means it is ignored")

### backbones parameters
parser.add_argument("--feature-maps", type=int, default=64, help="initial number of feature maps in first embedding, used as a base downstream convolutions")
parser.add_argument("--backbone", type=str, default="resnet18", help="backbone architecture")
parser.add_argument("--feature-processing", type=str, default="", help="feature processing before few-shot classifiers, can contain M (remove mean of feature vectors), and E (unit sphere projection of feature vectors)")

### criterion
parser.add_argument("--classifier", type=str, default="lr", help="define which classifier is used on top of selected backbone, can be any of lr for logistic regression, or L2 for euclidean distance regression")

### scheduler parameters
parser.add_argument("--epochs", type=int, default=350, help="total number of training epochs")
parser.add_argument("--milestones", type=str, default="[100,200,300]", help="milestones for scheduler")
parser.add_argument("--gamma", type=float, default=0.1, help="learning rate multiplier after each milestone")
parser.add_argument("--cosine", action="store_true", help="use cosine annealing instead of multisteplr")

### few shot evaluation
parser.add_argument("--few-shot", action="store_true", help="evaluation using few shot tasks")
parser.add_argument("--few-shot-runs", type=int, default=10000, help="total number of few shot runs")
parser.add_argument("--few-shot-ways", type=int, default=5, help="number of classes in generated few shot tasks")
parser.add_argument("--few-shot-shots", type=int, default=1, help="number of shots per class in generated few shot tasks")
parser.add_argument("--few-shot-queries", type=int, default=15, help="number of query vectors per class in generated few shot tasks")
parser.add_argument("--few-shot-unbalanced-queries", action="store_true", help="use unbalanced number of queries per class. The number of queries per class is sampled using a dirichlet distribution between 1 and the query set for all classes")
parser.add_argument("--few-shot-classifier", type=str, default="ncm", help="classifier for few-shot runs, can be ncm or knn where k is an integer")
parser.add_argument("--test-features", type=str, default="", help="test few-shot runs on saved features")

args = parser.parse_args()

if args.dataset != "":
    try:
        eval(args.dataset)
        args.training_dataset, args.validation_dataset, args.test_dataset = [], [], []
        for name in eval(args.dataset):
            tr, v, te = datasets[name]
            args.training_dataset.append(tr)
            args.validation_dataset.append(v)
            args.test_dataset.append(te)
    except NameError:
        args.training_dataset, args.validation_dataset, args.test_dataset = datasets[args.dataset]

if args.lr < 0:
    args.lr = 0.1 if args.optimizer.lower() == "sgd" else 1e-3
if args.wd < 0:
    args.wd = 5e-4 if args.optimizer.lower() == "sgd" else 0

if isinstance(eval(args.milestones), int):
    args.milestones = [eval(args.milestones) * i for i in range(1, args.epochs // eval(args.milestones))]
else:
    args.milestones = eval(args.milestones)
if args.epochs not in args.milestones:
    args.milestones.append(args.epochs)

print(" args,", end = '')
