from args import args
import torchvision
import json
import os

imagenet = torchvision.datasets.ImageNet(root = args.dataset_path + "imagenet", split="train")

all_results = {}

### generate data for miniimagenet
for dataset in ["train","validation","test"]:
    firstLine = True
    classToIdx = {}
    nClass = 0
    result = {"data":[], "targets":[], "name":"miniimagenet_" + dataset, "num_classes":0, "name_classes":[]}
    with open(args.dataset_path + "miniimagenetimages/" + dataset + ".csv") as f:
        for row in f:
            if firstLine:
                firstLine = False
            else:
                fileName, classIdx = row.split(",")
                classIdx = classIdx.split("\n")[0]
                if classIdx in classToIdx.keys():
                    result["data"].append("miniimagenetimages/images/" + fileName)
                    result["targets"].append(classToIdx[classIdx])
                else:
                    classToIdx[classIdx] = nClass
                    result["data"].append("miniimagenetimages/images/" + fileName)
                    result["targets"].append(nClass)
                    result["name_classes"].append(str(imagenet.classes[imagenet.wnid_to_idx[classIdx]]))
                    nClass += 1
    result["num_classes"] = nClass
    all_results["miniimagenet_" + dataset] = result
    print("Done for miniimagenet_" + dataset + " with " + str(nClass) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

### generate data for tieredimagenet
for dataset,folderName in [("train","train"),("validation","val"),("test","test")]:
    directories = os.listdir(args.dataset_path + "tieredimagenet/" + folderName)
    result = {"data":[], "targets":[], "name":"tieredimagenet_" + dataset, "num_classes":0, "name_classes":[]}
    for i,classIdx in enumerate(directories):
        for fileName in os.listdir(args.dataset_path + "tieredimagenet/" + folderName + "/" + classIdx):
            result["data"].append("tieredimagenet/" + folderName + "/" + classIdx + "/" + fileName)
            result["targets"].append(i)
        result["name_classes"].append(str(imagenet.classes[imagenet.wnid_to_idx[classIdx]]))
        
    result["num_classes"] = i + 1
    all_results["tieredimagenet_" + dataset] = result
    print("Done for tieredimagenet_" + dataset + " with " + str(nClass) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

### generate data for cifarfs
for dataset,folderName in [("train","meta-train"),("validation","meta-val"),("test","meta-test")]:
    directories = os.listdir(args.dataset_path + "cifar_fs/" + folderName)
    result = {"data":[], "targets":[], "name":"cifarfs_" + dataset, "num_classes":0, "name_classes":[]}
    for i,classIdx in enumerate(directories):
        for fileName in os.listdir(args.dataset_path + "cifar_fs/" + folderName + "/" + classIdx):
            result["data"].append("cifar_fs/" + folderName + "/" + classIdx + "/" + fileName)
            result["targets"].append(i)
        result["name_classes"].append(classIdx)
        
    result["num_classes"] = i + 1
    all_results["cifarfs_" + dataset] = result
    print("Done for cifarfs_" + dataset + " with " + str(nClass) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

f = open(args.dataset_path + "datasets.json", "w")
f.write(json.dumps(all_results))
f.close()
