# Loading main libraries
import torch
import random # for mixup
import numpy as np # for manifold mixup
from colorama import Fore, Style

# Loading other files
print("Loading local files... ", end ='')
from args import args
from utils import *
from dataloaders import trainSet, validationSet, testSet
import classifiers
import backbones
import backbones1d
from few_shot_evaluation import EpisodicGenerator, ImageNetGenerator, OmniglotGenerator
print(" done.")

print()
print(args)
print()

### generate random seeds
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
if args.deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(epoch, backbone, criterion, optimizer, scheduler):
    backbone.train()
    for c in criterion:
        c.train()
    iterators = [enumerate(dataset["dataloader"]) for dataset in trainSet]
    losses, accuracies, total_elt = torch.zeros(len(iterators)), torch.zeros(len(iterators)), torch.zeros(len(iterators))
    while True:
        try:
            optimizer.zero_grad()
            text = ""
            for trainingSetIdx in range(len(iterators)):
                if args.dataset_size > 0 and total_elt[trainingSetIdx] > args.dataset_size:
                    raise StopIteration
                batchIdx, (data, target) = next(iterators[trainingSetIdx])
                data, target = data.to(args.device), target.to(args.device)

                for step in eval(args.steps):
                    dataStep = data.clone()
                    
                    if "mixup" in step or "manifold mixup" in step:
                        perm = torch.randperm(dataStep.shape[0])
                        if "mixup" in step:
                            lbda = random.random()
                            mixupType = "mixup"
                        else:
                            lbda = np.random.beta(2,2)
                            mixupType = "manifold mixup"
                    else:
                        lbda, perm, mixupType = None, None, None

                    if "rotations" in step:
                        bs = dataStep.shape[0] // 4
                        targetRot = torch.LongTensor(dataStep.shape[0]).to(args.device)
                        targetRot[:bs] = 0
                        dataStep[bs:] = dataStep[bs:].transpose(3,2).flip(2)
                        targetRot[bs:2*bs] = 1
                        dataStep[2*bs:] = dataStep[2*bs:].transpose(3,2).flip(2)
                        targetRot[2*bs:3*bs] = 2
                        dataStep[3*bs:] = dataStep[3*bs:].transpose(3,2).flip(2)
                        targetRot[3*bs:] = 3
                    else:
                        targetRot = None

                    loss, score = criterion[trainingSetIdx](backbone(dataStep, mixup = mixupType, lbda = lbda, perm = perm), target, yRotations = targetRot if "rotations" in step else None, lbda = lbda, perm = perm)
                    loss.backward()

                losses[trainingSetIdx] += data.shape[0] * loss.item()
                accuracies[trainingSetIdx] += data.shape[0] * score.item()
                total_elt[trainingSetIdx] += data.shape[0]
                finished = (batchIdx + 1) / len(trainSet[trainingSetIdx]["dataloader"])
                text += " {:s} {:3d}% {:.3f} {:3.2f}%".format(trainSet[trainingSetIdx]["name"], round(100*finished), losses[trainingSetIdx] / total_elt[trainingSetIdx], 100 * accuracies[trainingSetIdx] / total_elt[trainingSetIdx])
            optimizer.step()
            scheduler.step()
            display("\r{:3d} {:.5f}".format(epoch, float(scheduler.get_last_lr()[0])) + text, end = '', force = finished == 1)
        except StopIteration:
            return torch.stack([losses / total_elt, 100 * accuracies / total_elt]).transpose(0,1)

def test(backbone, datasets, criterion):
    backbone.eval()
    for c in criterion:
        c.eval()
    results = []
    for testSetIdx, dataset in enumerate(datasets):
        losses, accuracies, total_elt = 0, 0, 0
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataset["dataloader"]):
                data, target = data.to(args.device), target.to(args.device)
                loss, score = criterion[testSetIdx](backbone(data), target)
                losses += data.shape[0] * loss.item()
                accuracies += data.shape[0] * score.item()
                total_elt += data.shape[0]
        results.append((losses / total_elt, 100 * accuracies / total_elt))
        display(" {:s} {:.3f} {:3.2f}%".format(dataset["name"], losses / total_elt, 100 * accuracies / total_elt), end = '', force = True)
    return torch.tensor(results)

def testFewShot(features, datasets = None):
    results = torch.zeros(len(features), 2)
    for i in range(len(features)):
        accs = []
        feature = features[i]
        Generator = {'metadataset_omniglot':OmniglotGenerator, 'metadataset_imagenet':ImageNetGenerator}.get(datasets[i]['name'].replace('_train', '').replace('_test', '').replace('_validation', '') if datasets != None else datasets, EpisodicGenerator)
        generator = Generator(datasetName=None if datasets is None else datasets[i]["name"], num_elements_per_class= [len(feat['features']) for feat in feature], dataset_path=args.dataset_path)
        for run in range(args.few_shot_runs):
            shots = []
            queries = []
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries)
            shots, queries = generator.get_features_from_indices(feature, episode)
            accs.append(classifiers.evalFewShotRun(shots, queries))
        accs = 100 * torch.tensor(accs)
        low, up = confInterval(accs)
        results[i, 0] = torch.mean(accs).item()
        results[i, 1] = (up - low) / 2
        if datasets is not None:
            display(" {:s} {:.2f}% (±{:.2f}%)".format(datasets[i]["name"], results[i, 0], results[i, 1]), end = '', force = True)
    return results

def process(featuresSet, mean):
    for features in featuresSet:
        if "M" in args.feature_processing:
            for feat in features:
                feat["features"] = feat["features"] - mean.unsqueeze(0)
        if "E" in args.feature_processing:
            for feat in features:
                feat["features"] = feat["features"] / torch.norm(feat["features"], dim = 1, keepdim = True)
    return featuresSet

def computeMean(featuresSet):
    avg = None
    for features in featuresSet:
        if avg == None:
            avg = torch.cat([features[i]["features"] for i in range(len(features))]).mean(dim = 0)
        else:
            avg += torch.cat([features[i]["features"] for i in range(len(features))]).mean(dim = 0)
    return avg / len(featuresSet)

def generateFeatures(backbone, datasets):
    """
    Generate features for all datasets
    Inputs:
        - backbone: torch model
        - datasets: list of datasets to generate features from, each dataset is a dictionnary following the structure in dataloaders.py
    Returns:
        - results:  list of results for each dataset. Format for each dataset is a list of dictionnaries: [{"name_class":str, "features": torch.Tensor} for all classes]
    """
    backbone.eval()
    results = []
    for testSetIdx, dataset in enumerate(datasets):
        allFeatures = [{"name_class": name_class, "features": []} for name_class in dataset["name_classes"]]
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataset["dataloader"]):
                data, target = data.to(args.device), target.to(args.device)
                features = backbone(data).to("cpu")
                for i in range(features.shape[0]):
                    allFeatures[target[i]]["features"].append(features[i])
        results.append([{"name_class": allFeatures[i]["name_class"], "features": torch.stack(allFeatures[i]["features"])} for i in range(len(allFeatures))])
    return results

if args.test_features != "":
    features = [torch.load(args.test_features, map_location=args.device)]
    print(testFewShot(features))
    exit()

allRunTrainStats = None
allRunValidationStats = None
allRunTestStats = None
createCSV(trainSet, validationSet, testSet)
for nRun in range(args.runs):
    print("Preparing backbone... ", end='')
    if args.audio:
        backbone, outputDim = backbones1d.prepareBackbone()
    else:
        backbone, outputDim = backbones.prepareBackbone()
    if args.load_backbone != "":
        backbone.load_state_dict(torch.load(args.load_backbone))
    backbone = backbone.to(args.device)
    numParamsBackbone = torch.tensor([m.numel() for m in backbone.parameters()]).sum().item()
    print(" containing {:,} parameters.".format(numParamsBackbone))

    print("Preparing criterion(s) and classifier(s)... ", end='')
    criterion = [classifiers.prepareCriterion(outputDim, dataset["num_classes"]) for dataset in trainSet]
    numParamsCriterions = 0
    for c in criterion:
        c.to(args.device)
        numParamsCriterions += torch.tensor([m.numel() for m in c.parameters()]).sum().item()
    print(" total is {:,} parameters.".format(numParamsBackbone + numParamsCriterions))

    print("Preparing optimizer... ", end='')
    if not args.freeze_backbone:
        parameters = list(backbone.parameters())
    else:
        parameters = []
    for c in criterion:
        parameters += list(c.parameters())    
    print(" done.")
    print()

    tick = time.time()
    best_val = 1e10 if not args.few_shot else 0
    lr = args.lr

    try:
        nSteps = torch.min(torch.tensor([len(dataset["dataloader"]) for dataset in trainSet])).item()
        if args.dataset_size > 0 and args.dataset_size // args.batch_size < nSteps:
            nSteps = args.dataset_size // args.batch_size
    except:
        nSteps = 0

    for epoch in range(args.epochs):
        if epoch == 0 and not args.cosine:
            optimizer = torch.optim.SGD(parameters, lr = lr, weight_decay = args.wd, momentum = 0.9, nesterov = True) if args.optimizer.lower() == "sgd" else torch.optim.Adam(parameters, lr = lr, weight_decay = args.wd)
            if not args.cosine:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [n * nSteps for n in args.milestones], gamma = args.gamma)
        if args.cosine and (epoch in args.milestones or epoch == 0):
            optimizer = torch.optim.SGD(parameters, lr = lr, weight_decay = args.wd, momentum = 0.9, nesterov = True) if args.optimizer.lower() == "sgd" else torch.optim.Adam(parameters, lr = lr, weight_decay = args.wd)
            if epoch == 0:
                interval = nSteps * args.milestones[0]
            else:
                index = args.milestones.index(epoch)
                interval = nSteps * (args.milestones[index + 1] - args.milestones[index])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = interval)
            lr = lr * args.gamma
        
        continueTest = False
        meanVector = None
        if trainSet != []:
            print(Fore.CYAN, end='')
            trainStats = train(epoch + 1, backbone, criterion, optimizer, scheduler)
            updateCSV(trainStats, epoch = epoch)
            if (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "":
                if epoch >= args.skip_epochs:
                    featuresTrain = generateFeatures(backbone, trainSet)
                    meanVector = computeMean(featuresTrain)
                    featuresTrain = process(featuresTrain, meanVector)
            print(Style.RESET_ALL, end='')
        if validationSet != [] and epoch >= args.skip_epochs:
            print(Fore.GREEN, end = '')
            if args.few_shot or args.save_features_prefix != "":
                featuresValidation = generateFeatures(backbone, validationSet)
                featuresValidation = process(featuresValidation, meanVector)
                tempValidationStats = testFewShot(featuresValidation, validationSet)
            else:
                tempValidationStats = test(backbone, validationSet, criterion)
            updateCSV(tempValidationStats)
            if (tempValidationStats[:,0].mean().item() < best_val and not args.few_shot) or (args.few_shot and tempValidationStats[:,0].mean().item() > best_val):
                validationStats = tempValidationStats
                best_val = validationStats[:,0].mean().item()
                continueTest = True
            print(Style.RESET_ALL, end = '')
        else:
            continueTest = True
        if testSet != [] and epoch >= args.skip_epochs:
            print(Fore.RED, end = '')
            if args.few_shot or args.save_features_prefix != "":
                featuresTest = generateFeatures(backbone, testSet)
                featuresTest = process(featuresTest, meanVector)
                tempTestStats = testFewShot(featuresTest, testSet)
            else:
                tempTestStats = test(backbone, testSet, criterion)
            updateCSV(tempTestStats)
            if continueTest:
                testStats = tempTestStats
            print(Style.RESET_ALL, end = '')
        if continueTest and args.save_backbone != "" and epoch >= args.skip_epochs:
            torch.save(backbone.to("cpu").state_dict(), args.save_backbone)
            backbone.to(args.device)
        if continueTest and args.save_features_prefix != "" and epoch >= args.skip_epochs:
            for i, dataset in enumerate(trainSet):
                torch.save(featuresTrain[i], args.save_features_prefix + dataset["name"] + "_features.pt")
            for i, dataset in enumerate(validationSet):
                torch.save(featuresValidation[i], args.save_features_prefix + dataset["name"] + "_features.pt")
            for i, dataset in enumerate(testSet):
                torch.save(featuresTest[i], args.save_features_prefix + dataset["name"] + "_features.pt")

        scheduler.step()
        print(" " + timeToStr(time.time() - tick))
    if trainSet != []:
        if allRunTrainStats is not None:
            allRunTrainStats = torch.cat([allRunTrainStats, trainStats.unsqueeze(0)])
        else:
            allRunTrainStats = trainStats.unsqueeze(0)
    if validationSet != []:
        if allRunValidationStats is not None:
            allRunValidationStats = torch.cat([allRunValidationStats, validationStats.unsqueeze(0)])
        else:
            allRunValidationStats = validationStats.unsqueeze(0)
    if testSet != []:
        if allRunTestStats is not None:
            allRunTestStats = torch.cat([allRunTestStats, testStats.unsqueeze(0)])
        else:
            allRunTestStats = testStats.unsqueeze(0)

    print()
    print("Run " + str(nRun+1) + "/" + str(args.runs) + " finished")
    for phase, nameSet, stats in [("Train", trainSet, allRunTrainStats), ("Validation", validationSet, allRunValidationStats),  ("Test", testSet, allRunTestStats)]:
        print(phase)
        if nameSet != []:
            for dataset in range(stats.shape[1]):
                print("\tDataset " + nameSet[dataset]["name"])
                for stat in range(stats.shape[2]):
                    low, up = confInterval(stats[:,dataset,stat])
                    print("\t{:.3f} ±{:.3f} (conf. [{:.3f}, {:.3f}])".format(stats[:,dataset,stat].mean().item(), stats[:,dataset,stat].std().item(), low, up), end = '')
                print()
    print()
