# Loading main libraries
import torch
import random # for mixup
import numpy as np # for manifold mixup
import math
from colorama import Fore, Back, Style

# Loading other files
from args import args
if not args.silent:
    print("Loading local files... ", end ='')
from utils import *
from dataloaders import trainSet, validationSet, testSet
import classifiers
import backbones
import backbones1d
from few_shot_evaluation import EpisodicGenerator, ImageNetGenerator, OmniglotGenerator
from tqdm import tqdm
if args.wandb!='':
    import wandb

if not args.silent:
    print(" done.")
    
    print()

print(args)
print()

#for pretty printing
opener = ""
ender = ""

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
                if args.dataset_size > 0 and total_elt[trainingSetIdx] >= args.dataset_size:
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
                text += " " + opener + "{:3d}% {:.2e} {:6.2f}%".format(round(100*finished), losses[trainingSetIdx] / total_elt[trainingSetIdx], 100 * accuracies[trainingSetIdx] / total_elt[trainingSetIdx]) + ender
                if 21 < 2 + len(trainSet[trainingSetIdx]["name"]):
                    text = " " * (2 + len(trainSet[trainingSetIdx]["name"]) - 21) + text
            optimizer.step()
            scheduler.step()
            if args.wandb!='':
                wandb.log({"epoch":epoch, "train_loss": losses / total_elt})
            display("\r" + Style.RESET_ALL + "{:4d} {:.2e}".format(epoch, float(scheduler.get_last_lr()[0])) + text, end = '', force = (finished == 1))
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
        if args.wandb!='':
            wandb.log({ "test_loss_{}".format(dataset["name"]) : losses / total_elt, "test_acc_{}".format(dataset["name"]) : accuracies / total_elt})
        display(" " * (1 + max(0, len(datasets[testSetIdx]["name"]) - 16)) + opener + "{:.2e}  {:6.2f}%".format(losses / total_elt, 100 * accuracies / total_elt) + ender, end = '', force = True)
    return torch.tensor(results)

def testFewShot(features, datasets = None, write_file=True):
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
            display(" " * (1 + max(0, len(datasets[i]["name"]) - 16)) + opener + "{:6.2f}% (±{:6.2f})".format(results[i, 0], results[i, 1]) + ender, end = '', force = True)
    if write_file:
        result_file = torch.load('results_dic.pt')
        try:
            result_file[args.test_dataset][args.load_backbone ] =  results
        except:    
            result_file[args.test_dataset] = {args.load_backbone : results}

        torch.save(result_file , 'results_dic.pt')
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

def generateFeatures(backbone, datasets, sample_aug=args.sample_aug):
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
        n_aug = 1 if 'train' in dataset['name'] else sample_aug
        allFeatures = [{"name_class": name_class, "features": []} for name_class in dataset["name_classes"]]
        with torch.no_grad():
            for augs in range(n_aug):
                features = [{"name_class": name_class, "features": []} for name_class in dataset["name_classes"]]
                for batchIdx, (data, target) in tqdm(enumerate(dataset["dataloader"])):
                    data, target = data.to(args.device), target.to(args.device)
                    feats = backbone(data).to("cpu")
                    for i in range(feats.shape[0]):
                        features[target[i]]["features"].append(feats[i])
                for c in range(len(allFeatures)):
                    if augs == 0:
                        allFeatures[c]["features"] = torch.stack(features[c]["features"])/n_aug
                    else:
                        allFeatures[c]["features"] += torch.stack(features[c]["features"])/n_aug

        results.append([{"name_class": allFeatures[i]["name_class"], "features": allFeatures[i]["features"]} for i in range(len(allFeatures))])
    return results

if args.test_features != "":
    features = [torch.load(args.test_features, map_location=args.device)]
    print(testFewShot(features, write_file=True))
    exit()

allRunTrainStats = None
allRunValidationStats = None
allRunTestStats = None
createCSV(trainSet, validationSet, testSet)
for nRun in range(args.runs):
    if args.wandb!='':
        tag = (args.dataset != '')*[args.dataset] + (args.dataset == '')*['cross-domain'] + ['run_'+str(nRun)] * (args.runs != 1)
        run_wandb = wandb.init(reinit = True, project=args.wandbProjectName, 
            entity=args.wandb, 
            tags=tag, 
            config=vars(args))
    if not args.silent:
        print("Preparing backbone... ", end='')
    if args.audio:
        backbone, outputDim = backbones1d.prepareBackbone()
    else:
        backbone, outputDim = backbones.prepareBackbone()
    if args.load_backbone != "":
        backbone.load_state_dict(torch.load(args.load_backbone))
    backbone = backbone.to(args.device)
    if not args.silent:
        numParamsBackbone = torch.tensor([m.numel() for m in backbone.parameters()]).sum().item()
        print(" containing {:,} parameters.".format(numParamsBackbone))

        print("Preparing criterion(s) and classifier(s)... ", end='')
    criterion = [classifiers.prepareCriterion(outputDim, dataset["num_classes"]) for dataset in trainSet]
    numParamsCriterions = 0
    for c in criterion:
        c.to(args.device)
        numParamsCriterions += torch.tensor([m.numel() for m in c.parameters()]).sum().item()
    if not args.silent:
        print(" total is {:,} parameters.".format(numParamsBackbone + numParamsCriterions))

        print("Preparing optimizer... ", end='')
    if not args.freeze_backbone:
        parameters = list(backbone.parameters())
    else:
        parameters = []
    for c in criterion:
        parameters += list(c.parameters())
    if not args.silent:
        print(" done.")
        print()

    tick = time.time()
    best_val = 1e10 if not args.few_shot else 0
    lr = args.lr

    try:
        nSteps = torch.min(torch.tensor([len(dataset["dataloader"]) for dataset in trainSet])).item()
        if args.dataset_size > 0 and math.ceil(args.dataset_size / args.batch_size) < nSteps:
            nSteps = math.ceil(args.dataset_size / args.batch_size)
    except:
        nSteps = 0

    for epoch in range(args.epochs):
        if (epoch % 30 == 0 and not args.silent) or epoch == 0 or epoch == args.skip_epochs:
            if epoch > 0 and args.silent:
                print()
            print(" ep.       lr ".format(), end='')
            for dataset in trainSet:
                print(Back.CYAN + " {:>19s} ".format(dataset["name"]) + Style.RESET_ALL, end='')
            if epoch >= args.skip_epochs:
                for dataset in validationSet:
                    print(Back.GREEN + " {:>16s} ".format(dataset["name"]) + Style.RESET_ALL, end='')
                for dataset in testSet:
                    print(Back.RED + " {:>16s} ".format(dataset["name"]) + Style.RESET_ALL, end='')
            print()
        if epoch == 0 and not args.cosine and len(parameters)>0:
            optimizer = torch.optim.SGD(parameters, lr = lr, weight_decay = args.wd, momentum = 0.9, nesterov = True) if args.optimizer.lower() == "sgd" else torch.optim.Adam(parameters, lr = lr, weight_decay = args.wd)
            if not args.cosine:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [n * nSteps for n in args.milestones], gamma = args.gamma)
        if args.cosine and (epoch in args.milestones or epoch == 0) and len(parameters)>0:
            optimizer = torch.optim.SGD(parameters, lr = lr, weight_decay = args.wd, momentum = 0.9, nesterov = True) if args.optimizer.lower() == "sgd" else torch.optim.Adam(parameters, lr = lr, weight_decay = args.wd)
            if epoch == 0:
                interval = nSteps * args.milestones[0]
            else:
                index = args.milestones.index(epoch)
                interval = nSteps * (args.milestones[index + 1] - args.milestones[index])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = interval, eta_min = lr * 1e-3)
            lr = lr * args.gamma
        
        continueTest = False
        meanVector = None
        trainStats = None
        if trainSet != []:
            opener = Fore.CYAN
            if not args.freeze_backbone:
                trainStats = train(epoch + 1, backbone, criterion, optimizer, scheduler)
                updateCSV(trainStats, epoch = epoch)
            if (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "":
                if epoch >= args.skip_epochs:
                    #print('Generating Train Features')
                    featuresTrain = generateFeatures(backbone, trainSet)
                    meanVector = computeMean(featuresTrain)
                    featuresTrain = process(featuresTrain, meanVector)
            ender = Style.RESET_ALL
        if validationSet != [] and epoch >= args.skip_epochs:
            opener = Fore.GREEN
            if args.few_shot or args.save_features_prefix != "":
                #print('Generating Validation Features')
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
            ender = Style.RESET_ALL
        else:
            continueTest = True
        if testSet != [] and epoch >= args.skip_epochs:
            opener = Fore.RED
            if args.few_shot or args.save_features_prefix != "":
                #print('Generating Test Features')
                featuresTest = generateFeatures(backbone, testSet)
                featuresTest = process(featuresTest, meanVector)
                tempTestStats = testFewShot(featuresTest, testSet)
            else:
                tempTestStats = test(backbone, testSet, criterion)
            updateCSV(tempTestStats)
            if continueTest:
                testStats = tempTestStats
            ender = Style.RESET_ALL
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
        if args.wandb!='':
            log = {'epoch' : epoch}
            if epoch >= args.skip_epochs:
                if validationSet!=[]:
                    log['validation'] = tempValidationStats[:,0].mean().item()
                    log['best_val'] = best_val
                if testSet!=[]:
                    log['test'] = tempTestStats[:,0].mean().item()
            wandb.log(log)
        print(Style.RESET_ALL + " " + timeToStr(time.time() - tick), end = '' if args.silent else '\n')
    if trainSet != [] and trainStats is not None:
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
            if stats is not None:
                for dataset in range(stats.shape[1]):
                    print("\tDataset " + nameSet[dataset]["name"])
                    for stat in range(stats.shape[2]):
                        low, up = confInterval(stats[:,dataset,stat])
                        print("\t{:.3f} ±{:.3f} (conf. [{:.3f}, {:.3f}])".format(stats[:,dataset,stat].mean().item(), stats[:,dataset,stat].std().item(), low, up), end = '')
                    print()
    print()
    if args.wandb!='':
        run_wandb.finish()
