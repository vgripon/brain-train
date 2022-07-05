# Loading main libraries
import torch
import random # for mixup

# Loading other files
print("Loading local files... ", end ='')
from args import args
from utils import *
from dataloaders import trainSet, validationSet, testSet
import classifiers
import backbones
print(" done.")

print()
print(args)
print()

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
                batchIdx, (data, target) = next(iterators[trainingSetIdx])
                data, target = data.to(args.device), target.to(args.device)

                for step in eval(args.steps):
                    dataStep = data.clone()
                    
                    if "mixup" in step or "manifold mixup" in step:
                        perm = torch.randperm(dataStep.shape[0])
                        lbda = random.random()                        

                    if "rotations" in step:
                        bs = data.shape[0] // 4
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

                    if "mixup" not in step and "manifold mixup" not in step:
                        loss, score = criterion[trainingSetIdx](backbone(dataStep), target, yRotations = targetRot if "rotations" in step else None)
                    else:                        
                        features = backbone(dataStep, mixup = "mixup" if "mixup" in step else "manifold mixup", lbda = lbda, perm = perm)
                        loss_1, score_1 = criterion[trainingSetIdx](features, target, yRotations = targetRot if "rotations" in step else None)
                        loss_2, score_2 = criterion[trainingSetIdx](features, target[perm], yRotations = targetRot[perm] if "rotations" in step else None)
                        loss = lbda * loss_1 + (1 - lbda) * loss_2
                        score = lbda * score_1 + (1 - lbda) * score_2

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
        for run in range(args.few_shot_runs):
            feature = features[i]
            shots = []
            queries = []
            choice_classes = torch.randperm(len(feature))
            for j in range(args.few_shot_ways):
                choice_samples = torch.randperm(feature[choice_classes[j]]["features"].shape[0])
                shots.append(feature[choice_classes[j]]["features"][choice_samples[:args.few_shot_shots]])
                queries.append(feature[choice_classes[j]]["features"][choice_samples[args.few_shot_shots:args.few_shot_shots + args.few_shot_queries]])
            accs.append(classifiers.evalFewShotRun(shots, queries))
        accs = 100 * torch.tensor(accs)
        low, up = confInterval(accs)
        results[i, 0] = torch.mean(accs).item()
        results[i, 1] = (up - low) / 2
        if datasets is not None:
            display(" {:s} {:.2f}% (±{:.2f}%)".format(datasets[i]["name"], results[i, 0], results[i, 1]), end = '', force = True)
    return results

def process(features):
    if "M" in args.feature_processing:
        avg = torch.cat([features[i]["features"] for i in range(len(features))]).mean(dim = 0)
        for feat in features:
            feat["features"] = feat["features"] - avg.unsqueeze(0)
    if "E" in args.feature_processing:
        for feat in features:
            feat["features"] = feat["features"] / torch.norm(feat["features"], dim = 1, keepdim = True)
    return features

def generateFeatures(backbone, datasets):
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
        results.append(process([{"name_class": allFeatures[i]["name_class"], "features": torch.stack(allFeatures[i]["features"])} for i in range(len(allFeatures))]))
    return results

if args.test_features != "":
    features = [torch.load(args.test_features)]
    print(testFewShot(features))
    exit()

allRunTrainStats = None
allRunValidationStats = None
allRunTestStats = None
createCSV(trainSet, validationSet, testSet)
for nRun in range(args.runs):
    print("Preparing backbone... ", end='')
    backbone, outputDim = backbones.prepareBackbone()
    if args.load_backbone != "":
        backbone = torch.load(args.load_backbone)
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
    parameters = list(backbone.parameters())
    for c in criterion:
        parameters += list(c.parameters())    
    print(" done.")
    print()

    tick = time.time()
    best_val = 1e10
    lr = args.lr

    nSteps = torch.max(torch.tensor([len(dataset["dataloader"]) for dataset in trainSet])).item()

    for epoch in range(args.epochs):
        if epoch == 0:
            optimizer = torch.optim.SGD(parameters, lr = lr, weight_decay = args.wd, momentum = 0.9, nesterov = True) if args.optimizer.lower() == "sgd" else torch.optim.Adam(parameters, lr = lr, weight_decay = args.weight_decay)
            if not args.cosine:
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [n * nSteps for n in args.milestones], gamma = args.gamma)
        if args.cosine and (epoch in args.milestones or epoch == 0):
            if epoch == 0:
                interval = nSteps * args.milestones[0]
            else:
                index = args.milestones.index(epoch)
                interval = nSteps * (args.milestones[index + 1] - args.milestones[index])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = interval)
            lr = lr * args.gamma
        
        continueTest = False
        if trainSet != []:
            trainStats = train(epoch + 1, backbone, criterion, optimizer, scheduler)
            updateCSV(trainStats, epoch = epoch)
        if validationSet != []:
            if args.few_shot:
                features = generateFeatures(backbone, validationSet)
                validationStats = testFewShot(features, validationSet)
            else:
                validationStats = test(backbone, validationSet, criterion)
            updateCSV(validationStats)
            if args.save_features_prefix != "":
                if not args.few_shot:
                    features = generateFeatures(backbone, validationSet)
                for i, dataset in enumerate(validationSet):
                    torch.save(features[i], args.save_features_prefix + dataset["name"] + "_features.pt")
            if validationStats[:,0].mean().item() < best_val:
                best_val = validationStats[:,0].mean().item()
                continueTest = True
        else:
            continueTest = True
        if testSet != []:
            if args.few_shot:
                features = generateFeatures(backbone, testSet)
                tempTestStats = testFewShot(features, testSet)
            else:
                tempTestStats = test(backbone, testSet, criterion)
            updateCSV(tempTestStats)
            if continueTest:
                testStats = tempTestStats
                if args.save_features_prefix != "":
                    if not args.few_shot:
                        features = generateFeatures(backbone, testSet)
                    for i, dataset in enumerate(testSet):
                        torch.save(features[i], args.save_features_prefix + dataset["name"] + "_features.pt")
        if continueTest and args.save_backbone != "":
            torch.save(backbone.to("cpu").state_dict(), args.save_backbone)
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
    for phase, nameSet, stats in [("Train", trainSet, allRunTrainStats), ("Test", testSet, allRunTestStats)]:
        print(phase)
        for dataset in range(stats.shape[1]):
            print("\tDataset " + nameSet[dataset]["name"])
            for stat in range(stats.shape[2]):
                low, up = confInterval(stats[:,dataset,stat])
                print("\t{:.3f} ±{:.3f} (conf. [{:.3f}, {:.3f}])".format(stats[:,dataset,stat].mean().item(), stats[:,dataset,stat].std().item(), low, up), end = '')
            print()
