# Loading main libraries
import torch
import torch.nn as nn 
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

def to(obj, device):
    if isinstance(obj, list):
        return [to(o, device) for o in obj]
    elif isinstance(obj, dict):
        return {k:to(v, device) for k,v in obj.items()}
    else:
        return obj.to(device)

def train(epoch, backbone, teacher, criterion, optimizer, scheduler):
    if not args.freeze_backbone:
        backbone.train()
    if not  args.freeze_classifier:
        for c in [item for sublist in criterion.values() for item in sublist]:
            c.train()
    iterators = [enumerate(dataset["dataloader"]) for dataset in trainSet]
    losses, accuracies, total_elt = torch.zeros(len(iterators)), torch.zeros(len(iterators)), torch.zeros(len(iterators))
    while True:
        try:
            optimizer.zero_grad()
            text = ""
            batch_idx_list = []
            for trainingSetIdx in range(len(iterators)):
                if args.dataset_size > 0 and total_elt[trainingSetIdx] >= args.dataset_size:
                    raise StopIteration
                batchIdx, (data, target) = next(iterators[trainingSetIdx])
                batch_idx_list.append(batchIdx)
                data = to(data, args.device)
                target = target.to(args.device)
                for step_idx, step in enumerate(eval(args.steps)):
                    loss, score = 0., torch.zeros(1)
                    if 'prototypical' in step: 
                        if isinstance(data, dict):
                            dataStep = data['supervised'].clone()
                        else:
                            dataStep=data
                        loss_proto, score_proto = criterion['prototypical'][trainingSetIdx](backbone, dataStep)
                        loss += args.step_coefficient[step_idx] * loss_proto
                        score += args.step_coefficient[step_idx] * score_proto
                        
                    if 'lr' in step or 'mixup' in step or 'manifold mixup' in step or 'rotations' in step:
                        if isinstance(data, dict):
                            dataStep = data['supervised'].clone()
                        else:
                            dataStep=data
                        if args.save_logits == '':
                            loss_lr, score = criterion['supervised'][trainingSetIdx](backbone, dataStep, target, lr="lr" in step, rotation="rotations" in step, mixup="mixup" in step, manifold_mixup="manifold mixup" in step)
                        else:
                            loss_lr, score , logit = criterion['supervised'][trainingSetIdx](backbone, dataStep, target, lr="lr" in step, rotation="rotations" in step, mixup="mixup" in step, manifold_mixup="manifold mixup" in step)
                        loss += args.step_coefficient[step_idx]*loss_lr

                    if 'dino' in step:
                        dataStep = data['dino']
                        loss_dino = criterion['dino'][trainingSetIdx](backbone, teacher['dino'], dataStep, target, epoch-1)
                        loss += args.step_coefficient[step_idx]*loss_dino

                    if 'simclr' in step:
                        dataStep = data['simclr']
                        loss_simclr = criterion['simclr'][trainingSetIdx](backbone, dataStep, target)
                        loss += args.step_coefficient[step_idx]*loss_simclr

                    if 'simclr_supervised' in step:
                        dataStep = data['simclr_supervised']
                        loss_simclr_supervised = criterion['simclr_supervised'][trainingSetIdx](backbone, dataStep, target)
                        loss += args.step_coefficient[step_idx]*loss_simclr_supervised

                    if 'simsiam' in step:
                        dataStep = data['simsiam']
                        loss_simsiam = criterion['simsiam'][trainingSetIdx](backbone, dataStep)
                        loss += args.step_coefficient[step_idx]*loss_simsiam

                    if 'barlowtwins' in step:
                        dataStep = data['barlowtwins']
                        loss_barlowtwins = criterion['barlowtwins'][trainingSetIdx](backbone, dataStep)
                        loss += args.step_coefficient[step_idx]*loss_barlowtwins
               
                    loss.backward()

                losses[trainingSetIdx] += args.batch_size * loss.item()
                accuracies[trainingSetIdx] += args.batch_size * score.item()
                total_elt[trainingSetIdx] += args.batch_size
                finished = (batchIdx + 1) / len(trainSet[trainingSetIdx]["dataloader"])
                text += " " + opener + "{:3d}% {:.2e} {:6.2f}%".format(round(100*finished), losses[trainingSetIdx] / total_elt[trainingSetIdx], 100 * accuracies[trainingSetIdx] / total_elt[trainingSetIdx]) + ender
                if 21 < 2 + len(trainSet[trainingSetIdx]["name"]):
                    text = " " * (2 + len(trainSet[trainingSetIdx]["name"]) - 21) + text
            optimizer.step()
            if args.wandb!='':
                wandb.log({"epoch":epoch, "train_loss": losses / total_elt})
            display("\r" + Style.RESET_ALL + "{:4d} {:.2e}".format(epoch, float(scheduler.get_last_lr()[0])) + text, end = '', force = (finished == 1))
            scheduler.step()
            # update teachers in case of momentum encoders
            if teacher != {}:
                for trainingSetIdx in range(len(iterators)):
                    for step in eval(args.steps):
                        if 'dino' in step:
                            criterion['dino'][trainingSetIdx].update_teacher(backbone, teacher['dino'], epoch-1, batch_idx_list[trainingSetIdx])   
        except StopIteration:
            return torch.stack([losses / total_elt, 100 * accuracies / total_elt]).transpose(0,1)
    

def test(backbone, datasets, criterion):
    backbone.eval()
    for c in criterion:
        c.eval()
    results = []
    for testSetIdx, dataset in enumerate(datasets):
        losses, accuracies, total_elt = 0, 0, 0
        alloutputs = [{"name_class": name_class, "logits": []} for name_class in dataset["name_classes"]]
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataset["dataloader"]):
                data = to(data, args.device)
                target = target.to(args.device)
                if args.save_logits == '':
                    loss, score = criterion[testSetIdx](backbone, data, target, lr=True)
                else:
                    n_classes_classifier = criterion[testSetIdx].fc.out_features # this is useful if the number of classes of the dataset is larger than the output of the classifier (use case:  get its logit)
                    loss, score, logit = criterion[testSetIdx](backbone, data,target%n_classes_classifier, lr=True)
                    for i in range(logit.shape[0]):
                        alloutputs[target[i]]["logits"].append(logit[i])
                losses += data.shape[0] * loss.item()
                accuracies += data.shape[0] * score.item()
                total_elt += data.shape[0]
                
            if args.save_logits != '':
                for c in range(len(alloutputs)):
                    alloutputs[c]["logits"] = torch.stack(alloutputs[c]["logits"])
                torch.save(alloutputs, args.save_logits)
        results.append((losses / total_elt, 100 * accuracies / total_elt))
        if args.wandb!='':
            wandb.log({ "test_loss_{}".format(dataset["name"]) : losses / total_elt, "test_acc_{}".format(dataset["name"]) : accuracies / total_elt})
        display(" " * (1 + max(0, len(datasets[testSetIdx]["name"]) - 16)) + opener + "{:.2e}  {:6.2f}%".format(losses / total_elt, 100 * accuracies / total_elt) + ender, end = '', force = True)
    return torch.tensor(results)

def testFewShot(features, datasets = None, write_file=False):
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
        write_file(results)
    return results


def write_file(results):
        try:
            result_file = torch.load(args.save_test)
        except:
            result_file = {}
        try:
            result_file[args.test_dataset][args.load_backbone ] =  results
        except:    
            result_file[args.test_dataset] = {args.load_backbone : results}

        torch.save(result_file , args.save_test)

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
                for batchIdx, (data, target) in enumerate(dataset["dataloader"]):
                    if isinstance(data, dict):
                        data = data["supervised"]
                    data, target = to(data, args.device), target.to(args.device)
                    feats = backbone(data).to("cpu")    
                    for i in range(feats.shape[0]):
                        features[target[i]]["features"].append(feats[i])
                for c in range(len(allFeatures)):
                    if augs == 0:
                        allFeatures[c]["features"] = torch.stack(features[c]["features"])/n_aug
                    else:
                        allFeatures[c]["features"] += torch.stack(features[c]["features"])/n_aug

        results.append([{"name_class": allFeatures[i]["name_class"], "features": allFeatures[i]["features"]} for i in range(len(allFeatures))])
    # Get the GPU memory usage
    gpu_memory = torch.cuda.memory_allocated() / (1024**2)
    print('GPU memory usage:', gpu_memory, 'MB')
    return results
def get_optimizer(parameters, name, lr, weight_decay):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay, momentum=0.9, nesterov=True)
    elif name == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f'Optimizer {name} not supported')
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
            config=vars(args),
            dir=args.wandb_dir)
        if not args.freeze_classifier:
            end = str(args.info) +str(args.lr)
        else:
            end = 'baseline' 
        if args.test_dataset != '':
            run_wandb.name = args.test_dataset[12:-5] +'_'+ end
        else:
            run_wandb.name = args.training_dataset[12:-5] +'_'+ end

    if not args.silent:
        print("Preparing backbone... ", end='')
    if args.audio:
        import backbones1d
        backbone, outputDim = backbones1d.prepareBackbone()
    else:
        import backbones
        backbone, outputDim = backbones.prepareBackbone()
    if args.load_backbone != "":
        backbone.load_state_dict(torch.load(args.load_backbone))
    backbone = backbone.to(args.device)
    if not args.silent:
        numParamsBackbone = torch.tensor([m.numel() for m in backbone.parameters()]).sum().item()
        print(" containing {:,} parameters and feature space of dim {:d}.".format(numParamsBackbone, outputDim))

        print("Preparing criterion(s) and classifier(s)... ", end='')
    
    try:
        nSteps = torch.min(torch.tensor([len(dataset["dataloader"]) for dataset in trainSet])).item()
        if args.dataset_size > 0 and math.ceil(args.dataset_size / args.batch_size) < nSteps:
            nSteps = math.ceil(args.dataset_size / args.batch_size)
    except:
        nSteps = 0
    
    criterion = {}
    teacher = {}
    all_steps = [item for sublist in eval(args.steps) for item in sublist]
    if 'lr' in all_steps or 'mixup' in all_steps or 'manifold mixup' in all_steps or 'rotations' in all_steps:
        criterion['supervised'] = [classifiers.prepareCriterion(outputDim, dataset["num_classes"]) for dataset in trainSet]
        if args.task_queries and not args.few_shot: #useful to measure the support set accuracy
            criterion['supervised'] = [classifiers.prepareCriterion(outputDim, dataset["num_classes"]) for dataset in testSet]
    if args.episodic and 'prototypical' in all_steps:
        criterion['prototypical'] = [classifiers.ProtoNet() for dataset in trainSet]
    if 'dino' in all_steps:
        from selfsupervised.dino import DINO
        criterion['dino'] = [DINO(in_dim=outputDim, epochs=args.epochs, nSteps=nSteps) for _ in trainSet]
        teacher['dino'] = backbones.prepareBackbone()[0].to(args.device) # Same backbone but with a different init
         
        for p in teacher['dino'].parameters(): # Freeze teacher + teacher head
            p.requires_grad = False
         
        for crit in criterion['dino']:
            for p in crit.teacher_head.parameters():
                p.requires_grad = False
  
    if 'simclr' in all_steps:
        from selfsupervised.simclr import SIMCLR
        criterion['simclr'] = [SIMCLR(in_dim=outputDim, supervised=False) for _ in trainSet]
    if 'simclr_supervised' in all_steps:
        from selfsupervised.simclr import SIMCLR
        criterion['simclr_supervised'] = [SIMCLR(in_dim=outputDim, supervised=True) for _ in trainSet]
    if 'simsiam' in all_steps:
        from selfsupervised.simsiam import SIMSIAM
        criterion['simsiam'] = [SIMSIAM(in_dim=outputDim) for _ in trainSet]
    if 'barlowtwins' in all_steps:
        from selfsupervised.barlowtwins import BARLOWTWINS
        criterion['barlowtwins'] = [BARLOWTWINS(in_dim=outputDim) for _ in trainSet]
        
    numParamsCriterions = 0
    for c in [item for sublist in criterion.values() for item in sublist] :
        c.to(args.device)
        numParamsCriterions += torch.tensor([m.numel() for m in c.parameters()]).sum().item()
    if not args.silent:
        print(" total is {:,} parameters.".format(numParamsBackbone + numParamsCriterions))

        print("Preparing optimizer... ", end='')
    if not args.freeze_backbone:
        parameters = list(backbone.parameters())
    else:
        parameters = []
    for c in [item for sublist in criterion.values() for item in sublist] :
        parameters += list(c.parameters())
        if args.load_classifier!= '':
            c.load_state_dict(torch.load(args.load_classifier))
    if not args.silent:
        print(" done.")
        print()

    tick = time.time()
    best_val = 1e10 if not args.few_shot else 0
    lr = args.lr
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
        
        if epoch == 0 and args.warmup_epochs>0:
            optimizer = get_optimizer(parameters, args.optimizer.lower(), lr=lr, weight_decay=args.wd)
            scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1/(args.warmup_epochs+1), end_factor=1, total_iters=args.warmup_epochs*nSteps, last_epoch=-1) # warmup scheduler (linear)
        if (epoch == args.warmup_epochs or (epoch in args.milestones)) and len(parameters)>0:
            if args.scheduler == "multistep" and epoch == args.warmup_epochs:
                optimizer = get_optimizer(parameters, args.optimizer.lower(), lr=lr, weight_decay=args.wd)
                scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer = optimizer, milestones = [(n-args.warmup_epochs) * nSteps for n in args.milestones], gamma = args.gamma)
            if args.scheduler != "multistep":
                optimizer = get_optimizer(parameters, args.optimizer.lower(), lr=lr, weight_decay=args.wd)
                if epoch == args.warmup_epochs:
                    interval = nSteps * (args.milestones[0]-args.warmup_epochs-1)
                else:
                    index = args.milestones.index(epoch)
                    interval = nSteps * (args.milestones[index + 1] - args.milestones[index]-1)
                if args.scheduler == "cosine":                
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer, T_max = interval, eta_min = lr * args.end_lr_factor)
                elif args.scheduler == "linear":
                    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer = optimizer, start_factor = 1, end_factor = args.end_lr_factor, total_iters = interval, last_epoch=-1)
                else:
                    raise ValueError(f"Unknown scheduler {args.scheduler}")
                lr = lr * args.gamma

        continueTest = False
        meanVector = None
        trainStats = None
        if trainSet != []:
            #opener = Fore.CYAN
            if not args.freeze_backbone or args.force_train:
                trainStats = train(epoch + 1, backbone, teacher, criterion, optimizer, scheduler)
                # Get the GPU memory usage
                gpu_memory = torch.cuda.memory_allocated() / (1024**2)
                print('GPU memory usage:', gpu_memory, 'MB') 
                updateCSV(trainStats, epoch = epoch)
            if args.save_classifier:
                for i,c in enumerate([item for sublist in criterion.values() for item in sublist]):
                    torch.save(c.cpu().state_dict(), args.save_classifier)
                    c.to(args.device)
            if (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "":
                if epoch >= args.skip_epochs:
                    featuresTrain = generateFeatures(backbone, trainSet)
                    meanVector = computeMean(featuresTrain)
                    featuresTrain = process(featuresTrain, meanVector)
            ender = Style.RESET_ALL
        if validationSet != [] and epoch >= args.skip_epochs:
            #opener = Fore.GREEN
            if args.few_shot or args.save_features_prefix != "":
                featuresValidation = generateFeatures(backbone, validationSet)
                featuresValidation = process(featuresValidation, meanVector)
                tempValidationStats = testFewShot(featuresValidation, validationSet)
            else:
                tempValidationStats = test(backbone, validationSet, criterion['supervised'])
            updateCSV(tempValidationStats)
            if (tempValidationStats[:,0].mean().item() < best_val and not args.few_shot) or (args.few_shot and tempValidationStats[:,0].mean().item() > best_val):
                validationStats = tempValidationStats
                best_val = validationStats[:,0].mean().item()
                continueTest = True
            ender = Style.RESET_ALL
        else:
            continueTest = True
        if testSet != [] and epoch >= args.skip_epochs:
            #opener = Fore.RED
            if args.few_shot or args.save_features_prefix != "":
                #print('Generating Test Features')
                featuresTest = generateFeatures(backbone, testSet)
                featuresTest = process(featuresTest, meanVector)
                tempTestStats = testFewShot(featuresTest, testSet)
            else:
                tempTestStats = test(backbone, testSet, criterion['supervised'])
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
    if args.save_stats!='':
        create_table()
    for phase, nameSet, stats in [("Train", trainSet, allRunTrainStats), ("Validation", validationSet, allRunValidationStats),  ("Test", testSet, allRunTestStats)]:
        print(phase)
        if nameSet != []:
            if stats is not None:
                print(stats.shape)
                for dataset in range(stats.shape[1]):
                    print("\tDataset " + nameSet[dataset]["name"])
                    for stat in range(stats.shape[2]):
                        low, up = confInterval(stats[:,dataset,stat])
                        print("\t{:.3f} ±{:.3f} (conf. [{:.3f}, {:.3f}])".format(stats[:,dataset,stat].mean().item(), stats[:,dataset,stat].std().item(), low, up), end = '')
                    if args.save_stats!='' and phase=='Test':
                        key = args.index_subset
                        insert_data(key, stats[:,dataset,stat].mean().item())
                    print()
    print()
    if args.wandb!='':
        run_wandb.finish()
