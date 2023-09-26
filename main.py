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

if args.wandb!='':
    import wandb

if not args.silent:
    print(" done.")
    
    print()


from datetime import datetime

now = datetime.now()

current_time = now.strftime("%H:%M:%S")
print("Current Time =", current_time)


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
    backbone.train()
    for c in [item for sublist in criterion.values() for item in sublist] :
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
                        dataStep = data['supervised'].clone()
                        loss_proto, score_proto = criterion['prototypical'][trainingSetIdx](backbone, dataStep)
                        loss += args.step_coefficient[step_idx] * loss_proto
                        score += args.step_coefficient[step_idx] * score_proto
                        
                    if 'lr' in step or 'mixup' in step or 'manifold mixup' in step or 'rotations' in step:
                        dataStep = data['supervised'].clone()
                        loss_lr, score = criterion['supervised'][trainingSetIdx](backbone, dataStep, target, lr="lr" in step, rotation="rotations" in step, mixup="mixup" in step, manifold_mixup="manifold mixup" in step)
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
        with torch.no_grad():
            for batchIdx, (data, target) in enumerate(dataset["dataloader"]):
                data = to(data, args.device)
                target = target.to(args.device)
                loss, score = criterion[testSetIdx](backbone, data, target, lr=True)
                losses += data.shape[0] * loss.item()
                accuracies += data.shape[0] * score.item()
                total_elt += data.shape[0]
        results.append((losses / total_elt, 100 * accuracies / total_elt))
        if args.wandb!='':
            wandb.log({ "test_loss_{}".format(dataset["name"]) : losses / total_elt, "test_acc_{}".format(dataset["name"]) : accuracies / total_elt})
        display(" " * (1 + max(0, len(datasets[testSetIdx]["name"]) - 16)) + opener + "{:.2e}  {:6.2f}%".format(losses / total_elt, 100 * accuracies / total_elt) + ender, end = '', force = True)
    return torch.tensor(results)

def repvgg_model_convert(model, save_path=None, do_copy=True):
    if do_copy:
        model = copy.deepcopy(model)
    for module in model.modules():
        if hasattr(module, 'inference_transform'):
            module.inference_transform()
    if save_path is not None:
        torch.save(model.state_dict(), save_path)
    return model

def make_episodes(list_of_episodes, keys):
    # Initialize dictionary with keys pointing to empty lists
    episodes = {key: [] for key in keys}

    # Loop over each episode
    for episode in list_of_episodes:
        # Loop over each key
        for key in keys:
            # Append the corresponding data from the episode to the dictionary
            episodes[key].append(episode[key])

    return episodes


def testFewShot(features, datasets = None):
    results = torch.zeros(len(features), 2)
    for i in range(len(features)):
        accs = []
        feature = features[i]
        Generator = {'metadataset_omniglot':OmniglotGenerator, 'metadataset_imagenet':ImageNetGenerator}.get(datasets[i]['name'].replace('_train', '').replace('_test', '').replace('_validation', '') if datasets != None else datasets, EpisodicGenerator)
        num_elements_per_class= [len(feat['features']) for feat in feature]
        if args.max_elts_per_class != -1:
            print("#### \n UPDATED NUM ELEMENTS PER CLASS \n ####")
            num_elements_per_class = [min(args.max_elts_per_class, n) for n in num_elements_per_class]
            print("#### \n" ,num_elements_per_class,"\n ####")   
            if args.shuffle_features:
                print("#### \n SHUFFLING FEATURES \n ####")
                for feat in feature:
                    indices = np.arange(feat['features'].shape[0])
                    np.random.shuffle(indices)
                    feat['features'] = feat['features'][indices]
        generator = Generator(datasetName=None if datasets is None else datasets[i]["name"], num_elements_per_class= num_elements_per_class, dataset_path=args.dataset_path)
        episodes = []
        for run in range(args.few_shot_runs):
            shots = []
            queries = []
            episode = generator.sample_episode(ways=args.few_shot_ways, n_shots=args.few_shot_shots, n_queries=args.few_shot_queries, unbalanced_queries=args.few_shot_unbalanced_queries, allow_replacement= not args.no_replacement, allow_reset=args.allow_reset, n_retries=args.FSsampling_n_retries)
            
            if episode is None:
                print('task n°{} out of {} failed after {} retries'.format(run,args.few_shot_runs,args.FSsampling_n_retries))
                break
            else:
                shots, queries = generator.get_features_from_indices(feature, episode)
                accs.append(classifiers.evalFewShotRun(shots, queries))
                episodes.append(episode)
        episodes = make_episodes(episodes, episodes[0].keys()) 
        accs = 100 * torch.tensor(accs)
        torch.save({'accs': accs, 'episodes' : episodes }, 'accs_episodes_{}_{}_shots_{}_ways.pt'.format(args.test_dataset, args.few_shot_shots, args.few_shot_ways))
        if args.wandb!='':
            log={'test_performance': accs.mean().item(), 'std_performance' : accs.std().item()}
            wandb.log(log)
        low, up = confInterval(accs)
        results[i, 0] = torch.mean(accs).item()
        results[i, 1] = (up - low) / 2
        if datasets is not None:
            display(" " * (1 + max(0, len(datasets[i]["name"]) - 16)) + opener + "{:6.2f}% (±{:6.2f})".format(results[i, 0], results[i, 1]) + ender, end = '', force = True)
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
    if args.wandb!='':
        wandb.init(reinit = True, project=args.wandbProjectName, 
            entity=args.wandb, 
            config=vars(args),
            dir=args.wandb_dir)
    features = [torch.load(args.test_features, map_location=args.device)]
    if args.folds == -1:
        print(testFewShot(features))
        exit()
    else:
        n_iter = 100
        L_final = []
        for j in range(n_iter):
            L_results = []
            n=len(features[0])
            indices = np.arange(n)
            np.random.shuffle(indices)
            for k in range(args.folds):
                print('fold',k,'iteration', j)
                idx = indices[k*int(n/args.folds):(k+1)*int(n/args.folds)]
                res = testFewShot([[features[0][i] for i in idx]])
                print(idx,res)
                L_results.append(res)
            print('### \n \n \n  ######')
            L_final.append(torch.stack(L_results))
        print('final')
        print(torch.stack(L_final))
        avg = torch.stack(L_final).mean(0).mean(0)[0]
        print('avg',avg)
        avg_updated_w_indep_folds=torch.zeros(2)
        avg_updated_w_indep_folds[0] = avg[0]
        avg_updated_w_indep_folds[1] = avg[1]/args.folds
        print('final' , avg_updated_w_indep_folds)
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
            opener = Fore.CYAN
            if not args.freeze_backbone:
                trainStats = train(epoch + 1, backbone, teacher, criterion, optimizer, scheduler)
                updateCSV(trainStats, epoch = epoch)
            if (args.few_shot and "M" in args.feature_processing) or args.save_features_prefix != "":
                if epoch >= args.skip_epochs:
                    featuresTrain = generateFeatures(backbone, trainSet)
                    meanVector = computeMean(featuresTrain)
                    featuresTrain = process(featuresTrain, meanVector)
            ender = Style.RESET_ALL
        if validationSet != [] and epoch >= args.skip_epochs:
            opener = Fore.GREEN
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
            opener = Fore.RED
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
        if epoch == 0 or (epoch+1) % 20 == 0:
            torch.save(backbone, f"backbone_repvgg2_{epoch}.pth")
            torch.save(criterion["supervised"][0], f"class_repvgg2_{epoch}.pth")
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
                    log['test_CI'] = tempTestStats[:,1].mean().item()
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
