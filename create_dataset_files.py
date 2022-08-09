from args import args
import torchvision
import json
import os
from torchvision import transforms, datasets
import torch 
import numpy as np
from PIL import Image
import scipy.io
import tqdm
from collections import defaultdict
import pandas as pd


available_datasets = os.listdir(args.dataset_path)
print('Available datasets:', available_datasets)

# Read Graph for imagenet names and classes
with open(os.path.join('datasets', 'ilsvrc_2012_dataset_spec.json'), 'r') as file:
    graph = json.load(file)

imagenet_class_names = {}
for subset in graph['split_subgraphs'].values():
    for entity in subset:
        if len(entity['children_ids']) == 0:
            imagenet_class_names[entity['wn_id']] = entity['words']

all_results = defaultdict(dict)

### generate data for miniimagenet
if 'miniimagenetimages' in available_datasets:
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
                        result["name_classes"].append(imagenet_class_names[classIdx])
                        nClass += 1
        result["num_classes"] = nClass
        result["num_elements_per_class"] = [600]*nClass
        
        all_results["miniimagenet_" + dataset] = result
        print("Done for miniimagenet_" + dataset + " with " + str(nClass) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

### generate data for tieredimagenet
if 'tieredimagenet' in available_datasets:
    for dataset,folderName in [("train","train"),("validation","val"),("test","test")]:
        directories = os.listdir(args.dataset_path + "tieredimagenet/" + folderName)
        result = {"data":[], "targets":[], "name":"tieredimagenet_" + dataset, "num_classes":0, "name_classes":[]}
        for i,classIdx in enumerate(directories):
            num_elements_per_class = 0
            for fileName in os.listdir(args.dataset_path + "tieredimagenet/" + folderName + "/" + classIdx):
                result["data"].append("tieredimagenet/" + folderName + "/" + classIdx + "/" + fileName)
                result["targets"].append(i)
                num_elements_per_class += 1
            result["num_elements_per_class"] = num_elements_per_class
            result["name_classes"].append(imagenet_class_names[classIdx])
            
        result["num_classes"] = i + 1
        all_results["tieredimagenet_" + dataset] = result
        print("Done for tieredimagenet_" + dataset + " with " + str(i+1) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

### generate data for cifarfs
if 'cifar_fs' in available_datasets:
    for dataset,folderName in [("train","meta-train"),("validation","meta-val"),("test","meta-test")]:
        directories = os.listdir(args.dataset_path + "cifar_fs/" + folderName)
        result = {"data":[], "targets":[], "name":"cifarfs_" + dataset, "num_classes":0, "name_classes":[]}
        for i,classIdx in enumerate(directories):
            num_elements_per_class = 0
            for fileName in os.listdir(args.dataset_path + "cifar_fs/" + folderName + "/" + classIdx):
                result["data"].append("cifar_fs/" + folderName + "/" + classIdx + "/" + fileName)
                result["targets"].append(i)
            num_elements_per_class += 1
            result["num_elements_per_class"] = num_elements_per_class
            result["name_classes"].append(classIdx)

        result["num_classes"] = i + 1
        all_results["cifarfs_" + dataset] = result
        print("Done for cifarfs_" + dataset + " with " + str(i+1) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")

## generate data for mnist
try:
    for dataset in ['train', 'test']:
        result = {"data":[], "targets":[], "name":"mnist_" + dataset, "num_classes":0, "name_classes":[], "num_elements_per_class": []}
        pytorchDataset = datasets.MNIST(args.dataset_path, train = dataset != "test", download = not ('MNIST' in available_datasets)) # download if not existing
        targets = pytorchDataset.targets
        for c in range(targets.max()):
            result["num_elements_per_class"].append(len(torch.where(targets==c)[0]))
        result["num_classes"] = len(result["num_elements_per_class"])+1
        all_results['mnist_'+ dataset] = result
        print("Done for mnist_" + dataset + " with " + str(result["num_classes"]) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")
except:
    pass
### generate data for imagenet metadatasets
try:
    if 'imagenet' in available_datasets:
        # Parse Graph 
        class_folders = {k:[p['wn_id'] for p in graph['split_subgraphs'][k] if len(p['children_ids']) == 0] for k in ['TRAIN', 'TEST', 'VALID']} # Existing classes    
        # Get duplicates from other datasets which should be removed from ImageNet
        duplicates = []
        duplicate_files =  ['ImageNet_CUBirds_duplicates.txt', 'ImageNet_Caltech101_duplicates.txt', 'ImageNet_Caltech256_duplicates.txt']
        for file in duplicate_files:
            with open(os.path.join('datasets', 'metadatasets', 'ilsvrc_2012', file), 'r') as f:
                duplicates_tmp = f.read().split('\n')
            duplicates += [p.split('#')[0].replace(' ','') for p in duplicates_tmp if len(p)>0 and p[0] not in ['#']] # parse the duplicates files
        # check which file exists:
        path = os.path.join('imagenet', 'ILSVRC2012_img_train' if os.path.exists(os.path.join(args.dataset_path, 'imagenet', 'ILSVRC2012_img_train')) else 'train')
        for dataset, folderName in [('train', 'TRAIN'), ('test', 'TEST'), ('validation','VALID')]:
            result = {"data":[], "targets":[], "name":"metadataset_imagenet_" + dataset, "num_classes":0, "name_classes":[], "num_elements_per_class":[], "classIdx":{}}
            for i, classIdx in enumerate(class_folders[folderName]):
                num_elements_per_class = 0
                for fileName in os.listdir(os.path.join(args.dataset_path, path, classIdx)):
                    if os.path.join(classIdx, fileName) not in duplicates:
                        result["data"].append(os.path.join(path, classIdx, fileName))
                        result["targets"].append(i)
                        num_elements_per_class +=1
                result["name_classes"].append(imagenet_class_names[classIdx])
                result["classIdx"][classIdx] = i
                result["num_elements_per_class"].append(num_elements_per_class)

            result["num_classes"] = i + 1   
            all_results["metadataset_imagenet_" + dataset] = result
            print("Done for metadataset_imagenet_" + dataset + " with " + str(i+1) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")
except:
    pass
def split_fn(json_path):
    with open(json_path) as jsonFile:
        split = json.load(jsonFile)
        jsonFile.close()
    return split

def get_data(jsonpath, image_dir, datasetName):
    split = split_fn(jsonpath)
    data ,num_classes, num_elts = {},{},{}
    split = {"validation" if k == 'valid' else k:v for k,v in split.items()}
    for index_subset, subset in enumerate(split.keys()):
        data[subset] = {'data': [], 'targets' : [] , 'name': 'metadataset_' + datasetName + '_' +subset,'name_classes' : [],'num_elements_per_class':[], 'num_classes': []}
        l_classes = split[subset]
        data[subset]['num_classes'] = len(l_classes)
        for index_class, cl in enumerate(l_classes):
            
            cl_path = args.dataset_path + image_dir + cl
            images = sorted(os.listdir(cl_path)) #Careful here you might mix the order (not sure that sorted is good enough)
            data[subset]['num_elements_per_class'].append(len(images))
            for index_image , im in enumerate(images):
                data[subset]['data'].append( image_dir + cl +'/' + im)
                data[subset]['targets'].append(index_class)
            
            data[subset]['name_classes'].append(cl)
    return data

def read_info_fungi():
    info_sub={}
    for subset in ['train', 'val']:
        json_path = args.dataset_path + 'fungi/'+subset+'.json'
        info_sub[subset] = split_fn(json_path)
    L_id,L_fl ,L_ida,L_ca,L_imgid,L_idc,L_name,L_sup = [],[],[],[],[],[],[],[]
    for subset in ['train', 'val']:
        for x in info_sub[subset]['images']:
            L_id.append(x['id'])
            L_fl.append(x['file_name'])
        for x in info_sub[subset]['annotations']:
            L_ida.append(x['id'])
            L_ca.append(x['category_id'])
            L_imgid.append(x['image_id'])
        for x in info_sub[subset]['categories']:
            L_idc.append(x['id'])
            L_name.append(x['name'])
            L_sup.append(x['supercategory'])
    np_ca = np.array(L_ca)
    np_fl=np.array(L_fl)
    np_idc = np.array(L_idc)
    np_sup = np.array(L_sup)
    return np_ca, np_fl, np_idc, np_sup

def get_data_fungi():
    split = split_fn('datasets/metadatasets/fungi/fungi_splits.json' )
    np_ca, np_fl, np_idc, np_sup = read_info_fungi()
    data = {}
    split = {"validation" if k == 'valid' else k:v for k,v in split.items()}
    for index_subset, subset in enumerate(split.keys()):
        data[subset] = {'data': [], 'targets' : [] , 'name': 'metadataset_fungi_'+subset,'name_classes' : [],'num_elements_per_class':[], 'num_classes': []}
        l_classes = split[subset]
        data[subset]['num_classes'] = len(l_classes)
        for index_class, cl in enumerate(l_classes):
            clx = int(split['train'][index_class][:4])
            idx = np.where(np_ca==clx)[0]
            data[subset]['num_elements_per_class'].append(idx.shape[0])
            for index_image , im in enumerate(np_fl[idx]):
                data[subset]['data'].append('/fungi/'+im)
                data[subset]['targets'].append(index_class)
            
            data[subset]['name_classes'].append(cl)
        
    return data

def get_images_class_aircraft():
    with open('datasets/metadatasets/aircraft/images_variant.txt') as f:
        lines = f.readlines()
    print(len(lines))
    couples = [x.split(' ', maxsplit=1) for x in lines]
    images = [x[0] for x in couples]
    classes = [x[1][:-1] for x in couples]
    dico_class = {}
    dico_class = defaultdict(list)
    for i ,x in enumerate(images):
        cl = classes[i]
        dico_class[cl].append(x)
    return dico_class

def get_data_aircraft():
    split = split_fn('datasets/metadatasets/aircraft/aircraft_splits.json')
    dico_class = get_images_class_aircraft()
    data  = {}
    split = {"validation" if k == 'valid' else k:v for k,v in split.items()}
    for index_subset, subset in enumerate(split.keys()):
        data[subset] = {'data': [], 'targets' : [] , 'name': 'metadataset_aircraft_'+subset,'name_classes' : [],'num_elements_per_class':[], 'num_classes': []}
        l_classes = split[subset]
        data[subset]['num_classes'] = len(l_classes)
        for index_class, cl in enumerate(l_classes):
            images = dico_class[cl]
            if images != []:
                data[subset]['num_elements_per_class'].append(len(images))
                for index_image , im in enumerate(images):
                    data[subset]['data'].append('/fgvc-aircraft-2013b/data/images_cropped/'+im+'.jpg')
                    data[subset]['targets'].append(index_class)
                data[subset]['name_classes'].append(cl)
            else:
                print(cl, 'not found')
    return data

##### generate data for CUB and DTD
if 'CUB_200_2011' in available_datasets:
    results_cub = get_data("./datasets/metadatasets/cub/cu_birds_splits.json", "CUB_200_2011/images/", 'cub')
if 'dtd' in available_datasets:
    results_dtd  = get_data('./datasets/metadatasets/dtd/dtd_splits.json', 'dtd/images/', 'dtd')
if 'fungi' in available_datasets:
    results_fungi  = get_data_fungi()
if 'fgvc-aircraft-2013b' in available_datasets:
    results_aircraft  = get_data_aircraft()
if 'mscoco' in available_datasets:
    with open(args.dataset_path + 'mscoco/cropped_mscoco.json') as jsonFile:   #this file is obtained by running datasets/metadatasets/mscoco/RUN_ONLY_ONCE.sh (read instructions carefully)
        results_mscoco  = json.load(jsonFile)
        jsonFile.close()
    
for dataset in ['train', 'test', 'validation']:
    if 'CUB_200_2011' in available_datasets:
        all_results["metadataset_cub_" + dataset] = results_cub[dataset]
        print("Done for metadataset_cub_" + dataset + " with " + str(results_cub[dataset]['num_classes']) + " classes and " + str(len(results_cub[dataset]["data"])) + " samples (" + str(len(results_cub[dataset]["targets"])) + ")")
    if 'dtd' in available_datasets:
        all_results["metadataset_dtd_" + dataset] = results_dtd[dataset]
        print("Done for metadataset_dtd_" + dataset + " with " + str(results_dtd[dataset]['num_classes']) + " classes and " + str(len(results_dtd[dataset]["data"])) + " samples (" + str(len(results_dtd[dataset]["targets"])) + ")")
    if 'fungi' in available_datasets:
        all_results["metadataset_fungi_" + dataset] = results_fungi[dataset]
        print("Done for metadataset_fungi_" + dataset + " with " + str(results_fungi[dataset]['num_classes']) + " classes and " + str(len(results_fungi[dataset]["data"])) + " samples (" + str(len(results_fungi[dataset]["targets"])) + ")")
    if 'fgvc-aircraft-2013b' in available_datasets:
        all_results["metadataset_aircraft_" + dataset] = results_aircraft[dataset]
        print("Done for metadataset_aircraft_" + dataset + " with " + str(results_aircraft[dataset]['num_classes']) + " classes and " + str(len(results_aircraft[dataset]["data"])) + " samples (" + str(len(results_aircraft[dataset]["targets"])) + ")")
    if 'mscoco' in available_datasets and dataset != 'train':
        all_results["metadataset_mscoco_" + dataset] = results_mscoco[dataset]
        print("Done for metadataset_mscoco_" + dataset + " with " + str(results_mscoco[dataset]['num_classes']) + " classes and " + str(len(results_mscoco[dataset]["data"])) + " samples (" + str(len(results_mscoco[dataset]["targets"])) + ")")



# generate data for omniglot
if 'omniglot' in available_datasets:
    with open("./datasets/metadatasets/omniglot/"+"omniglot_dataset_spec.json") as jsonFile:
            split = json.load(jsonFile)
            jsonFile.close()

    superclass_count = 0
    for splitName,dataset in [("TRAIN","train"),("VALID","validation"),("TEST","test")]:
        class_count = 0
        result = {"data":[], "targets":[], "name":"metadataset_omniglot_" + dataset, "num_classes":0, "name_classes":[], "num_superclasses":0, "classes_per_superclass":defaultdict(list), "num_elements_per_class": []}
        for superclass_id in range(superclass_count,superclass_count+split["superclasses_per_split"][splitName]):
            result['num_superclasses'] = split["superclasses_per_split"][splitName]
            superclass_name = split["superclass_names"][str(superclass_id)]

            superclass_path = "omniglot/images_background/"+superclass_name+'/' 
            if dataset=='test':
                superclass_path = "omniglot/images_evaluation/"+superclass_name+'/'
            for class_name in os.listdir(args.dataset_path+'/'+superclass_path):
                result['classes_per_superclass'][superclass_id-superclass_count].append(class_count)
                class_path = superclass_path+class_name+'/'
                result['num_classes'] +=1
                result['name_classes'].append(superclass_name+'-'+class_name)
                result['num_elements_per_class'].append(len(os.listdir(args.dataset_path+'/'+class_path)))

                for filename in os.listdir(args.dataset_path+'/'+class_path):
                    result['data'].append(class_path+filename)
                    result['targets'].append(class_count)
                class_count += 1
        superclass_count += split["superclasses_per_split"][splitName]
        all_results["metadataset_omniglot_" + dataset] = result
        print("Done for metadataset_omniglot_ " + dataset + " with " + str(result['num_classes']) + " classes ")


## generate data for vgg_flower
if 'vgg_flower' in available_datasets:
    labels = scipy.io.loadmat(args.dataset_path+'vgg_flower/'+'imagelabels.mat')['labels'][0]
    with open('./datasets/metadatasets/vgg_flower/'+"vgg_flower_splits.json") as jsonFile:
            split = json.load(jsonFile)
            jsonFile.close()
    split_rev = defaultdict(str)
    for dataset,splitName in [("train","train"),("validation","valid"),("test","test")]:
        all_results["metadataset_vgg_flower_"+dataset] = {"data":[], "targets":[], "name":"metadataset_vgg_flower_" + dataset, "num_classes":0, "name_classes":[], "dataset_targets":defaultdict(int), "num_elements_per_class":[]}
        for class_name in split[splitName]:
            split_rev[int(class_name[:3])] = dataset
            all_results["metadataset_vgg_flower_"+dataset]['dataset_targets'][int(class_name[:3])] = all_results["metadataset_vgg_flower_"+dataset]['num_classes']
            all_results["metadataset_vgg_flower_"+dataset]['name_classes'].append(class_name)
            all_results["metadataset_vgg_flower_"+dataset]['num_classes']+=1
        #print("Initialized for Vgg Flower " + dataset + " with " + str(all_results["metadataset_vgg_flower_"+dataset]['num_classes']) + " classes" )

    for fileName in sorted(os.listdir(args.dataset_path + "vgg_flower/" + 'jpg')):
        label = int(labels[int(fileName[7:11])-1])
        dataset = split_rev[label]
        all_results["metadataset_vgg_flower_"+dataset]['data'].append('vgg_flower/jpg/'+fileName)
        all_results["metadataset_vgg_flower_"+dataset]['targets'].append(all_results["metadataset_vgg_flower_"+dataset]['dataset_targets'][label])

    for dataset in ['train','validation','test']:
        all_results["metadataset_vgg_flower_"+dataset]['num_elements_per_class']=all_results["metadataset_vgg_flower_"+dataset]['num_classes']*[0]
        for i in all_results["metadataset_vgg_flower_"+dataset]['targets']:
            all_results["metadataset_vgg_flower_"+dataset]['num_elements_per_class'][i]+= 1    
        print("Done for metadataset_vgg_flower_" + dataset + " with " + str(all_results["metadataset_vgg_flower_"+dataset]['num_classes']) + " classes ")

### generate data for quickdraw
if 'quickdraw' in available_datasets:
    all_samples_path = args.dataset_path + "quickdraw/"+'all_samples/'
    with open("./datasets/metadatasets/quickdraw/"+"quickdraw_splits.json") as jsonFile:
            split = json.load(jsonFile)
            jsonFile.close()
    for dataset,splitName in [("train","train"),("validation","valid"),("test","test")]:
        class_count = 0
        directories = os.listdir(args.dataset_path + "quickdraw/")
        result = {"data":[], "targets":[], "name":"metadataset_quickdraw_" + dataset, "num_classes":0, "name_classes":[], "num_elements_per_class": []}
        for class_name in split[splitName]:
            samples = np.load(args.dataset_path + "quickdraw/"+class_name +'.npy')
            result['num_elements_per_class'].append(samples.shape[0])
            result['num_classes'] +=1
            result['name_classes'].append(class_name)
            for i in range(samples.shape[0]):
                class_path = all_samples_path+class_name+'/'
                sample_path = os.path.join('quickdraw/all_samples/'+class_name+'/', str(i)+'.JPEG')
                result['data'].append(sample_path)
                result['targets'].append(class_count)
            class_count += 1
        all_results["metadataset_quickdraw_" + dataset] = result
        print("Done for metadatasets_quickdraw_" + dataset + " with " + str(result['num_classes']) + " classes ")



### generate data for traffic_sign
if 'GTSRB' in available_datasets:
    with open('./datasets/metadatasets/traffic_signs/'+"traffic_sign_splits.json") as jsonFile:
            split = json.load(jsonFile)
            jsonFile.close()
    dataset = 'test'
    directories = sorted(os.listdir(args.dataset_path + "GTSRB/Final_Training/Images/"))
    result = {"data":[], "targets":[], "name":"metadataset_traffic_signs_" + dataset, "num_classes":0, "name_classes":[], "num_elements_per_class": []}
    for class_dir in directories:
        filenames = os.listdir(args.dataset_path + "GTSRB/Final_Training/Images/"+class_dir)
        class_target = int(class_dir)
        result['name_classes'].append(split['test'][result['num_classes']])
        result['num_classes'] +=1
        result['num_elements_per_class'].append(len(filenames))
        for filename in filenames:
            result['data'].append("GTSRB/Final_Training/Images/"+class_dir+'/'+filename)
            result['targets'].append(class_target)
    all_results['metadataset_traffic_signs_'+dataset] = result
    print('Done for metadataset_traffic_signs_' + dataset +'with '+str(result['num_classes'])+' classes and '+str(np.sum(np.array(result['num_elements_per_class']))) +' samples')

if 'audioset' in available_datasets:
    ### generate data for quickdraw
    for dataset in ['train', 'test']:
        result = {"data":[], "targets":[], "name":"audioset_" + dataset, "num_classes":0, "name_classes":[], "num_elements_per_class": []}
        path = os.path.join(args.dataset_path, 'audioset', 'processed', dataset)
        class_num_tracker = {}
        num_classes = 0
        for filename in os.listdir(path):
            result['data'].append(os.path.join('audioset', 'processed', dataset, filename))
            targets = list(map(int, filename.replace('.pt','').split('_')[1:]))
            result['targets'].append(targets)
            for c in targets:
                if c in class_num_tracker.keys():
                    class_num_tracker[c]+=1
                else:
                    class_num_tracker[c] = 1
            num_classes = max(num_classes, max(targets))
        result['num_elements_per_class'] = list(dict(sorted(class_num_tracker.items())).values())
        result['num_classes'] = num_classes+1
        all_results["audioset_" + dataset] = result
        print("Done for audioset_" + dataset + " with " + str(result['num_classes']) + " classes and " + str(len(result["data"])) + " samples (" + str(len(result["targets"])) + ")")


def get_data_source_metaalbum(labels, path, source, SET):
    L_classes =[]
    for x in labels['CATEGORY']:
        cl= os.path.join(path ,str(x))
        if cl[10:] not in L_classes:
            L_classes.append(cl[10:])
    train_dic = {'data': [], 'targets' : [] , 'name': 'metaalbum_'+source,'name_classes' : L_classes,'num_elements_per_class':[0]*len(L_classes), 'num_classes': len(L_classes)}
    for i in range(len(labels['FILE_NAME'])):
        train_dic['data'].append(os.path.join(path, source, 'images', labels['FILE_NAME'][i]))
        class_id = L_classes.index(os.path.join(path, str(labels['CATEGORY'][i]))[10:]) ### PB conflict with others here
        train_dic['targets'].append(class_id)
        train_dic['num_elements_per_class'][class_id]+=1
    return train_dic

def get_labels(SET):
    l = os.listdir(os.path.join(args.dataset_path, 'MetaAlbum', SET))
    labels =[]
    for source in l:
        if os.path.isdir(os.path.join(args.dataset_path,'MetaAlbum' ,SET, source)):
            labels.append([pd.read_csv(os.path.join(args.dataset_path ,'MetaAlbum', SET, source ,'labels.csv')), source])
    return labels

def get_data_metaalbum(SET):
    labels = get_labels(SET)
    l_data = []
    sources =[]
    for x in labels:
        data = get_data_source_metaalbum(labels = x[0],path = os.path.join('MetaAlbum' , SET,) , source =x[1], SET=SET ) 
        l_data.append(data)
        sources.append(x[1])
    return l_data, sources

def merge_data(list_data):
    train_dic = {'data': [], 'targets' : [] , 'name': 'metaalbum','name_classes' : [],'num_elements_per_class':[], 'num_classes': 0}
    for data in list_data:
        train_dic['data']+=data['data']
        train_dic['targets']+=[x + train_dic['num_classes'] for x in data['targets']]
        train_dic['name_classes']+=data['name_classes']
        train_dic['num_elements_per_class']+=data['num_elements_per_class']
        train_dic['num_classes']+=data['num_classes']
    return train_dic #{'TRAIN': train_dic}

if 'MetaAlbum' in available_datasets:
    for set_size in ['Micro', 'Mini', 'Extended']:
        data, sources = get_data_metaalbum('Set0_'+set_size)
        for i,source in enumerate(sources):
            all_results['metaalbum_'+source] = data[i]
            print("Done for "+'metaalbum_' + source + " with " + str(data[i]['num_classes']) + " classes and " + str(len(data[i]["data"])) + " samples (" + str(len(data[i]["targets"])) + ")")
        full_data = merge_data(data)
        all_results['metaalbum_'+set_size] = full_data
        print("Done for metaalbum_" + set_size+ " with " + str(full_data['num_classes']) + " classes and " + str(len(full_data["data"])) + " samples (" + str(len(full_data["targets"])) + ")")



f = open(args.dataset_path + "datasets.json", "w")
f.write(json.dumps(all_results))
f.close()
