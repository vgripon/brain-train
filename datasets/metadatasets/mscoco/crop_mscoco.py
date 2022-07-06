import sys
dataset_path = str(sys.argv[1])
print('AS SUCH THE CROPPING WILL BE UNEFFECTIVE (READ BELOW) \n this may take between one and two hours')
print('As a security the names of files were changed \n -----> IF you really intend to crop the MSCOCO folder (2h) you should : <----')
print('1) replace "cropped" by "cropped_imgs" at line 27 and 28 in RUN_ONLY_ONCE.sh and at line 84 of crop_mscoco.py')
print('2) replace "cropped_mscoco_test.json" by "cropped_mscoco.json" at line 103 of crop_mscoco.py')
print('3) run RUN_ONLY_ONCE.sh completely (quite long)')
import numpy as np
import torch
import matplotlib.pyplot as plt
import json
import os
from PIL import Image
import collections
from tqdm import tqdm 


def split_fn(json_path):
    with open(json_path) as jsonFile:
        split = json.load(jsonFile)
        jsonFile.close()
    return split

def generate_mscoco_examples(metadata, paths, box_scale_ratio=1.2):
    subset_data = {'data': [], 'targets' : [] , 'name_classes' : []}
    """Generates MSCOCO examples."""
    if box_scale_ratio < 1.0:
        raise ValueError('Box scale ratio must be greater or equal to 1.0.')

    image_dir = dataset_path+'metadatasets/mscoco/train2017/'
    print('here', image_dir)
    annotations = split_fn(dataset_path+'metadatasets/mscoco/instances_train2017.json')

    class_name_to_category_id = {
        category['name']: category['id']
        for category in annotations['categories']
    }
    coco_id_to_label = {
        class_name_to_category_id[class_name]: label
        for label, class_name in enumerate(metadata['class_names'])
    }

    label_to_annotations = collections.defaultdict(list)
    for annotation in tqdm(annotations['annotations']):
        category_id = annotation['category_id']
        if category_id in coco_id_to_label:
            label_to_annotations[coco_id_to_label[category_id]].append(annotation)

    print('number of classes', len(metadata['class_names']))
    for label, class_name in enumerate(tqdm(metadata['class_names'])):
        for annotation in label_to_annotations[label]:
            image_path = image_dir + f"{annotation['image_id']:012d}.jpg"

            # The bounding box is represented as (x_topleft, y_topleft, width, height)
            bbox = annotation['bbox']
            with open(image_path, 'rb') as f:
                image = Image.open(f)

                # The image shape is [?, ?, 3] and the type is uint8.
                image = image.convert(mode='RGB')
            image_w, image_h = image.size

            x, y, w, h = bbox
            x = x - 0.5 * w * (box_scale_ratio - 1.0)
            y = y - 0.5 * h * (box_scale_ratio - 1.0)
            w = w * box_scale_ratio
            h = h * box_scale_ratio

            # Convert half-integer to full-integer representation.
            # The Python Imaging Library uses a Cartesian pixel coordinate system,
            # with (0,0) in the upper left corner. Note that the coordinates refer
            # to the implied pixel corners; the centre of a pixel addressed as
            # (0, 0) actually lies at (0.5, 0.5). Since COCO uses the later
            # convention and we use PIL to crop the image, we need to convert from
            # half-integer to full-integer representation.
            xmin = max(int(round(x - 0.5)), 0)
            ymin = max(int(round(y - 0.5)), 0)
            xmax = min(int(round(x + w - 0.5)) + 1, image_w)
            ymax = min(int(round(y + h - 0.5)) + 1, image_h)
            image = image.crop((xmin, ymin, xmax, ymax))
            crop_width, crop_height = image.size
            if crop_width <= 0 or crop_height <= 0:
                raise ValueError('crops are not valid.')
            filename = dataset_path+"metadatasets/mscoco/cropped/"+ f"{annotation['image_id']:012d}.jpg"
            image.save(filename, "JPEG")
            subset_data['data'].append(filename)
            subset_data['targets'].append(label)
        subset_data['name_classes'].append(class_name)
    subset_data['num_classes'] = label+1
    return subset_data
      
metadata = {}
split = split_fn('mscoco_splits.json')                                   #split from metadataset
data ,num_classes, num_elts = {},{},{}
split = {"validation" if k == 'valid' else k:v for k,v in split.items()}  #adapt to metadatasets standarts
for index_subset, subset in enumerate(split.keys()):
    metadata['class_names'] = split[subset]
    try: 
        data[subset] = generate_mscoco_examples(metadata, paths = '', box_scale_ratio=1.2) #fill the folder with jpg images and get the data dictionary
        print(subset, 'sucesss')
    except Exception as e: print(e, subset, 'if the subet is "train" is empty it is normal this subset is not used in metadataset-mscoco')
        
with open(dataset_path+'metadatasets/mscoco/cropped_mscoco_test.json', 'w') as fp:
    json.dump(data, fp)