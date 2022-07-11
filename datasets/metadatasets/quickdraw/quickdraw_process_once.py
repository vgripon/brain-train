from args import args
from PIL import Image
import os
import numpy as np
from collections import defaultdict

all_results = {}

all_samples_path = args.dataset_path + "quickdraw/"+'all_samples/'
if not os.path.exists(all_samples_path):
    os.mkdir(all_samples_path)
for idx,class_npy in tqdm.tqdm(enumerate([f for f in os.listdir(args.dataset_path + "quickdraw/") if f.endswith('npy')])):
    class_name = class_npy[:-4]
    print('processing class: ',class_name)
    class_path = all_samples_path+class_name+'/'
    if not os.path.exists(class_path):
        os.mkdir(class_path)
        samples = np.load(os.path.join(args.dataset_path + "quickdraw/",class_npy))
        for i in range(samples.shape[0]):
            sample = Image.fromarray(samples[i])
            sample.save(os.path.join(class_path, str(i)+'.JPEG'))