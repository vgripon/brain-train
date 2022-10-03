import sys
sys.path.insert(0,'../../../')
from args import args
from PIL import Image
import os
import numpy as np
from collections import defaultdict
import tqdm 
all_results = {}

<<<<<<< HEAD
all_samples_path = os.path.join(args.dataset_path , "quickdraw",'all_samples4')
=======
all_samples_path = os.path.join(args.dataset_path , "quickdraw",'all_samples3')
>>>>>>> main
if not os.path.exists(all_samples_path):
    os.mkdir(all_samples_path)
for idx,class_npy in tqdm.tqdm(enumerate([f for f in os.listdir(os.path.join(args.dataset_path , "quickdraw")) if f.endswith('npy')])):
    class_name = class_npy[:-4]
    print('processing class: ',class_name)
    class_path = os.path.join(all_samples_path,class_name)
    if not os.path.exists(class_path):
        os.mkdir(class_path)
        samples = np.load(os.path.join(args.dataset_path , "quickdraw",class_npy))
        samples  = samples.astype(np.uint8)
        samples = samples.reshape(-1, 28,28)*255
        for i in range(samples.shape[0]):
<<<<<<< HEAD
            sample = samples[i]
            sample = Image.fromarray(samples[i])
            sample = sample.convert('RGB')
=======
            sample = Image.fromarray(samples[i].reshape(28,28))
>>>>>>> main
            sample.save(os.path.join(class_path, str(i)+'.JPEG'))