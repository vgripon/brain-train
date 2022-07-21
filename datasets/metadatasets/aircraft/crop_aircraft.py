import numpy as np
import json
import os
import collections
from tqdm import tqdm
import sys
from PIL import Image


dataset_path = str(sys.argv[1])

def crop(img, xmin, ymin, xmax , ymax):
    arr = np.array(img)
    arr = arr[ymin: ymax,xmin:xmax]
    img_c = Image.fromarray(arr)
    return img_c

def make_new_files():
    with open(dataset_path + 'fgvc-aircraft-2013b/data/images_box.txt') as f:
        lines = f.readlines()
    L=[]
    for x in lines:
        x=x[:-1].split(' ')
        L.append([int(i) for i in x ])
    for x in tqdm(L):
        try:
            file = dataset_path+'fgvc-aircraft-2013b/data/images/'+str(x[0])+'.jpg'
            img = Image.open(file)
            img_c = crop(img, x[1], x[2], x[3] , x[4])
            img_c = img_c.save(dataset_path + 'fgvc-aircraft-2013b/data/images_cropped/'+str(x[0])+'.jpg')
        except:
            #print(x[0],' not found')
            pass