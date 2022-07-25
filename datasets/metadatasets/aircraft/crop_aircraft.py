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
    with open('images_box.txt') as f:
        lines = f.readlines()
    print('bouding box file size : ',len(lines))
    L=[]
    for x in lines:
        x=x[:-1].split(' ')
        L.append([i for i in x ])
    l_files = os.listdir(dataset_path+'fgvc-aircraft-2013b/data/images/')
    print('files number : ', len(l_files))
    for x in tqdm(L):
        if x[0]+'.jpg' in l_files:
            file = dataset_path+'fgvc-aircraft-2013b/data/images/'+x[0]+'.jpg'
            img = Image.open(file)
            img_c = crop(img, int(x[1]), int(x[2]), int(x[3]) , int(x[4]))
            img_c = img_c.save(dataset_path + 'fgvc-aircraft-2013b/data/images_cropped/'+str(x[0])+'.jpg')
        else:
            print(str(x[0])+'.jpg', 'not found')


if __name__ == '__main__':
    make_new_files()